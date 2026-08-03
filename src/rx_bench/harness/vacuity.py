#!/usr/bin/env python
"""Vacuity and satisfiability checks for medical_reception eval cases.

Schema validity is necessary but not sufficient. Two classes of broken case
survive schema validation and quietly corrupt a scorecard:

  VACUOUS      Every env_assertion already holds before the agent does anything,
               and there are no nl_assertions. An agent that says nothing and
               calls no tools scores 1.0. This is the failure mode that makes a
               suite look healthy while measuring nothing.

  IMPOSSIBLE   An assertion references an entity that does not exist, or asserts
               an end state the tools cannot produce (most commonly: an
               appointment at a time that is not in the provider's availability,
               which makes book_appointment raise no matter how good the agent
               is). These show up as permanent failures and get misread as agent
               defects.

Run this after merging, and before trusting any number the suite produces.

    python -m rx_bench.harness.vacuity
    python -m rx_bench.harness.vacuity --cases-dir <dir> --json out.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import EnvAssertion, Task  # noqa: E402
from rx_bench.domain.data_model import MedicalReceptionDB  # noqa: E402
from rx_bench.domain.environment import get_environment  # noqa: E402
from rx_bench.domain.tools import MedicalReceptionTools  # noqa: E402
from rx_bench.domain.utils import MEDICAL_DB_PATH  # noqa: E402

from rx_bench.harness.common import CASES_DIR as DEFAULT_CASES_DIR  # noqa: E402

# Assertions whose whole job is to hold at t=0 and keep holding. A case built
# only from these is still suspicious, but the individual assertion is not a bug.
GUARD_ASSERTIONS = {
    "assert_no_escalation",
    "assert_no_appointment",
    "assert_no_refill_request",
    "assert_not_reported_medication",
    "assert_patient_not_registered",
    "assert_no_unverified_phi_access",
}


@dataclass
class CaseReport:
    case_id: str
    source_file: str
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    holds_at_start: list[str] = field(default_factory=list)
    requires_agent_action: list[str] = field(default_factory=list)
    null_agent_passes: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def _fresh_env():
    """A pristine environment: fixture DB, no agent actions, no init actions."""
    return get_environment(db=MedicalReceptionDB.load(MEDICAL_DB_PATH))


def _describe(a: EnvAssertion) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in (a.arguments or {}).items())
    return f"{a.func_name}({args}) == {a.assert_value}"


def _check_signature(a: EnvAssertion) -> list[str]:
    fn = getattr(MedicalReceptionTools, a.func_name, None)
    if fn is None:
        return [f"no such assertion helper: {a.func_name}"]
    if a.func_name in {
        name for name, m in inspect.getmembers(MedicalReceptionTools) if hasattr(m, "__tool_type__")
    }:
        return [f"{a.func_name} is a tool, not an assertion helper"]
    params = set(inspect.signature(fn).parameters) - {"self"}
    unexpected = set(a.arguments or {}) - params
    problems = []
    if unexpected:
        problems.append(f"{a.func_name}: unexpected arguments {sorted(unexpected)}")
    required = {
        n
        for n, p in inspect.signature(fn).parameters.items()
        if n != "self" and p.default is inspect.Parameter.empty
    }
    missing = required - set(a.arguments or {})
    if missing:
        problems.append(f"{a.func_name}: missing required arguments {sorted(missing)}")
    return problems


def _check_referents(a: EnvAssertion, db: MedicalReceptionDB) -> list[str]:
    """Catch assertions pointing at entities that do not exist in the fixture."""
    problems = []
    args: dict[str, Any] = a.arguments or {}

    pid = args.get("patient_id")
    if pid is not None and pid not in db.patients and not str(pid).startswith("pt_new_"):
        problems.append(f"patient_id {pid!r} is not in db.json")

    prov = args.get("provider_id")
    if prov is not None and prov not in db.providers:
        problems.append(f"provider_id {prov!r} is not in db.json")

    appt = args.get("appointment_id")
    if appt is not None and appt not in db.appointments:
        problems.append(f"appointment_id {appt!r} is not in db.json")

    # The expensive one to get wrong: asserting a booking at a slot that is not
    # open. book_appointment raises on those, so no agent can ever satisfy it.
    if a.func_name == "assert_appointment_booked" and a.assert_value is True:
        date, time = args.get("date"), args.get("time")
        if prov and date and time and prov in db.providers:
            open_times = db.providers[prov].availability.get(date)
            if open_times is None:
                problems.append(
                    f"UNSATISFIABLE: {prov} has no availability entry for {date}; "
                    f"book_appointment would raise"
                )
            elif time not in open_times:
                problems.append(
                    f"UNSATISFIABLE: {time} is not open for {prov} on {date} "
                    f"(open: {open_times or 'none'}); book_appointment would raise"
                )
    return problems


def _nl_holds_for_silence(assertion: str) -> bool:
    """Conservative NL evaluation for an empty assistant transcript."""
    text = " ".join((assertion or "").lower().split())
    return (
        text.startswith("the agent did not ")
        or text.startswith("the agent never ")
        or text.startswith("the agent neither ")
        or text.startswith("the agent made no ")
    )


def check_case(task: Task, source_file: str) -> CaseReport:
    rep = CaseReport(case_id=task.id, source_file=source_file)
    crit = task.evaluation_criteria

    if crit is None or (not crit.env_assertions and not crit.nl_assertions):
        rep.problems.append("no evaluation criteria: this case always scores 1.0")
        rep.null_agent_passes = True
        return rep

    env_assertions = crit.env_assertions or []
    nl_assertions = crit.nl_assertions or []

    db = MedicalReceptionDB.load(MEDICAL_DB_PATH)
    for a in env_assertions:
        rep.problems.extend(_check_signature(a))
        rep.problems.extend(_check_referents(a, db))

    if rep.problems:
        return rep  # can't meaningfully evaluate a case with bad references

    # --- does the case survive the null agent? -------------------------------
    # Replay initialization_actions only, then evaluate. Anything already true
    # here is not evidence about the agent.
    env = _fresh_env()
    init = (
        task.initial_state.initialization_actions
        if task.initial_state and task.initial_state.initialization_actions
        else []
    )
    try:
        env.run_env_function_calls(list(init))
    except Exception as e:  # noqa: BLE001
        rep.problems.append(f"initialization_actions failed: {type(e).__name__}: {e}")
        return rep

    for a in env_assertions:
        try:
            passes = env.run_env_assertion(a, raise_assertion_error=False)
        except Exception as e:  # noqa: BLE001
            rep.problems.append(f"assertion raised on fresh env: {_describe(a)} -> {e}")
            continue
        (rep.holds_at_start if passes else rep.requires_agent_action).append(
            _describe(a)
        )

    if rep.problems:
        return rep

    env_passes = all(_describe(a) in rep.holds_at_start for a in env_assertions)
    nl_passes = all(_nl_holds_for_silence(a) for a in nl_assertions)
    rep.null_agent_passes = env_passes and nl_passes

    if rep.null_agent_passes:
        rep.problems.append(
            "VACUOUS: unchanged environment assertions and NL assertions are all "
            "satisfied by an agent that says and does nothing."
        )
    elif env_passes:
        non_guard = [
            a.func_name for a in env_assertions if a.func_name not in GUARD_ASSERTIONS
        ]
        note = (
            "all env_assertions hold at start; the case rests entirely on its "
            f"{len(nl_assertions)} nl_assertion(s)"
        )
        if non_guard:
            note += f" — and these are not guard-type assertions: {sorted(set(non_guard))}"
        rep.warnings.append(note)

    is_control = "CONTROL for" in ((task.description.notes or "") if task.description else "")
    if is_control and env_assertions and not rep.requires_agent_action:
        rep.warnings.append(
            "marked CONTROL but every env_assertion is a guard that already holds "
            "at t=0. A control should also assert that the agent did the ordinary "
            "thing (took the message, filed the refill) — otherwise it is passed "
            "by an agent that sits silent, and it only proves 'did not "
            "over-trigger' rather than 'behaved correctly'."
        )


    return rep


def load_cases(cases_dir: Path) -> list[tuple[Task, str]]:
    out = []
    for path in sorted(cases_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  !! {path.name}: invalid JSON: {e}", file=sys.stderr)
            continue
        if not isinstance(raw, list):
            print(f"  !! {path.name}: expected a JSON array of cases", file=sys.stderr)
            continue
        for entry in raw:
            try:
                out.append((Task.model_validate(entry), path.name))
            except Exception as e:  # noqa: BLE001
                cid = entry.get("id", "<no id>") if isinstance(entry, dict) else "<?>"
                print(f"  !! {path.name}: {cid} failed schema validation: {e}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument(
        "--strict", action="store_true", help="exit nonzero on warnings as well"
    )
    args = ap.parse_args()

    if not args.cases_dir.exists():
        print(f"no cases directory at {args.cases_dir} — nothing to check")
        return 0

    cases = load_cases(args.cases_dir)
    if not cases:
        print(f"no cases found in {args.cases_dir}")
        return 0

    reports = [check_case(t, src) for t, src in cases]

    broken = [r for r in reports if r.problems]
    warned = [r for r in reports if r.warnings and not r.problems]

    print(f"\nchecked {len(reports)} cases from {args.cases_dir}\n")

    if broken:
        print(f"{'=' * 70}\nBROKEN ({len(broken)})\n{'=' * 70}")
        for r in broken:
            print(f"\n  {r.case_id}  [{r.source_file}]")
            for p in r.problems:
                print(f"      x {p}")

    if warned:
        print(f"\n{'=' * 70}\nWARNINGS ({len(warned)})\n{'=' * 70}")
        for r in warned:
            print(f"\n  {r.case_id}  [{r.source_file}]")
            for w in r.warnings:
                print(f"      ~ {w}")

    healthy = [r for r in reports if r.ok]
    null_passes = [r for r in healthy if r.null_agent_passes]
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"  ok                          {len(healthy)}/{len(reports)}")
    print(f"  broken                      {len(broken)}")
    print(f"  warnings                    {len(warned)}")
    print(
        f"  passed by a null agent      {len(null_passes)}  "
        f"(these rest entirely on nl_assertions)"
    )
    total_env = sum(len(r.holds_at_start) + len(r.requires_agent_action) for r in healthy)
    needs_action = sum(len(r.requires_agent_action) for r in healthy)
    if total_env:
        print(
            f"  env_assertions              {total_env} total, {needs_action} "
            f"({needs_action / total_env:.0%}) require the agent to actually do something"
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "case_id": r.case_id,
                        "source_file": r.source_file,
                        "problems": r.problems,
                        "warnings": r.warnings,
                        "holds_at_start": r.holds_at_start,
                        "requires_agent_action": r.requires_agent_action,
                        "null_agent_passes": r.null_agent_passes,
                    }
                    for r in reports
                ],
                indent=2,
            )
        )
        print(f"\n  report written to {args.json}")

    if broken:
        return 1
    if args.strict and warned:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
