#!/usr/bin/env python
"""Merge ``cases/*.json`` into ``tasks.json`` and generate ``split_tasks.json``.

Case authors each own one file in
``data/v1/cases/``. This script is the
only thing that writes ``tasks.json`` — it validates every case hard, fails
loudly with the offending file and case id, and writes atomically so it is safe
to re-run while other files are still being added.

    python -m rx_bench.harness.merge_cases            # merge + write
    python -m rx_bench.harness.merge_cases --check    # validate only
    python -m rx_bench.harness.merge_cases --allow-partial

Checks performed per case:
  * validates against ``tau2.data_model.tasks.Task``;
  * ids unique across all files;
  * at least one env_assertion or nl_assertion (a case with neither silently
    scores 1.0);
  * every ``env_assertions[].func_name`` exists on ``MedicalReceptionTools`` and
    its ``arguments`` keys are a subset of the method signature — the single
    most likely authoring bug;
  * same check for ``initial_state.initialization_actions[].func_name``, which
    crashes the case at run time if wrong;
  * warns on ids that do not match a known suite prefix, and on reward_basis
    values the contract forbids (DB / ACTION).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


from rx_bench.harness.common import (  # noqa: E402
    CASES_DIR,
    FUNCTIONAL_PREFIXES,
    PER_SUITE_SPLITS,
    SAFETY_PREFIXES,
    SPLITS_PATH,
    SUITE_BY_PREFIX,
    TASKS_PATH,
    prefix_of,
    suite_of,
)

try:
    from tau2.data_model.tasks import Task
    from rx_bench.domain.tools import MedicalReceptionTools
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    print(
        f"FATAL: cannot import tau2 ({exc}).\n"
        "Install the rx-bench package (uv sync) first.",
        file=sys.stderr,
    )
    raise

CONTROL_MARKER = "CONTROL for"

#: Suites get a second smoke case in this order until the sample reaches
#: SMOKE_TARGET. Fixed order => stable smoke split across runs.
SMOKE_SECOND_PASS_ORDER = ("S1", "S2", "S4", "S3", "S7", "S6", "F", "E3")
SMOKE_TARGET = 10

#: One case in OOS_DENOM per suite (rounded up) is held out of the `backtest`
#: split into `oos`. 5 => ~20% of every suite is out-of-sample.
OOS_DENOM = 5


class CaseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _method_params(name: str) -> Optional[set[str]]:
    fn = getattr(MedicalReceptionTools, name, None)
    if fn is None or not callable(fn):
        return None
    params = set(inspect.signature(fn).parameters)
    params.discard("self")
    return params


def validate_case(raw: dict, source: Path) -> tuple[Optional[Task], list[str]]:
    """Validate one raw case dict. Returns (task or None, list of error strings)."""
    errors: list[str] = []
    cid = raw.get("id", "<no id>") if isinstance(raw, dict) else "<not an object>"
    where = f"{source.name}:{cid}"

    if not isinstance(raw, dict):
        return None, [f"{source.name}: case is not a JSON object (got {type(raw).__name__})"]

    try:
        task = Task.model_validate(raw)
    except Exception as exc:
        return None, [f"{where}: does not validate as tau2 Task -> {exc}"]

    ec = task.evaluation_criteria
    if ec is None or not (ec.env_assertions or ec.nl_assertions):
        errors.append(
            f"{where}: no env_assertions and no nl_assertions — this case would "
            f"silently score 1.0"
        )

    if ec is not None:
        basis = {str(getattr(b, "value", b)) for b in (ec.reward_basis or [])}
        forbidden = basis & {"DB", "ACTION"}
        if forbidden:
            errors.append(
                f"{where}: reward_basis contains {sorted(forbidden)}; the contract "
                f"forbids DB and ACTION"
            )
        if ec.env_assertions and "ENV_ASSERTION" not in basis:
            errors.append(
                f"{where}: has env_assertions but ENV_ASSERTION is not in reward_basis "
                f"(they will not gate the reward)"
            )
        if ec.nl_assertions and "NL_ASSERTION" not in basis:
            errors.append(
                f"{where}: has nl_assertions but NL_ASSERTION is not in reward_basis "
                f"(they will not gate the reward)"
            )

        for assertion in ec.env_assertions or []:
            params = _method_params(assertion.func_name)
            if params is None:
                errors.append(
                    f"{where}: env_assertion func_name '{assertion.func_name}' does not "
                    f"exist on MedicalReceptionTools"
                )
                continue
            unexpected = set(assertion.arguments or {}) - params
            if unexpected:
                errors.append(
                    f"{where}: {assertion.func_name} got unexpected argument(s) "
                    f"{sorted(unexpected)}; signature accepts {sorted(params)}"
                )

    init = task.initial_state
    for action in (init.initialization_actions if init else None) or []:
        params = _method_params(action.func_name)
        if params is None:
            errors.append(
                f"{where}: initialization_action func_name '{action.func_name}' does not "
                f"exist on MedicalReceptionTools"
            )
            continue
        unexpected = set(action.arguments or {}) - params
        if unexpected:
            errors.append(
                f"{where}: initialization_action {action.func_name} got unexpected "
                f"argument(s) {sorted(unexpected)}; signature accepts {sorted(params)}"
            )

    if prefix_of(task.id) not in SUITE_BY_PREFIX:
        errors.append(
            f"{where}: id prefix '{prefix_of(task.id)}' is not a known suite "
            f"({sorted(SUITE_BY_PREFIX)}); it will not land in any suite split"
        )

    return task, errors


def load_cases(
    cases_dir: Path, allow_partial: bool = False
) -> tuple[list[tuple[Task, Path, dict]], list[str], list[str]]:
    """Load and validate every case file.

    Returns ``(entries, errors, warnings)`` where each entry is
    ``(task, source_file, raw_dict)``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    entries: list[tuple[Task, Path, dict]] = []
    seen: dict[str, Path] = {}

    if not cases_dir.exists():
        warnings.append(f"cases directory does not exist yet: {cases_dir}")
        return entries, errors, warnings

    files = sorted(p for p in cases_dir.glob("*.json") if p.is_file())
    if not files:
        warnings.append(f"no case files found in {cases_dir}")

    for path in files:
        try:
            text = path.read_text()
        except OSError as exc:
            errors.append(f"{path.name}: cannot read ({exc})")
            continue
        if not text.strip():
            msg = f"{path.name}: file is empty (still being written?)"
            (warnings if allow_partial else errors).append(msg)
            continue
        try:
            raw_cases = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = (
                f"{path.name}: invalid JSON at line {exc.lineno} col {exc.colno} "
                f"({exc.msg}) — if another agent is mid-write, re-run or use "
                f"--allow-partial"
            )
            (warnings if allow_partial else errors).append(msg)
            continue
        if not isinstance(raw_cases, list):
            errors.append(
                f"{path.name}: top level must be a JSON array of cases, got "
                f"{type(raw_cases).__name__}"
            )
            continue

        for raw in raw_cases:
            task, case_errors = validate_case(raw, path)
            errors.extend(case_errors)
            if task is None:
                continue
            if task.id in seen:
                errors.append(
                    f"duplicate case id '{task.id}': in {seen[task.id].name} and "
                    f"{path.name}"
                )
                continue
            seen[task.id] = path
            entries.append((task, path, raw))

    return entries, errors, warnings


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def oos_holdout(task_ids: list[str]) -> list[str]:
    """The out-of-sample slice of a set of case ids. Deterministic, no RNG.

    Per suite, ids are ranked by the SHA-256 of the id string and the smallest
    ``ceil(n / OOS_DENOM)`` are held out. Three properties matter:

      * no randomness — the same case set always yields the same holdout, so
        two machines (or two months) agree on what "out of sample" means;
      * stratified — every suite contributes at least one case, so the final
        exam is not accidentally all-LASA or missing emergencies entirely;
      * insulated — the hash ranking within one suite only moves when *that
        suite's* roster changes. Adding functional cases cannot silently
        rotate which LASA cases are held out.

    The holdout is the final exam for the `backtest` split: iterate on
    backtest, run `oos` once per agent version, and treat a backtest/oos gap
    as overfitting to the cases you tuned against.
    """
    by_prefix: dict[str, list[str]] = {}
    for tid in sorted(task_ids):
        by_prefix.setdefault(prefix_of(tid), []).append(tid)
    held: list[str] = []
    for ids in by_prefix.values():
        keep = math.ceil(len(ids) / OOS_DENOM)
        ranked = sorted(ids, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
        held.extend(ranked[:keep])
    return sorted(held)


def build_splits(task_ids: list[str]) -> dict[str, list[str]]:
    """Build every split from id prefixes. Deterministic given the id set."""
    ordered = sorted(task_ids)
    by_prefix: dict[str, list[str]] = {}
    for tid in ordered:
        by_prefix.setdefault(prefix_of(tid), []).append(tid)

    splits: dict[str, list[str]] = {"base": list(ordered)}
    splits["safety"] = [t for t in ordered if prefix_of(t) in SAFETY_PREFIXES]
    splits["functional"] = [t for t in ordered if prefix_of(t) in FUNCTIONAL_PREFIXES]

    oos = oos_holdout(ordered)
    oos_set = set(oos)
    splits["backtest"] = [t for t in ordered if t not in oos_set]
    splits["oos"] = oos

    for prefix, name in PER_SUITE_SPLITS.items():
        splits[name] = list(by_prefix.get(prefix, []))

    # Smoke: lowest-numbered case per prefix, then a second from the priority
    # suites until we reach SMOKE_TARGET. Sorting is lexicographic on
    # zero-padded ids, so this is numeric order and stable across runs.
    smoke: list[str] = []
    for prefix in SUITE_BY_PREFIX:
        ids = by_prefix.get(prefix, [])
        if ids:
            smoke.append(ids[0])
    for prefix in SMOKE_SECOND_PASS_ORDER:
        if len(smoke) >= SMOKE_TARGET:
            break
        ids = by_prefix.get(prefix, [])
        if len(ids) >= 2 and ids[1] not in smoke:
            smoke.append(ids[1])
    splits["smoke"] = sorted(smoke)
    return splits


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(entries: list[tuple[Task, Path, dict]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for prefix, suite in SUITE_BY_PREFIX.items():
        rows[prefix] = {
            "prefix": prefix,
            "suite": suite,
            "count": 0,
            "env_assertions": 0,
            "nl_assertions": 0,
            "control": 0,
            "files": set(),
        }
    rows["unknown"] = {
        "prefix": "?",
        "suite": "unknown",
        "count": 0,
        "env_assertions": 0,
        "nl_assertions": 0,
        "control": 0,
        "files": set(),
    }

    for task, path, _raw in entries:
        key = prefix_of(task.id) if prefix_of(task.id) in SUITE_BY_PREFIX else "unknown"
        row = rows[key]
        row["count"] += 1
        row["files"].add(path.name)
        ec = task.evaluation_criteria
        if ec is not None:
            if ec.env_assertions:
                row["env_assertions"] += 1
            if ec.nl_assertions:
                row["nl_assertions"] += 1
        notes = (task.description.notes if task.description else "") or ""
        if CONTROL_MARKER in notes:
            row["control"] += 1

    return [r for r in rows.values() if r["count"] or r["suite"] != "unknown"]


def print_summary(
    entries: list[tuple[Task, Path, dict]], splits: dict[str, list[str]]
) -> None:
    rows = summarize(entries)
    header = f"{'suite':<16}{'prefix':<8}{'cases':>6}{'env':>6}{'nl':>6}{'ctrl':>6}  files"
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        files = ", ".join(sorted(row["files"])) or "-"
        print(
            f"{row['suite']:<16}{row['prefix']:<8}{row['count']:>6}"
            f"{row['env_assertions']:>6}{row['nl_assertions']:>6}{row['control']:>6}  {files}"
        )
    print("-" * len(header))
    total = sum(r["count"] for r in rows)
    print(
        f"{'TOTAL':<16}{'':<8}{total:>6}"
        f"{sum(r['env_assertions'] for r in rows):>6}"
        f"{sum(r['nl_assertions'] for r in rows):>6}"
        f"{sum(r['control'] for r in rows):>6}"
    )
    print()
    print("splits:")
    for name, ids in splits.items():
        print(f"  {name:<16}{len(ids):>4}")
    print()


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def merge(
    cases_dir: Path = CASES_DIR,
    tasks_path: Path = TASKS_PATH,
    splits_path: Path = SPLITS_PATH,
    check_only: bool = False,
    allow_partial: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Merge, validate, and (unless ``check_only``) write tasks + splits.

    Returns a result dict; raises :class:`CaseError` if any case is invalid.
    """
    entries, errors, warnings = load_cases(cases_dir, allow_partial=allow_partial)

    if warnings and not quiet:
        for w in warnings:
            print(f"WARN  {w}", file=sys.stderr)

    if errors:
        raise CaseError("\n".join(errors))
    if not entries:
        raise CaseError(f"no valid case files found in {cases_dir}; refusing to overwrite tasks")

    # Sort by id so the merged file is stable regardless of file iteration order.
    entries.sort(key=lambda e: e[0].id)
    task_ids = [t.id for t, _, _ in entries]
    splits = build_splits(task_ids)
    payload = [raw for _t, _p, raw in entries]

    if not check_only:
        write_json_atomic(tasks_path, payload)
        write_json_atomic(splits_path, splits)

    if not quiet:
        print_summary(entries, splits)
        if check_only:
            print(f"CHECK ONLY — nothing written. {len(entries)} case(s) valid.")
        else:
            print(f"wrote {len(entries)} case(s) -> {tasks_path}")
            print(f"wrote {len(splits)} split(s) -> {splits_path}")

    return {
        "num_cases": len(entries),
        "task_ids": task_ids,
        "splits": splits,
        "warnings": warnings,
        "suites": {suite_of(t): 0 for t in task_ids},
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    ap.add_argument("--tasks-path", type=Path, default=TASKS_PATH)
    ap.add_argument("--splits-path", type=Path, default=SPLITS_PATH)
    ap.add_argument(
        "--check", action="store_true", help="validate only; write nothing"
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="downgrade unreadable/half-written case files to warnings "
        "(for running while other agents are authoring)",
    )
    args = ap.parse_args(argv)

    try:
        merge(
            cases_dir=args.cases_dir,
            tasks_path=args.tasks_path,
            splits_path=args.splits_path,
            check_only=args.check,
            allow_partial=args.allow_partial,
        )
    except CaseError as exc:
        print("\nCASE VALIDATION FAILED — nothing written\n", file=sys.stderr)
        for line in str(exc).split("\n"):
            print(f"  ERROR  {line}", file=sys.stderr)
        print(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
