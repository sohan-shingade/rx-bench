#!/usr/bin/env python
"""Turn a tau2 results file into ``scorecard.json`` plus a readable table.

    python -m rx_bench.harness.scorecard <results.json> \
        [--baseline previous_scorecard.json] [--out scorecard.json]

Reports per-suite pass rates, every scorer in :mod:`scorers`, tau2's own
pass^k when it can be computed, and — when a baseline is given — which case ids
newly pass, which newly fail, and the delta on every headline metric.

Newly-failing cases come first and are loud. They are the thing a developer
needs at a glance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


from rx_bench.harness.common import (  # noqa: E402
    MUTANTS_DIR,
    POLICY_PATH,
    is_successful,
    safe_rate,
    sim_actually_ran,
    sim_reward,
    suite_of,
)
from rx_bench.harness.scorers import load_results, run_all  # noqa: E402

BANNER = "=" * 78

#: Metrics that get a delta line in the diff, and how to read them.
#: (path into the scorecard, label, "up_is_good")
HEADLINE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("totals.pass_rate", "overall pass rate", True),
    ("readback.readback_rate", "readback rate", True),
    ("readback.false_confirmation_rate", "FALSE CONFIRMATION rate", False),
    ("readback.silent_guess_rate", "silent guess rate", False),
    ("readback.ceremonial_readback_rate", "ceremonial readback rate", False),
    ("readback.honest_doubt_rate", "honest doubt (flagged) rate", True),
    ("provenance.provenance_violation_count", "provenance violations", False),
    ("provenance.canary_interpretations", "canary interpretations", False),
    ("drug_entities.lasa_substitution_rate", "LASA substitution rate", False),
    ("drug_entities.generic_asr_error_rate", "generic ASR error rate", False),
    ("drug_entities.drug_accuracy", "drug accuracy", True),
    ("control_pairs.paired_task_accuracy", "paired aggregate-task accuracy", True),
    ("control_pairs.control_task_failure_rate", "control task failure rate", False),
    ("control_pairs.positive_task_failure_rate", "positive task failure rate", False),
    ("turn_of_flip.mean_turn_of_flip", "mean turn of flip", True),
    ("ease_of_use.mean_turns_to_resolution", "mean turns to resolution", False),
    ("ease_of_use.total_repeat_request_count", "caller repeat requests", False),
    ("ease_of_use.total_redundant_question_count", "redundant questions", False),
    # Mandated §3.7/§3.8 readbacks. Broken out so that the redundant-question
    # count above cannot be read as punishing the agent for doing readbacks --
    # confirmatory restatements are excluded from it by construction.
    ("ease_of_use.total_confirmatory_readback_questions",
     "confirmatory readbacks (not redundant)", True),
)


def _dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _identify_policy(results: Any) -> dict[str, Any]:
    """Which policy was this run made against? Detects mutants by content hash."""
    info: dict[str, Any] = {"policy_sha256": None, "policy_name": "unknown"}
    try:
        policy = results.info.environment_info.policy or ""
    except Exception:
        return info
    digest = hashlib.sha256(policy.encode()).hexdigest()
    info["policy_sha256"] = digest
    candidates = [("policy.md", POLICY_PATH)]
    if MUTANTS_DIR.exists():
        candidates += [(f"mutant:{p.stem}", p) for p in sorted(MUTANTS_DIR.glob("*.md"))]
    for name, path in candidates:
        try:
            if hashlib.sha256(path.read_text().encode()).hexdigest() == digest:
                info["policy_name"] = name
                break
        except OSError:
            continue
    return info


def _task_set_sha256(results: Any) -> str:
    """Hash the declared task set independent of task and object key order."""
    tasks = []
    for task in getattr(results, "tasks", None) or []:
        raw = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
        tasks.append(raw)
    tasks.sort(key=lambda task: str(task.get("id", "")))
    canonical = json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _per_case(results: Any) -> dict[str, dict[str, Any]]:
    tasks = getattr(results, "tasks", None) or []
    expected_trials = max(1, int(getattr(getattr(results, "info", None), "num_trials", 1)))
    tickets = {
        getattr(t, "id", None): getattr(t, "ticket", None)
        for t in tasks
    }

    def _empty_entry(tid: str) -> dict[str, Any]:
        return {
            "task_id": tid,
            "suite": suite_of(tid),
            "ticket": tickets.get(tid),
            "expected_trials": expected_trials,
            "trials_present": 0,
            "trials": 0,
            "successes": 0,
            "infra_errors": 0,
            "rewards": [],
            "termination_reasons": [],
        }

    # Results.tasks is the declared run scope. Start there rather than deriving
    # cases only from SimulationRun rows: a task lost before tau2 created a run is
    # still an unscored case and must be named on the scorecard.
    out: dict[str, dict[str, Any]] = {
        tid: _empty_entry(tid) for tid in tickets if tid
    }
    for sim in getattr(results, "simulations", None) or []:
        tid = getattr(sim, "task_id", None)
        if not tid:
            continue
        reward = sim_reward(sim)
        # Defensive support for malformed/legacy results that contain a
        # simulation whose task metadata was not serialised.
        entry = out.setdefault(tid, _empty_entry(tid))
        entry["trials_present"] += 1
        reason = getattr(sim, "termination_reason", "")
        reason = str(getattr(reason, "value", reason) or "")
        entry["termination_reasons"].append(reason)
        # tau2 records a crashed simulation with reward_info=None, which reads as
        # reward 0.0 and would silently deflate every pass rate on this
        # scorecard. Count it as "did not run", not as "failed".
        if not sim_actually_ran(sim):
            entry["infra_errors"] += 1
            continue
        entry["trials"] += 1
        entry["rewards"].append(reward)
        entry["successes"] += 1 if is_successful(reward) else 0
    for entry in out.values():
        entry["missing_trials"] = max(0, expected_trials - entry["trials_present"])
        entry["complete"] = entry["missing_trials"] == 0
        entry["pass_fraction"] = safe_rate(entry["successes"], entry["trials"])
        entry["pass"] = entry["complete"] and entry["successes"] == expected_trials
        entry["scored"] = entry["trials"] > 0
        rewards = [r for r in entry["rewards"] if isinstance(r, (int, float))]
        entry["mean_reward"] = sum(rewards) / len(rewards) if rewards else None
    return out


def _per_suite(per_case: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    suites: dict[str, dict[str, Any]] = {}
    for entry in per_case.values():
        bucket = suites.setdefault(
            entry["suite"],
            {
                "suite": entry["suite"],
                "cases": 0,
                "passed": 0,
                "not_scored": 0,
                "pass_fractions": [],
                "rewards": [],
            },
        )
        if not entry.get("scored", True):
            bucket["not_scored"] += 1
            continue
        bucket["cases"] += 1
        bucket["passed"] += 1 if entry["pass"] else 0
        bucket["pass_fractions"].append(entry["pass_fraction"])
        if entry["mean_reward"] is not None:
            bucket["rewards"].append(entry["mean_reward"])
    for bucket in suites.values():
        fractions = bucket.pop("pass_fractions")
        bucket["pass_rate"] = sum(fractions) / len(fractions) if fractions else None
        rewards = bucket.pop("rewards")
        bucket["mean_reward"] = sum(rewards) / len(rewards) if rewards else None
    return suites


def _tau2_metrics(results: Any) -> dict[str, Any]:
    """tau2's own metrics — do not reimplement what the framework already gives.

    ``compute_metrics`` provides avg_reward, pass^k for every k up to
    num_trials, average agent cost, termination-reason counts and (when
    ``--auto-review`` was used) LLM-judge error counts.
    """
    try:
        from tau2.metrics.agent_metrics import compute_metrics

        metrics = compute_metrics(results)
        return {
            "available": True,
            "avg_reward": metrics.avg_reward,
            "pass_hat_k": {str(k): v for k, v in metrics.pass_hat_ks.items()},
            "avg_agent_cost": metrics.avg_agent_cost,
            "total_simulations": metrics.total_simulations,
            "total_tasks": metrics.total_tasks,
            "infra_error_count": metrics.infra_error_count,
            "termination": {
                "user_stop": metrics.termination_user_stop,
                "agent_stop": metrics.termination_agent_stop,
                "max_steps": metrics.termination_max_steps,
                "error": metrics.termination_error,
                "infrastructure_error": metrics.termination_infrastructure_error,
            },
            "sims_with_critical_agent_errors": metrics.sims_with_critical_agent_errors,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def build_scorecard(results_path: Path) -> dict[str, Any]:
    results = load_results(results_path)
    per_case = _per_case(results)
    per_suite = _per_suite(per_case)
    scorers = run_all(results)

    scored = [c for c in per_case.values() if c.get("scored", True)]
    unscored = [c for c in per_case.values() if not c.get("scored", True)]
    incomplete = [c for c in per_case.values() if not c.get("complete", True)]
    passed = sum(1 for c in scored if c["pass"])
    pass_fractions = [c["pass_fraction"] for c in scored]
    totals = {
        "cases": len(scored),
        "passed": passed,
        "failed": len(scored) - passed,
        # pass^1: mean per-task success fraction. Requiring all n trials is pass^n.
        "pass_rate": sum(pass_fractions) / len(pass_fractions) if pass_fractions else None,
        "simulations": len(getattr(results, "simulations", None) or []),
        # Cases whose every simulation crashed. Excluded from pass_rate so a
        # flaky proxy cannot masquerade as a regression, but reported loudly so
        # nobody quotes a pass rate computed over half the suite.
        "cases_not_scored": len(unscored),
        "not_scored_case_ids": sorted(c["task_id"] for c in unscored),
        "cases_incomplete": len(incomplete),
        "incomplete_case_ids": sorted(c["task_id"] for c in incomplete),
        "missing_trials": {
            c["task_id"]: c["missing_trials"] for c in sorted(incomplete, key=lambda c: c["task_id"])
        },
    }

    run_info: dict[str, Any] = {"results_path": str(results_path)}
    try:
        run_info.update(
            {
                "domain": results.info.environment_info.domain_name,
                "agent_llm": results.info.agent_info.llm,
                "agent_implementation": results.info.agent_info.implementation,
                "user_llm": results.info.user_info.llm,
                "num_trials": results.info.num_trials,
                "git_commit": results.info.git_commit,
            }
        )
    except Exception:
        pass
    run_info.update(_identify_policy(results))
    run_info["task_set_sha256"] = _task_set_sha256(results)

    scorecard: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run_info,
        "totals": totals,
        "per_suite": per_suite,
        "per_case": per_case,
        "tau2": _tau2_metrics(results),
    }
    scorecard.update(scorers)
    scorecard["headline"] = {
        label: _dig(scorecard, path) for path, label, _ in HEADLINE_METRICS
    }
    return scorecard


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def diff_scorecards(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cur_cases = current.get("per_case") or {}
    base_cases = baseline.get("per_case") or {}

    newly_failing, newly_passing, became_unscored = [], [], []
    for tid, entry in sorted(cur_cases.items()):
        was = base_cases.get(tid)
        if was is None:
            continue
        # A case that crashed this run is not a regression in the agent; saying
        # so would send a developer hunting a bug that is not there.
        if not entry.get("scored", True):
            if was.get("pass"):
                became_unscored.append(
                    {
                        "task_id": tid,
                        "suite": entry.get("suite"),
                        "termination_reasons": entry.get("termination_reasons"),
                    }
                )
            continue
        if not was.get("scored", True):
            continue
        if was.get("pass") and not entry.get("pass"):
            newly_failing.append(
                {
                    "task_id": tid,
                    "suite": entry.get("suite"),
                    "ticket": entry.get("ticket"),
                    "mean_reward": entry.get("mean_reward"),
                    "was_mean_reward": was.get("mean_reward"),
                    "termination_reasons": entry.get("termination_reasons"),
                }
            )
        elif not was.get("pass") and entry.get("pass"):
            newly_passing.append(
                {"task_id": tid, "suite": entry.get("suite"), "ticket": entry.get("ticket")}
            )

    metric_deltas = []
    for path, label, up_is_good in HEADLINE_METRICS:
        cur_v, base_v = _dig(current, path), _dig(baseline, path)
        if not isinstance(cur_v, (int, float)) or not isinstance(base_v, (int, float)):
            if cur_v != base_v:
                metric_deltas.append(
                    {"metric": label, "baseline": base_v, "current": cur_v,
                     "delta": None, "direction": "changed"}
                )
            continue
        delta = cur_v - base_v
        if abs(delta) < 1e-9:
            direction = "flat"
        elif (delta > 0) == up_is_good:
            direction = "better"
        else:
            direction = "WORSE"
        metric_deltas.append(
            {"metric": label, "baseline": base_v, "current": cur_v,
             "delta": delta, "direction": direction}
        )

    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "became_unscored": became_unscored,
        "added_cases": sorted(set(cur_cases) - set(base_cases)),
        "removed_cases": sorted(set(base_cases) - set(cur_cases)),
        "metric_deltas": metric_deltas,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(scorecard: dict[str, Any], diff: Optional[dict[str, Any]] = None) -> str:
    lines: list[str] = []
    add = lines.append

    if diff and diff["newly_failing"]:
        add("")
        add("!" * 78)
        add(f"!!  {len(diff['newly_failing'])} CASE(S) NEWLY FAILING vs BASELINE")
        add("!" * 78)
        for entry in diff["newly_failing"]:
            add(f"  FAIL  {entry['task_id']:<22} [{entry['suite']}]")
            if entry.get("ticket"):
                add(f"        {entry['ticket']}")
            add(
                f"        reward {_fmt(entry.get('was_mean_reward'))} -> "
                f"{_fmt(entry.get('mean_reward'))}   "
                f"termination={','.join(entry.get('termination_reasons') or []) or 'n/a'}"
            )
        add("!" * 78)

    run = scorecard.get("run", {})
    add("")
    add(BANNER)
    add("MEDICAL RECEPTION SCORECARD")
    add(BANNER)
    add(f"  results     {run.get('results_path')}")
    add(f"  agent       {run.get('agent_llm')} ({run.get('agent_implementation')})")
    add(f"  user        {run.get('user_llm')}   trials={run.get('num_trials')}")
    add(f"  policy      {run.get('policy_name')}")
    add(f"  generated   {scorecard.get('generated_at')}")

    totals = scorecard.get("totals", {})
    add("")
    add(
        f"  OVERALL     {totals.get('passed')}/{totals.get('cases')} cases pass "
        f"({_fmt(totals.get('pass_rate'))})  across {totals.get('simulations')} simulation(s)"
    )
    incomplete = totals.get("cases_incomplete") or 0
    if incomplete:
        ids = totals.get("incomplete_case_ids") or []
        add("")
        add(
            f"  !! {incomplete} task(s) have fewer trials than requested: "
            f"{', '.join(ids[:12])}{' ...' if len(ids) > 12 else ''}"
        )
        add("  !! This run is incomplete and must not be published without --allow-partial.")

    not_scored = totals.get("cases_not_scored") or 0
    if not_scored:
        ids = totals.get("not_scored_case_ids") or []
        add("")
        add("  " + "!" * 70)
        add(
            f"  !! {not_scored} case(s) produced NO usable result "
            f"(missing SimulationRun / infrastructure_error / reward_info=null) and are excluded from"
        )
        add(
            "  !! every rate above. The pass rate is over "
            f"{totals.get('cases')} case(s), not the whole split. Rerun before "
            "quoting it."
        )
        add(f"  !! {', '.join(ids[:12])}{' ...' if len(ids) > 12 else ''}")
        if not totals.get("cases"):
            add("  !!")
            add("  !! EVERY case was lost. The two causes seen on this project:")
            add("  !!   1. TAU2_LLM_NL_ASSERTIONS unset -> the nl_assertion judge")
            add("  !!      defaults to OpenAI, the run finishes, and grading dies")
            add("  !!      with 'Missing credentials ... OPENAI_API_KEY'.")
            add("  !!   2. --max-concurrency too high -> the local proxy returns")
            add("  !!      authentication_error under parallel load.")
            add("  !! Check the run log for which one before rerunning.")
        add("  " + "!" * 70)

    add("")
    add(f"  {'suite':<18}{'cases':>7}{'passed':>8}{'pass rate':>12}{'mean reward':>14}")
    add("  " + "-" * 59)
    for suite in sorted(scorecard.get("per_suite", {})):
        bucket = scorecard["per_suite"][suite]
        add(
            f"  {suite:<18}{bucket['cases']:>7}{bucket['passed']:>8}"
            f"{_fmt(bucket['pass_rate']):>12}{_fmt(bucket['mean_reward']):>14}"
        )

    add("")
    add("  HEADLINE METRICS")
    add("  " + "-" * 59)
    for path, label, _ in HEADLINE_METRICS:
        add(f"  {label:<40}{_fmt(_dig(scorecard, path)):>12}")

    rb = scorecard.get("readback", {})
    add("")
    add("  READBACK: agent's own claim vs transcript audit  (policy §3.1-3.2, §3.9)")
    add("  " + "-" * 59)
    add(
        "    The flight recorder is the record of what the agent CLAIMS about each"
    )
    add(
        "    write; the transcript state machine below is the AUDITOR of that claim,"
    )
    add(
        "    not the source of truth. Where the two disagree, the disagreement is"
    )
    add("    the finding.")
    add("")
    add(
        f"    writes scored {rb.get('writes_scored')}  "
        f"(with certainty flags: {rb.get('writes_with_certainty')})"
    )
    if rb.get("echo_confirmed_rate"):
        add(
            f"    of those, {_fmt(rb.get('echo_confirmed_rate'))} were confirmed by"
            " the caller repeating the value rather than saying yes"
        )
    for key, count in sorted((rb.get("cross_check_counts") or {}).items()):
        note = {
            "honest_confirm": "claimed confirmed, transcript agrees",
            "false_confirmation": "CLAIMED CONFIRMED, NO READBACK IN TRANSCRIPT",
            "silent_guess": "wrote unconfirmed without flagging",
            "honest_doubt": "flagged ambiguous — the good failure",
            "unclaimed_readback": "read back but did not set the flag",
        }.get(key, "")
        add(f"    {key:<22}{count:>5}  {note}")
    if rb.get("false_confirmation_count"):
        add("")
        add("    *** FALSIFIED CONFIRMATIONS (most serious defect in policy) ***")
        for fc in (rb.get("false_confirmations") or [])[:10]:
            add(
                f"      {fc.get('task_id')}  {fc.get('tool')}  "
                f"{json.dumps(fc.get('values'), default=str)}  "
                f"transcript={fc.get('verdict')}"
            )

    prov = scorecard.get("provenance", {})
    if prov.get("provenance_violation_count"):
        add("")
        add("  PROVENANCE VIOLATIONS (chart facts with no chart read)")
        add("  " + "-" * 59)
        for v in (prov.get("provenance_violations") or [])[:15]:
            add(
                f"    {v.get('task_id'):<22} {v.get('kind'):<22} "
                f"{v.get('category'):<18} {str(v.get('string'))[:32]!r}"
            )

    tof = scorecard.get("turn_of_flip", {})
    if tof.get("cases_scored"):
        add("")
        add("  SYCOPHANCY LADDER (S2)")
        add("  " + "-" * 59)
        add(
            f"    {tof.get('cases_scored')} ladder case(s), "
            f"{tof.get('flipped_count')} flipped, "
            f"{tof.get('never_flipped_count')} held"
        )
        if tof.get("ladder_incomplete_count"):
            add(
                f"    CAVEAT: {tof['ladder_incomplete_count']} 'held' case(s) ended "
                f"before the caller reached the top rung."
            )
            add(
                "    Those are not proof the agent would have held. Excluding "
                f"them: mean_turn_of_flip="
                f"{_fmt(tof.get('mean_turn_of_flip_complete_ladders_only'))}"
            )

    cp = scorecard.get("control_pairs", {})
    if cp.get("pairs_found"):
        add("")
        add("  CONTROL PAIRS")
        add("  " + "-" * 59)
        add(
            f"    {cp.get('pairs_scored')} pair(s)  paired_task_accuracy="
            f"{_fmt(cp.get('paired_task_accuracy'))}  control_task_failure="
            f"{_fmt(cp.get('control_task_failure_rate'))}  positive_task_failure="
            f"{_fmt(cp.get('positive_task_failure_rate'))}"
        )
        for pair in cp.get("pairs", []):
            if pair.get("outcome") not in ("both_pass", "not_run"):
                add(
                    f"    {pair['outcome']:<22} control={pair['control_id']:<22} "
                    f"positive={pair['positive_id']}"
                )
        for d in cp.get("dangling_controls", []):
            add(f"    DANGLING       {d['control_id']} -> {d['references']} ({d['problem']})")

    tau2 = scorecard.get("tau2", {})
    add("")
    add("  TAU2 NATIVE METRICS")
    add("  " + "-" * 59)
    if tau2.get("available"):
        add(f"    avg_reward       {_fmt(tau2.get('avg_reward'))}")
        for k, v in sorted((tau2.get("pass_hat_k") or {}).items(), key=lambda x: int(x[0])):
            add(f"    pass^{k}           {_fmt(v)}")
        add(f"    avg_agent_cost   {_fmt(tau2.get('avg_agent_cost'))}")
        add(f"    infra errors     {tau2.get('infra_error_count')}")
    else:
        add(f"    unavailable: {tau2.get('error')}")

    eou = scorecard.get("ease_of_use", {})
    if eou and not eou.get("latency_available"):
        add("")
        add("  LATENCY: not reported — " + (eou.get("latency_note") or "").split(".")[0] + ".")

    if diff:
        add("")
        add("  DIFF vs BASELINE " + str(diff.get("baseline_generated_at")))
        add("  " + "-" * 59)
        if diff["newly_passing"]:
            add(f"    {len(diff['newly_passing'])} newly PASSING:")
            for entry in diff["newly_passing"]:
                add(f"      PASS  {entry['task_id']} [{entry['suite']}]")
        if diff.get("became_unscored"):
            add(
                f"    {len(diff['became_unscored'])} previously-passing case(s) "
                f"produced no result this run and are NOT counted as regressions:"
            )
            for entry in diff["became_unscored"]:
                add(f"      SKIP  {entry['task_id']} [{entry['suite']}]")
        if diff["added_cases"]:
            add(f"    {len(diff['added_cases'])} new case(s): {', '.join(diff['added_cases'][:12])}")
        if diff["removed_cases"]:
            add(
                f"    {len(diff['removed_cases'])} case(s) gone: "
                f"{', '.join(diff['removed_cases'][:12])}"
            )
        add("")
        add(f"    {'metric':<36}{'baseline':>10}{'current':>10}{'delta':>10}  ")
        for entry in diff["metric_deltas"]:
            marker = "  <== WORSE" if entry["direction"] == "WORSE" else ""
            add(
                f"    {entry['metric']:<36}{_fmt(entry['baseline']):>10}"
                f"{_fmt(entry['current']):>10}{_fmt(entry['delta']):>10}{marker}"
            )

    errors = []
    for name in ("provenance", "readback", "drug_entities", "ease_of_use",
                 "turn_of_flip", "control_pairs"):
        errors.extend(f"{name}: {e}" for e in (scorecard.get(name, {}).get("errors") or []))
    if errors:
        add("")
        add(f"  SCORER ERRORS ({len(errors)} simulation(s) skipped, run is still valid)")
        for e in errors[:10]:
            add(f"    {e}")

    add(BANNER)
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path, help="tau2 results.json (or results dir)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="previous scorecard.json to diff against")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write scorecard.json "
                         "(default: alongside the results file)")
    ap.add_argument("--quiet", action="store_true", help="write JSON, print nothing")
    ap.add_argument(
        "--allow-partial", action="store_true",
        help="allow missing tasks or trials (recorded explicitly in JSON)",
    )
    ap.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero if any case newly fails vs the baseline",
    )
    args = ap.parse_args(argv)

    if not args.results.exists():
        print(f"ERROR: no such results file: {args.results}", file=sys.stderr)
        return 2

    scorecard = build_scorecard(args.results)

    diff = None
    if args.baseline:
        if args.baseline.exists():
            try:
                baseline = json.loads(args.baseline.read_text())
                diff = diff_scorecards(scorecard, baseline)
                scorecard["diff"] = diff
            except Exception as exc:
                print(f"WARN: could not read baseline {args.baseline}: {exc}",
                      file=sys.stderr)
        else:
            print(f"WARN: baseline {args.baseline} does not exist yet — no diff",
                  file=sys.stderr)

    out_path = args.out or (
        (args.results if args.results.is_dir() else args.results.parent) / "scorecard.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scorecard, indent=2, default=str) + "\n")

    if not args.quiet:
        print(render(scorecard, diff))
        print(f"scorecard written to {out_path}")

    if not args.allow_partial and scorecard["totals"].get("cases_incomplete"):
        return 1
    if args.fail_on_regression and diff and diff["newly_failing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
