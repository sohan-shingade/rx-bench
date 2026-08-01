#!/usr/bin/env python
"""Re-run the simulations a results.json lost to infrastructure, and splice them in.

The local proxy rejects auth in short bursts. A burst does not fail a run --
tau2 records each casualty as ``termination_reason: infrastructure_error`` with
``reward_info: null``, zero messages and zero duration, and carries on. The run
then *looks* complete: 97 of 97 task ids present, one headline reward number.

Those casualties are not zeros, they are absences, and the difference matters in
both directions. Counted as failures they deflate every rate. Dropped silently
they shrink the denominator without saying so. And in a mutation matrix a case
missing from the baseline reads as "was not passing", which manufactures kills.

``--auto-resume`` will not fix it: the failed simulations exist, so tau2
considers those tasks done. This re-runs exactly the lost ones and splices the
successful retries back into the original file.

    python -m rx_bench.harness.repair data/simulations/base97_v2/results.json
    python -m rx_bench.harness.repair <results.json> --dry-run

The original is copied to ``results.json.bak`` before anything is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


from rx_bench.harness.common import MUTANTS_DIR, POLICY_PATH, SPLITS_PATH, TAU2_ROOT  # noqa: E402


def lost_simulations(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Simulations that never produced a usable result.

    Deliberately broader than ``termination_reason == "infrastructure_error"``:
    a null reward_info means ungraded whatever the recorded reason, and that is
    the property that matters downstream.
    """
    return [
        s
        for s in (results.get("simulations") or [])
        if s.get("reward_info") is None
        or s.get("termination_reason") == "infrastructure_error"
    ]


def policy_of(results: dict[str, Any]) -> str:
    """The policy text the run was actually conducted under."""
    return (
        ((results.get("info") or {}).get("environment_info") or {}).get("policy") or ""
    )


def detect_mutant(results: dict[str, Any]) -> str | None:
    """Which mutant policy produced this run, if any.

    A repair re-runs simulations, and a simulation's result is only meaningful
    under the policy the rest of the run used. Repairing a mutant run without
    MEDICAL_POLICY_MUTANT set would quietly re-run those cases under the FULL
    policy and splice them in beside mutant results -- the recovered cases would
    behave better than their neighbours for a reason nothing in the file
    records, and in a mutation matrix that shows up as the mutant failing to
    kill. Silent, directional, and indistinguishable from a real finding.

    The run records its own policy text, so match it rather than trusting the
    caller to remember.
    """
    policy = policy_of(results).strip()
    if not policy:
        return None
    if POLICY_PATH.exists() and POLICY_PATH.read_text().strip() == policy:
        return None  # the base policy: nothing to set
    for path in sorted(MUTANTS_DIR.glob("*.md")) if MUTANTS_DIR.exists() else []:
        if path.read_text().strip() == policy:
            return path.stem
    raise SystemExit(
        "the policy recorded in this run matches neither policy.md nor any file "
        "in mutants/. Re-running its lost cases would use a policy that is not "
        "the one the run used, and the spliced results would be quietly wrong. "
        "Regenerate the mutants (mutate.py --generate) or repair by hand."
    )


def repair(
    results_path: Path,
    concurrency: int = 1,
    max_steps: int = 40,
    max_steps_seconds: int = 300,
    agent_llm: str = "gpt-4.1-2025-04-14",
    user_llm: str | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
) -> int:
    results_path = Path(results_path)
    data = json.loads(results_path.read_text())
    lost = lost_simulations(data)
    if not lost:
        print(f"{results_path}: nothing lost, no repair needed")
        return 0

    ids = sorted({s["task_id"] for s in lost})
    mutant = detect_mutant(data)
    print(f"{results_path}: {len(lost)} lost simulation(s) over {len(ids)} case(s)")
    print(f"    policy: {mutant or 'policy.md (base)'}")
    for tid in ids:
        print(f"    {tid}")
    if dry_run:
        return 0

    # Concurrency 1 by default: the whole point is to get these few cases back,
    # and the burst that killed them scales its casualties with concurrency.
    # A repair run's task set is exactly the cases that were lost, so it SHRINKS
    # every time a repair succeeds partially. Reusing one name means the second
    # invocation asks tau2 to resume a saved run whose task set no longer
    # matches, and tau2 rightly refuses: "Tasks were removed from the task set."
    # Each repair therefore gets its own directory. The counter is deterministic
    # -- same sequence of repairs, same names -- so a run is still reproducible.
    base_name = run_name or f"repair_{results_path.parent.name}"
    sims_dir = TAU2_ROOT / "data" / "simulations"
    split_name = base_name
    n = 2
    while (sims_dir / split_name / "results.json").exists():
        split_name = f"{base_name}_{n}"
        n += 1
    splits = json.loads(Path(SPLITS_PATH).read_text()) if Path(SPLITS_PATH).exists() else {}
    splits[split_name] = ids
    Path(SPLITS_PATH).write_text(json.dumps(splits, indent=2) + "\n")

    env = dict(os.environ)
    env.setdefault("TAU2_LLM_NL_ASSERTIONS", agent_llm)
    # Set, not setdefault: an inherited MEDICAL_POLICY_MUTANT from whatever shell
    # launched this would silently outrank the policy the file itself records.
    if mutant:
        env["MEDICAL_POLICY_MUTANT"] = mutant
    else:
        env.pop("MEDICAL_POLICY_MUTANT", None)
    cmd = [
        sys.executable, "-m", "rx_bench.cli", "run",
        "--domain", "medical_reception",
        "--task-split-name", split_name,
        "--agent-llm", agent_llm,
        "--user-llm", user_llm or agent_llm,
        "--num-trials", "1",
        "--max-steps", str(max_steps),
        "--max-steps-seconds", str(max_steps_seconds),
        "--max-concurrency", str(concurrency),
        "--save-to", split_name,
        "--auto-resume",
    ]
    print("\n==> " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=str(TAU2_ROOT), stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        print(f"repair run failed (exit {proc.returncode})", file=sys.stderr)
        return 1

    fresh_path = TAU2_ROOT / "data" / "simulations" / split_name / "results.json"
    fresh = json.loads(fresh_path.read_text())
    good = {
        s["task_id"]: s
        for s in (fresh.get("simulations") or [])
        if s.get("reward_info") is not None
        and s.get("termination_reason") != "infrastructure_error"
    }
    if not good:
        print("the repair run recovered nothing; the original is untouched",
              file=sys.stderr)
        return 1

    # Back up before writing. A botched splice on the only copy of a two-hour
    # run is a worse outcome than the five missing cases.
    shutil.copy2(results_path, results_path.with_suffix(".json.bak"))

    replaced, still_lost = [], []
    out = []
    for s in data.get("simulations") or []:
        is_lost = (
            s.get("reward_info") is None
            or s.get("termination_reason") == "infrastructure_error"
        )
        if is_lost and s["task_id"] in good:
            fixed = dict(good[s["task_id"]])
            # Keep the original simulation id so anything keyed on it still
            # resolves; everything else comes from the successful retry.
            fixed["id"] = s.get("id", fixed.get("id"))
            fixed["trial"] = s.get("trial", fixed.get("trial", 0))
            out.append(fixed)
            replaced.append(s["task_id"])
        else:
            if is_lost:
                still_lost.append(s["task_id"])
            out.append(s)
    data["simulations"] = out
    results_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"\nspliced {len(replaced)} recovered simulation(s) into {results_path}")
    print(f"  backup: {results_path.with_suffix('.json.bak')}")
    if still_lost:
        print(f"  STILL LOST ({len(still_lost)}): {', '.join(sorted(set(still_lost)))}")
        print("  Re-run repair.py, or quote the split as incomplete.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what was lost and stop")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="the burst that caused the loss scales with this; "
                         "1 is the point")
    ap.add_argument("--max-steps", type=int, default=40,
                    help="must match the original run to stay comparable")
    ap.add_argument("--max-steps-seconds", type=int, default=300)
    ap.add_argument("--agent-llm", default="gpt-4.1-2025-04-14")
    ap.add_argument("--user-llm", default=None)
    args = ap.parse_args(argv)
    return repair(
        args.results,
        concurrency=args.concurrency,
        max_steps=args.max_steps,
        max_steps_seconds=args.max_steps_seconds,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
