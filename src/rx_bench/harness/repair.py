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


def splice_recovered(
    simulations: list[dict[str, Any]],
    recovered: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Splice retries by exact task/trial pair, preserving original identities."""
    out, replaced, still_lost = [], [], []
    for sim in simulations:
        is_lost = (
            sim.get("reward_info") is None
            or sim.get("termination_reason") == "infrastructure_error"
        )
        key = (sim["task_id"], int(sim.get("trial", 0)))
        if is_lost and key in recovered:
            fixed = dict(recovered[key])
            fixed["id"] = sim.get("id", fixed.get("id"))
            fixed["trial"] = key[1]
            out.append(fixed)
            replaced.append(key)
        else:
            if is_lost:
                still_lost.append(key)
            out.append(sim)
    return out, replaced, still_lost


PINNED_NL_JUDGE = "claude-gpt-5-6-luna"


def repair(
    results_path: Path,
    concurrency: int = 1,
    max_steps: int | None = None,
    max_steps_seconds: int | None = None,
    agent_llm: str | None = None,
    user_llm: str | None = None,
    dry_run: bool = False,
    run_name: str | None = None,
) -> int:
    results_path = Path(results_path)
    data = json.loads(results_path.read_text())
    info = data.get("info") or {}
    mutant = detect_mutant(data)
    agent_info = info.get("agent_info") or {}
    user_info = info.get("user_info") or {}
    agent_llm = agent_llm or agent_info.get("llm")
    user_llm = user_llm or user_info.get("llm")
    max_steps = max_steps if max_steps is not None else info.get("max_steps")
    if not agent_llm or not user_llm or max_steps is None:
        raise SystemExit("results.json does not record agent/user LLMs and max_steps")
    lost = lost_simulations(data)
    if not lost:
        print(f"{results_path}: nothing lost, no repair needed")
        return 0

    pairs = sorted(
        ((s["task_id"], int(s.get("trial", 0))) for s in lost),
        key=lambda pair: (pair[0], pair[1]),
    )
    print(f"{results_path}: {len(lost)} lost simulation(s) over {len(set(p[0] for p in pairs))} case(s)")
    print(f"    policy: {mutant or 'policy.md (base)'}")
    for tid, trial in pairs:
        print(f"    {tid} trial {trial}")
    if dry_run:
        return 0

    # Run each lost trial separately. A one-trial retry cannot be reused for two
    # original slots without fabricating identical trials and invalidating pass^k.
    base_name = run_name or f"repair_{results_path.parent.name}"
    sims_dir = TAU2_ROOT / "data" / "simulations"
    splits_path = Path(SPLITS_PATH)
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else {}
    env = dict(os.environ)
    env.setdefault("TAU2_LLM_NL_ASSERTIONS", PINNED_NL_JUDGE)
    if mutant:
        env["MEDICAL_POLICY_MUTANT"] = mutant
    else:
        env.pop("MEDICAL_POLICY_MUTANT", None)

    recovered: dict[tuple[str, int], dict[str, Any]] = {}
    for tid, trial in pairs:
        stem = base_name if trial == 0 else f"{base_name}_{trial}"
        split_name = stem
        n = 2
        while (sims_dir / split_name / "results.json").exists():
            split_name = f"{stem}_{n}"
            n += 1
        splits[split_name] = [tid]
        splits_path.write_text(json.dumps(splits, indent=2) + "\n")
        cmd = [
            sys.executable, "-m", "rx_bench.cli", "run",
            "--domain", "medical_reception",
            "--task-split-name", split_name,
            "--agent", agent_info.get("implementation", "llm_agent"),
            "--agent-llm", agent_llm,
            "--user", user_info.get("implementation", "user_simulator"),
            "--user-llm", user_llm,
            "--num-trials", "1",
            "--max-steps", str(max_steps),
            "--max-concurrency", str(concurrency),
            "--save-to", split_name,
            "--auto-resume",
        ]
        if max_steps_seconds is not None:
            cmd += ["--max-steps-seconds", str(max_steps_seconds)]
        if info.get("max_errors") is not None:
            cmd += ["--max-errors", str(info["max_errors"])]
        if info.get("seed") is not None:
            cmd += ["--seed", str(info["seed"])]
        if agent_info.get("llm_args") is not None:
            cmd += ["--agent-llm-args", json.dumps(agent_info["llm_args"])]
        if user_info.get("llm_args") is not None:
            cmd += ["--user-llm-args", json.dumps(user_info["llm_args"])]
        print("\n==> " + " ".join(cmd))
        proc = subprocess.run(cmd, env=env, cwd=str(TAU2_ROOT), stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            continue
        fresh_path = sims_dir / split_name / "results.json"
        fresh = json.loads(fresh_path.read_text())
        good = next((
            s for s in (fresh.get("simulations") or [])
            if s.get("task_id") == tid
            and s.get("reward_info") is not None
            and s.get("termination_reason") != "infrastructure_error"
        ), None)
        if good:
            recovered[(tid, trial)] = good

    if not recovered:
        print("the repair runs recovered nothing; the original is untouched",
              file=sys.stderr)
        return 1

    # Back up before writing. A botched splice on the only copy of a two-hour
    # run is a worse outcome than the five missing cases.
    shutil.copy2(results_path, results_path.with_suffix(".json.bak"))

    out, replaced, still_lost = splice_recovered(
        data.get("simulations") or [], recovered
    )
    data["simulations"] = out
    results_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"\nspliced {len(replaced)} recovered simulation(s) into {results_path}")
    print(f"  backup: {results_path.with_suffix('.json.bak')}")
    if still_lost:
        print(f"  STILL LOST ({len(still_lost)}): " + ", ".join(
            f"{tid}[{trial}]" for tid, trial in sorted(still_lost)
        ))
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
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override the original run's recorded turn cap")
    ap.add_argument("--max-steps-seconds", type=int, default=None)
    ap.add_argument("--agent-llm", default=None)
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
