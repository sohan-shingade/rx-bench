"""Run medical_reception cases against the voice-pipeline agent class.

    python -m rx_bench.voice_pipeline.run_pipeline --cases S1-lasa-001 \
        --trials 1 --profile realistic --run-name pipeline_dryrun

Per user turn: user-sim text -> macOS `say` TTS (persona voice rotated per
conversation) -> channel degradation profile -> Deepgram ASR -> the TRANSCRIPT
is what the agent receives. Agent replies pass through as text (transcript
injection; see README.md). Everything else is stock tau2: same ``llm_agent``,
same tools/policy/environment/evaluator as a text-tier run, so the output is
a standard ``results.json`` that ``rxbench score`` and every other scorer
read unchanged.

Outputs, under ``data/simulations/voice_pipeline/<run-name>/`` (gitignored):

    results.json          standard tau2 Results (scorecard-compatible)
    manifest.json         reproducibility record: models, profile, voices, seed
    pipeline_turns.jsonl  per-turn: original text, WAV paths, ASR transcript,
                          word confidences, WER, timings
    audio/                the WAVs (unless --no-keep-audio)

Deliberately additive: reads tau2 tasks through the registry (no re-merge of
case files — a text-tier eval may be running against this repo concurrently)
and registers its user under a name nothing else uses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root

# Quiet tau2's import-time chatter before anything imports it (same dance as
# rx_bench.live.talk, for the same reason).
from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import litellm  # noqa: E402

litellm.suppress_debug_info = True

import rx_bench  # noqa: E402,F401  (registers the medical_reception domain)
from rx_bench.voice_pipeline.pipeline import (  # noqa: E402
    DEFAULT_ASR_MODEL,
    PROFILES,
    RESULTS_ROOT,
    PipelineConfig,
    SpeechChannel,
    load_env_file,
)
from rx_bench.voice_pipeline.pipeline_user import bind_pipeline_user  # noqa: E402

DOMAIN = "medical_reception"
USER_NAME = "voice_pipeline_user"
DEFAULT_AGENT_LLM = "gpt-4.1-2025-04-14"
DEFAULT_USER_LLM = "gpt-4.1-2025-04-14"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Voice-pipeline (ASR->LLM) agent-under-test runner.",
    )
    ap.add_argument("--cases", required=False, default=None,
                    help="comma-separated task ids (e.g. S1-lasa-001,S1-lasa-002)")
    ap.add_argument("--list-cases", action="store_true", help="list case ids and exit")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--profile", default="realistic",
                    help=f"degradation profile: {PROFILES} or 'realistic' (=noisy_phone)")
    ap.add_argument("--agent-llm", default=os.environ.get("AGENT_LLM", DEFAULT_AGENT_LLM),
                    help="any litellm-routable text model")
    ap.add_argument("--user-llm", default=os.environ.get("USER_LLM", DEFAULT_USER_LLM))
    ap.add_argument("--asr-model", default=os.environ.get("DEEPGRAM_MODEL", DEFAULT_ASR_MODEL))
    ap.add_argument("--voices", default=None,
                    help="comma-separated archetype keys to rotate (default: all installed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None,
                    help="run directory name (default: pipeline-<timestamp>)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="keep at 1: `say` is serial anyway")
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", 40)))
    ap.add_argument("--max-errors", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--evaluate", default="env",
                    help="tau2 evaluation type ('env' needs no judge; "
                         "'all_with_nl_assertions' adds the LLM judge)")
    ap.add_argument("--no-keep-audio", action="store_true",
                    help="delete per-turn WAVs after ASR (transcripts still logged)")
    ap.add_argument("--no-scorecard", action="store_true")
    ap.add_argument("--log-level", default="WARNING")
    return ap.parse_args(argv)


def setup_env() -> None:
    """Default the NL-assertion judge so --evaluate all_with_nl_assertions works."""
    os.environ.setdefault("TAU2_LLM_NL_ASSERTIONS", DEFAULT_USER_LLM)


def all_tasks() -> list:
    from tau2.registry import registry

    return registry.get_tasks_loader(DOMAIN)(None)


def select_tasks(case_arg: Optional[str]) -> list:
    tasks = all_tasks()
    if not case_arg:
        raise SystemExit("--cases is required (or use --list-cases)")
    wanted = [c.strip() for c in case_arg.split(",") if c.strip()]
    by_id = {t.id: t for t in tasks}
    missing = [c for c in wanted if c not in by_id]
    if missing:
        known = ", ".join(sorted(by_id)[:8])
        raise SystemExit(f"unknown case ids {missing} (known: {known}, …)")
    return [by_id[c] for c in wanted]


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    load_env_file()  # DEEPGRAM_API_KEY lives in the repo-root .env
    setup_env()

    if args.list_cases:
        for task in all_tasks():
            purpose = ""
            if task.description and task.description.purpose:
                purpose = task.description.purpose.strip().splitlines()[0]
            print(f"{task.id:<16} {purpose[:96]}")
        return 0

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("DEEPGRAM_API_KEY is not set (and not found in .env) — the ASR leg "
              "cannot run.", file=sys.stderr)
        return 2

    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.batch import run_tasks

    tasks = select_tasks(args.cases)

    run_name = args.run_name or f"pipeline-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = RESULTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    channel = SpeechChannel(PipelineConfig(
        profile=args.profile,
        asr_model=args.asr_model,
        audio_dir=run_dir / "audio",
        keep_audio=not args.no_keep_audio,
        seed=args.seed,
        voices=[v.strip() for v in args.voices.split(",")] if args.voices else None,
    ))

    # The registry hands the user only its instructions string; give the bound
    # class a way back to the task id for audio paths and the turn log.
    task_id_by_instructions = {str(t.user_scenario): t.id for t in tasks}
    registry.register_user(
        bind_pipeline_user(channel, run_dir / "pipeline_turns.jsonl", task_id_by_instructions),
        USER_NAME,
    )

    config = TextRunConfig(
        domain=DOMAIN,
        agent="llm_agent",
        user=USER_NAME,
        llm_agent=args.agent_llm,
        llm_args_agent={
            "temperature": args.temperature,
            # Providers shed load in bursts; back off rather than die (cf. talk.py).
            "num_retries": 6,
            "retry_strategy": "exponential_backoff_retry",
        },
        llm_user=args.user_llm,
        llm_args_user={"temperature": args.temperature},
        num_trials=args.trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_concurrency=args.concurrency,
        seed=args.seed,
        log_level=args.log_level,
        auto_resume=True,
    )

    try:
        evaluation_type = EvaluationType(args.evaluate)
    except ValueError:
        valid = ", ".join(e.value for e in EvaluationType)
        raise SystemExit(f"--evaluate must be one of: {valid}")

    manifest = {
        "agent_class": "voice-pipeline (ASR -> text LLM with tools; TTS leg not rendered in v1)",
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": git_rev(),
        "domain": DOMAIN,
        "cases": [t.id for t in tasks],
        "trials": args.trials,
        "agent_llm": args.agent_llm,
        "user_llm": args.user_llm,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "evaluation_type": args.evaluate,
        "channel": channel.manifest(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"==> voice-pipeline run '{run_name}'")
    print(f"    cases   {', '.join(t.id for t in tasks)}  x{args.trials} trials")
    print(f"    agent   {args.agent_llm}")
    print(f"    user    {args.user_llm} -> say -> {channel.config.profile} -> "
          f"deepgram/{args.asr_model}")
    print(f"    saving  {run_dir}")

    save_path = run_dir / "results.json"
    run_tasks(
        config,
        tasks,
        save_path=save_path,
        save_dir=run_dir,
        evaluation_type=evaluation_type,
    )

    print(f"\n==> results at {save_path}")
    print(f"    per-turn channel log: {run_dir / 'pipeline_turns.jsonl'}")

    if not args.no_scorecard and save_path.exists():
        print("\n==> scorecard")
        proc = subprocess.run(
            [sys.executable, "-m", "rx_bench.cli", "score", str(save_path)],
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            print(f"    (scorecard step failed — raw results are still at {save_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
