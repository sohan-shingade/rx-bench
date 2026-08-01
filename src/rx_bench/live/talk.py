"""Talk to the medical_reception agent yourself, by keyboard or by voice.

    python -m rx_bench.live.talk                     # type at it
    python -m rx_bench.live.talk --task S1-lasa-001  # type at it, inside a graded case
    python -m rx_bench.live.talk --voice             # say it out loud

This is not a demo harness sitting next to the benchmark. It *is* the benchmark,
with the LLM lifted out of the caller's seat and a person dropped in. Same
orchestrator, same ``llm_agent``, same nineteen tools, same ``policy.md``, same
environment, same evaluator. The only substitution is who produces the user
turns, which is exactly the substitution you want when the question is "does the
thing the scorecard describes behave like that when I poke it".

Because nothing else changed, the session is written out as an ordinary
``results.json``. Every scorer in ``rx_bench.harness`` reads it: the flight recorder
will tell you whether the agent's self-reported certainty matched what it
actually did to you, and ``--task`` runs the real environment assertions against
the chart it left behind.

Two things this deliberately does not do:

* It does not seed. A person is not reproducible, so a live session is evidence
  about one conversation, not a measurement. The scorecard numbers still come
  from the simulated runs.
* It does not paper over the caller. Whatever you say goes to the agent
  verbatim --- including the transcription, when the channel is voice --- so a
  drug name the ASR mangled reaches the agent mangled. That failure mode is the
  subject of the project; hiding it here would be absurd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root

# Silence tau2's import-time logging before it happens. Registry construction
# logs at DEBUG and the data-dir resolution at INFO, all of it on stderr, all of
# it before argv has been looked at -- so this cannot wait until main(). The
# level is raised again from --log-level once the arguments are parsed.
from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

# litellm prints a red "Provider List" banner every time it is asked to price a
# model id that is not in its public price table -- which is every call here,
# because the proxy's model names are local. It is chatter about a price table,
# it lands in the middle of the conversation, and the exception it accompanies
# is caught upstream either way. Suppressed only for the banner; errors still
# raise and still surface.
import litellm  # noqa: E402

litellm.suppress_debug_info = True

from rx_bench.live.channels import (  # noqa: E402
    ASRUnavailable,
    Channel,
    TextChannel,
    VoiceChannel,
)
from rx_bench.live.human_user import HumanUser  # noqa: E402

DOMAIN = "medical_reception"
USER_NAME = "human_user"
WALKIN_TASK_ID = "live-walkin"

#: Recorded in the run's ``Info`` where a model id would normally sit. A live
#: session must never be mistaken for a simulated one when it is read back six
#: weeks later, and the user model is the field that would otherwise lie.
HUMAN_MARKER = "human"

DEFAULT_AGENT_LLM = "gpt-4.1-2025-04-14"


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

WALKIN_INSTRUCTIONS = """\
You are a real person calling a medical practice. There is no script: say
whatever you actually want to say. Nothing you say is checked against a
scenario, because there is no scenario --- this is a free-form call.
"""


def walkin_task() -> Any:
    """A task with no assertions, for when you just want to poke it.

    tau2 requires a ``Task`` to run an orchestrator, and a ``Task`` requires
    caller instructions. :class:`HumanUser` never reads them --- you are the
    instructions --- but they are stored on the run so a saved free-form session
    is self-describing rather than an unexplained blank.
    """
    from tau2.data_model.tasks import Description, Task, UserScenario

    return Task(
        id=WALKIN_TASK_ID,
        description=Description(
            purpose="Free-form live call. Not a graded case: no assertions.",
            notes="Caller is a human. Not reproducible; not part of any suite.",
        ),
        user_scenario=UserScenario(instructions=WALKIN_INSTRUCTIONS),
    )


def all_tasks() -> list[Any]:
    """Every case in the domain, not just the ``base`` split.

    ``--task`` is for pointing at one specific case you want to feel, and the
    interesting ones are frequently outside whatever split a batch run used.
    """
    from tau2.registry import registry

    return registry.get_tasks_loader(DOMAIN)(None)


def load_task(task_id: Optional[str]) -> Any:
    if task_id is None:
        return walkin_task()
    tasks = all_tasks()
    for task in tasks:
        if task.id == task_id:
            return task
    known = ", ".join(sorted(t.id for t in tasks)[:8])
    raise SystemExit(f"no task {task_id!r} in {DOMAIN} (e.g. {known}, …)")


def list_tasks() -> int:
    for task in all_tasks():
        purpose = ""
        if task.description and task.description.purpose:
            purpose = task.description.purpose.strip().splitlines()[0]
        print(f"{task.id:<16} {purpose[:96]}")
    return 0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def bind_channel(channel: Channel) -> type:
    """Produce a registrable user class that already knows its channel.

    tau2's registry takes a *class* and constructs it itself, passing a fixed
    keyword set. There is no place to hand it a channel, and a ``partial`` fails
    the ``issubclass`` check, so the channel is closed over by a subclass
    instead.
    """

    class BoundHumanUser(HumanUser):
        def __init__(self, **kwargs: Any):
            kwargs.pop("channel", None)
            super().__init__(channel=channel, **kwargs)

    BoundHumanUser.__name__ = "BoundHumanUser"
    return BoundHumanUser


def build_config(args: argparse.Namespace) -> Any:
    from tau2.data_model.simulation import TextRunConfig

    return TextRunConfig(
        domain=DOMAIN,
        agent="llm_agent",
        user=USER_NAME,
        llm_agent=args.model,
        llm_args_agent={
            "temperature": args.temperature,
            # litellm's default retry strategy is `constant_retry`: three
            # attempts with no wait between them. Against a proxy that rejects
            # auth in short bursts that is three failures in one burst, and the
            # whole conversation dies. A live call is worth far more than a
            # batch simulation --- it cannot be re-run, because a person said
            # those words once --- so back off and try for longer.
            "num_retries": args.retries,
            "retry_strategy": "exponential_backoff_retry",
        },
        # Not a model. See HUMAN_MARKER.
        llm_user=HUMAN_MARKER,
        llm_args_user={},
        max_steps=args.max_steps,
        num_trials=1,
        max_errors=args.max_errors,
        max_concurrency=1,
        seed=0,
        log_level=args.log_level,
    )


def session_dir(root: Path, task_id: str, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%d-%H%M%S")
    return root / f"live-{stamp}-{task_id}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def policy_fingerprint(policy: Optional[str]) -> str:
    if not policy:
        return "none"
    return hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12]


def banner(
    channel: Channel,
    *,
    task: Any,
    config: Any,
    policy_sha: str,
    asr: Optional[str],
    save_path: Path,
) -> None:
    graded = bool(task.evaluation_criteria)
    lines = [
        f"domain    {DOMAIN}  (policy {policy_sha})",
        f"agent     {config.llm_agent}",
        f"task      {task.id}" + ("" if graded else "  (free-form, no assertions)"),
        f"channel   {'voice · ' + asr if asr else 'text'}",
        f"saving to {save_path}",
        "",
        "You are the caller. The agent speaks first once it is ready.",
        "Say '/stop' (or ctrl-c, or ctrl-d) to hang up; the call is still saved.",
    ]
    for line in lines:
        channel.note(line) if line else print()


def summarise(channel: Channel, task: Any, simulation: Any, save_path: Path) -> None:
    """Say what the call produced, without overclaiming what it proves."""
    channel.note("")
    channel.note(f"call ended after {len(simulation.messages)} messages")

    graded = bool(task.evaluation_criteria)
    reward_info = getattr(simulation, "reward_info", None)
    if reward_info is not None:
        if graded:
            channel.note(f"reward    {reward_info.reward}")
            breakdown = getattr(reward_info, "reward_breakdown", None) or {}
            for key, value in breakdown.items():
                name = getattr(key, "value", key)
                channel.note(f"  {name:<22} {value}")
        else:
            # A task with no assertions cannot fail one. Printing 1.0 here
            # without saying so would manufacture a pass out of nothing.
            channel.note(
                f"reward    {reward_info.reward} — vacuous: this task has no "
                "assertions. Use --task to run a graded case."
            )

    show_actions(channel, simulation)

    # The flight recorder is the part of the suite that reads a transcript
    # rather than a chart, so it is the part that has anything to say about a
    # conversation that had no assertions attached to it.
    try:
        from rx_bench.harness.scorers import score_readback  # noqa: PLC0415

        readback = score_readback(save_path)
    except Exception as exc:  # noqa: BLE001 - reporting, never fatal
        channel.note(f"(readback scorer unavailable: {exc})")
        return

    writes = readback.get("total_writes", 0)
    if not writes:
        channel.note("no chart writes in this call — nothing for the recorder to check")
        return

    channel.note(f"chart writes {writes}")
    # Two separate things, and conflating them is the mistake the recorder
    # exists to prevent: what the agent CLAIMED about its own certainty, and
    # what a transcript state machine says actually happened.
    for name, count in (readback.get("certainty_counts") or {}).items():
        if count:
            channel.note(f"  claimed {name:<14} {count}")
    for name, count in (readback.get("cross_check_counts") or {}).items():
        if count:
            channel.note(f"  audited {name:<14} {count}")

    channel.note("")
    channel.note("score the whole suite's way with:")
    channel.note(f"  rxbench score {save_path}")


def show_actions(channel: Channel, simulation: Any) -> None:
    """List what the agent did to the chart while you were talking.

    Tool calls never reach the caller --- the orchestrator routes them to the
    environment --- so nothing on screen during the call reveals them. A voice
    agent whose actions are invisible to the person it is acting for is the
    whole problem, so they get printed the moment the call ends.
    """
    actions = []
    for message in simulation.messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            actions.append((call.name, call.arguments or {}))
    if not actions:
        channel.note("no tool calls in this call")
        return

    channel.note(f"actions taken ({len(actions)})")
    for name, arguments in actions:
        detail = ", ".join(
            f"{k}={v}" for k, v in arguments.items() if not isinstance(v, (dict, list))
        )
        if len(detail) > 120:
            detail = detail[:117] + "…"
        channel.note(f"  ⚙ {name}({detail})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def save_partial(task: Any, info: Any, save_path: Path, started: datetime) -> int:
    """Write whatever of the call actually happened.

    Not a substitute for a completed run: the environment was never evaluated,
    so there is no reward and the chart state is whatever the crash left. The
    transcript is real, though, and the transcript-reading scorers --- the
    flight recorder above all --- work on it unchanged.
    """
    from tau2.data_model.simulation import Results, SimulationRun, TerminationReason

    captured = list(HumanUser.latest.transcript) if HumanUser.latest else []
    if not captured:
        return 0

    ended = datetime.now(timezone.utc)
    simulation = SimulationRun(
        id=f"partial-{started.strftime('%Y%m%d-%H%M%S')}",
        task_id=task.id,
        start_time=started.isoformat(),
        end_time=ended.isoformat(),
        duration=(ended - started).total_seconds(),
        termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
        messages=captured,
        seed=None,
        # No reward_info: the call never reached evaluation, and an invented
        # zero would read as a failed case rather than an interrupted one.
    )
    results = Results(info=info, tasks=[task], simulations=[simulation])
    partial_path = save_path.with_name("partial-results.json")
    partial_path.write_text(results.model_dump_json(indent=2))
    print(f"  {len(captured)} messages salvaged -> {partial_path}", file=sys.stderr)
    print(
        "  No reward: the call never reached evaluation. Transcript scorers still read it.",
        file=sys.stderr,
    )
    return len(captured)


def report_call_failure(exc: BaseException, save_dir: Path) -> int:
    """Explain a mid-call crash instead of dumping sixty frames on the caller.

    The conversation is genuinely lost when this happens --- tau2 builds the
    ``SimulationRun`` at the end of the orchestrator loop, so there is no
    partial transcript to rescue --- and saying so is the only honest move. The
    common cause is the local proxy rejecting auth in bursts, which is a known
    condition of this setup and not a fault in the call.
    """
    name = type(exc).__name__
    detail = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    print(f"\ncall failed mid-conversation: {name}", file=sys.stderr)
    if detail:
        print(f"  {detail[:200]}", file=sys.stderr)
    if "AuthenticationError" in name or "authentication" in detail.lower():
        print(
            "  Some gateways reject auth in episodic bursts; the same call usually\n"
            "  succeeds on a retry. Check it is up:\n"
            "    curl -s -o /dev/null -w '%{http_code}\\n' $ANTHROPIC_API_BASE/v1/messages \\\n"
            "      -X POST -H 'content-type: application/json' \\\n"
            "      -d '{\"model\":\"x\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'",
            file=sys.stderr,
        )
    print(
        "  tau2 assembles a run only after the loop finishes, so the completed\n"
        f"  results.json does not exist. Per-turn logs, if any, are under {save_dir}",
        file=sys.stderr,
    )
    return 3

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="talk",
        description="Hold a live conversation with the medical_reception agent.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="use the microphone and speak replies instead of the terminal",
    )
    parser.add_argument(
        "--asr",
        choices=["deepgram", "whisper"],
        default=None,
        help="force a transcription backend (default: deepgram if keyed, else whisper)",
    )
    parser.add_argument(
        "--keep-audio",
        type=Path,
        default=None,
        help="keep each captured turn as a wav in this directory",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="force the speech-detection RMS threshold instead of measuring the room",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="run inside a real graded case (default: free-form walk-in)",
    )
    parser.add_argument("--list-tasks", action="store_true", help="list case ids and exit")
    parser.add_argument("--model", default=DEFAULT_AGENT_LLM, help="agent model id")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument(
        "--retries",
        type=int,
        default=6,
        help="agent LLM attempts, with exponential backoff (default 6)",
    )
    parser.add_argument(
        "--evaluate",
        default="env",
        help=(
            "tau2 evaluation type. 'env' checks the chart and needs no judge; "
            "'all_with_nl_assertions' also runs the LLM judge (extra model calls)"
        ),
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "live",
        help="where session directories are written",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def make_channel(args: argparse.Namespace) -> tuple[Channel, Optional[str]]:
    if not args.voice:
        return TextChannel(), None
    channel = VoiceChannel(
        asr_backend=args.asr,
        keep_audio=args.keep_audio,
        threshold_override=args.vad_threshold,
    )
    return channel, channel.asr_name


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Import-time logging was already muted at WARNING; this honours the flag.
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if args.list_tasks:
        return list_tasks()

    try:
        channel, asr = make_channel(args)
    except ASRUnavailable as exc:
        print(f"voice channel unavailable:\n{exc}", file=sys.stderr)
        return 2

    from tau2.data_model.simulation import Results
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.batch import run_single_task
    from tau2.runner.helpers import get_info

    registry.register_user(bind_channel(channel), USER_NAME)

    task = load_task(args.task)
    config = build_config(args)
    info = get_info(config)
    policy_sha = policy_fingerprint(info.environment_info.policy)

    now = datetime.now(timezone.utc)
    save_dir = session_dir(args.save_root, task.id, now)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "results.json"

    banner(
        channel,
        task=task,
        config=config,
        policy_sha=policy_sha,
        asr=asr,
        save_path=save_path,
    )

    try:
        evaluation_type = EvaluationType(args.evaluate)
    except ValueError:
        valid = ", ".join(e.value for e in EvaluationType)
        raise SystemExit(f"--evaluate must be one of: {valid}")

    try:
        simulation = run_single_task(
            config,
            task,
            evaluation_type=evaluation_type,
            save_dir=save_dir,
        )
    except Exception as exc:  # noqa: BLE001 - diagnosed, not swallowed
        channel.close()
        code = report_call_failure(exc, save_dir)
        save_partial(task, info, save_path, now)
        # A failure that salvaged nothing leaves an empty directory behind, and
        # a results tree littered with empty run dirs makes it harder to find
        # the runs that did happen.
        if not any(save_dir.iterdir()):
            save_dir.rmdir()
        return code

    results = Results(info=info, tasks=[task], simulations=[simulation])
    save_path.write_text(results.model_dump_json(indent=2))

    summarise(channel, task, simulation, save_path)
    channel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
