"""The experiment dashboard.

    python -m rx_bench.live.bench            # then open http://127.0.0.1:8990

One page that answers the question the scorecard alone cannot: *is the agent
getting better over time?* It hashes the files that define the agent
(``policy.md`` + the domain's Python + the model id) into a version id,
launches the ordinary ``scripts/run.sh`` pipeline against a chosen split,
records the resulting scorecard against that version, and charts the history.

The framing is a backtest. The ``backtest`` split (77 cases) is the set you
iterate against; the ``oos`` split (22 cases, a deterministic per-suite
holdout — see :func:`merge_cases.oos_holdout`) is the final exam you run once
per version. The page refuses to launch ``oos`` without an explicit
confirmation and counts how many times it has been run per version, because a
held-out set you peek at every commit is not held out — it is just more
training signal, and the number it produces stops predicting anything.

Nothing here re-implements the pipeline. A run is a subprocess of the same
``run.sh`` a terminal user runs — same merge, same vacuity check, same tau2
invocation, same concurrency and step caps — with ``SAVE_TO`` pinned so a
killed run resumes instead of restarting. The server only adds identity
(which agent was this?) and memory (what did the last twenty runs say?).
"""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root

from rx_bench.harness import experiments  # noqa: E402
from rx_bench.harness.common import TAU2_ROOT  # noqa: E402

BENCH_HTML = Path(__file__).with_name("bench.html")
RUN_SH = PROJECT_ROOT / "scripts" / "run.sh"
SIMULATIONS_DIR = TAU2_ROOT / "data" / "simulations"
DEFAULT_PORT = 8990
DEFAULT_AGENT_LLM = "gpt-4.1-2025-04-14"
REMOTE_ENV = "RXBENCH_ALLOW_REMOTE"
OOS_CONFIRM_TOKEN = "I_UNDERSTAND_OOS"
OOS_PEEKS_DIR = PROJECT_ROOT / "results" / "oos_peeks"

#: Rough wall-clock per case at CONCURRENCY=2 against the local proxy, so the
#: page can warn that `base` is a ~40 minute commitment, not a click.
SECONDS_PER_CASE_ESTIMATE = 25

_PROGRESS_RE = re.compile(r"\d+%\||\bit/s\b|\bs/it\b")


# ---------------------------------------------------------------------------
# Event bus (durable milestones + transient log stream)
# ---------------------------------------------------------------------------


@dataclass
class EventBus:
    """Like the call console's Bus, with one distinction it cannot afford:
    transient events. A benchmark run prints thousands of log lines; replaying
    them all to every reloading browser would drown the milestones that
    matter. Durable events (run started/finished/recorded) are kept and
    replayed; log and progress lines go only to whoever is watching live."""

    history: list[dict] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, kind: str, transient: bool = False, **payload: Any) -> dict:
        event = {
            "seq": -2 if transient else 0,
            "kind": kind,
            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **payload,
        }
        with self.lock:
            if not transient:
                event["seq"] = len(self.history)
                self.history.append(event)
            for subscriber in list(self.subscribers):
                subscriber.put(event)
        return event

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        subscriber: queue.Queue = queue.Queue()
        with self.lock:
            self.subscribers.append(subscriber)
            return list(self.history), subscriber

    def detach(self, subscriber: queue.Queue) -> None:
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------


class BenchRun:
    """One subprocess of run.sh, streamed to the page and recorded on success."""

    def __init__(
        self,
        bus: EventBus,
        *,
        split: str,
        trials: int,
        agent_llm: str,
        user_llm: str,
    ):
        self.bus = bus
        self.split = split
        self.trials = trials
        self.agent_llm = agent_llm
        self.user_llm = user_llm
        # The version is frozen at launch. If someone edits policy.md while the
        # run is in flight, the run is *neither* version's measurement; that is
        # detected at the end and stamped on the record rather than ignored.
        self.version = experiments.agent_version(agent_llm)
        existing = {
            r.get("run_name")
            for r in experiments.load_experiments()
            if r.get("run_name")
        }
        self.run_name = experiments.run_name_for(self.version["id"], split, existing)
        self.results_path = SIMULATIONS_DIR / self.run_name / "results.json"
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.state = "idle"
        self.stopped_by_user = False

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "trials": self.trials,
            "run_name": self.run_name,
            "version": self.version,
            "agent_llm": self.agent_llm,
            "user_llm": self.user_llm,
            "state": self.state,
            "resuming": self.results_path.exists(),
        }

    def start(self) -> None:
        self.state = "running"
        self.thread = threading.Thread(target=self._run, name="bench-run", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        # A killed tau2 run is recoverable: SAVE_TO is deterministic in
        # (version, split), so the next launch of the same pair resumes it.
        self.stopped_by_user = True
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()

    # -- internals -----------------------------------------------------------

    def _run(self) -> None:
        bus = self.bus
        bus.emit("run_started", **self.describe())
        try:
            env = dict(os.environ)
            env.update(
                {
                    "SAVE_TO": self.run_name,
                    "AGENT_LLM": self.agent_llm,
                    "USER_LLM": self.user_llm,
                }
            )
            self.proc = subprocess.Popen(
                ["bash", str(RUN_SH), self.split, str(self.trials)],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # so stop() can signal the whole group
            )
            self._pump(self.proc)
            code = self.proc.wait()
            if code == 0:
                self._record()
                self.state = "finished"
            elif self.stopped_by_user:
                self.state = "stopped"
                bus.emit(
                    "run_stopped",
                    run_name=self.run_name,
                    note="stopped; relaunching the same split on the same "
                    "version resumes from the completed cases",
                )
            else:
                self.state = "failed"
                bus.emit("run_failed", run_name=self.run_name, exit_code=code)
        except Exception as exc:  # noqa: BLE001 - surfaced to the page
            self.state = "failed"
            bus.emit(
                "run_failed",
                run_name=self.run_name,
                error=f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc()[-2000:],
            )

    def _pump(self, proc: subprocess.Popen) -> None:
        """Stream subprocess output, split on both newlines and the carriage
        returns tqdm animates with — otherwise a progress bar is one enormous
        line that arrives only when the run ends."""
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            buf += chunk
            *segments, buf = re.split(rb"[\r\n]", buf)
            for seg in segments:
                self._line(seg.decode("utf-8", "replace").strip())
        if buf.strip():
            self._line(buf.decode("utf-8", "replace").strip())

    def _line(self, line: str) -> None:
        if not line:
            return
        line = line[:400]
        if line.startswith("==>") or line.startswith("!!"):
            self.bus.emit("milestone", text=line)
        elif _PROGRESS_RE.search(line):
            # tqdm frames: keep the page's single progress line fresh without
            # flooding the log with hundreds of redraws of the same bar.
            self.bus.emit("run_progress", transient=True, text=line)
        else:
            self.bus.emit("run_log", transient=True, text=line)

    def _record(self) -> None:
        if not self.results_path.exists():
            self.bus.emit(
                "run_failed",
                run_name=self.run_name,
                error=f"run.sh exited 0 but {self.results_path} does not exist",
            )
            self.state = "failed"
            return
        from scorecard import build_scorecard  # noqa: PLC0415 (harness path)

        scorecard = build_scorecard(self.results_path)
        scorecard_path = self.results_path.parent / "scorecard.json"
        scorecard_path.write_text(json.dumps(scorecard, indent=2, default=str) + "\n")

        record_version = dict(self.version)
        drifted = experiments.changed_files(
            experiments.agent_version(self.agent_llm), self.version
        )
        if drifted:
            # The measurement is tainted, not discarded: the record says so.
            record_version["files_changed_during_run"] = drifted

        path = experiments.record_run(
            version=record_version,
            split=self.split,
            trials=self.trials,
            run_name=self.run_name,
            results_path=self.results_path,
            scorecard=scorecard,
            user_llm=self.user_llm,
        )
        self.bus.emit(
            "recorded",
            registry_file=path.name,
            run_name=self.run_name,
            split=self.split,
            version=record_version,
            summary=experiments.summarize_scorecard(scorecard),
            files_changed_during_run=drifted,
        )


# ---------------------------------------------------------------------------
# Investigation (per-case drill-down into a recorded run)
# ---------------------------------------------------------------------------


def _sim_scored(sim: dict[str, Any]) -> bool:
    """Same rule as the scorecard: a crashed sim (infrastructure_error /
    reward_info null) is not a failure, it is a non-measurement."""
    if sim.get("termination_reason") == "infrastructure_error":
        return False
    return sim.get("reward_info") is not None


def investigation_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per simulation in a results.json: the case list of a run."""
    from rx_bench.harness.common import suite_of  # noqa: PLC0415

    tasks_by_id = {t.get("id"): t for t in results.get("tasks") or []}
    rows: list[dict[str, Any]] = []
    for sim in results.get("simulations") or []:
        ri = sim.get("reward_info") or {}
        env = ri.get("env_assertions") or []
        nl = ri.get("nl_assertions") or []
        scored = _sim_scored(sim)
        task = tasks_by_id.get(sim.get("task_id")) or {}
        rows.append(
            {
                "task_id": sim.get("task_id"),
                "trial": sim.get("trial"),
                "suite": suite_of(sim.get("task_id") or ""),
                "purpose": (task.get("description") or {}).get("purpose"),
                "scored": scored,
                "passed": bool(scored and ri.get("reward") == 1.0),
                "reward": ri.get("reward") if scored else None,
                "termination_reason": sim.get("termination_reason"),
                "duration": sim.get("duration"),
                "messages": len(sim.get("messages") or []),
                "env_assertions_met": sum(1 for a in env if a.get("met")),
                "env_assertions_total": len(env),
                "nl_assertions_met": sum(1 for a in nl if a.get("met")),
                "nl_assertions_total": len(nl),
            }
        )
    rows.sort(key=lambda r: (r["task_id"] or "", r["trial"] or 0))
    return rows


def case_transcript(
    results: dict[str, Any], task_id: str, trial: int
) -> Optional[dict[str, Any]]:
    """The full conversation and grading detail for one simulation, or None.

    Message fields are picked explicitly rather than passed through — a tau2
    message drags along audio paths, raw model payloads and usage records that
    a transcript viewer has no business shipping to the browser.
    """
    from rx_bench.harness.common import suite_of  # noqa: PLC0415

    sim = next(
        (
            s
            for s in results.get("simulations") or []
            if s.get("task_id") == task_id and s.get("trial") == trial
        ),
        None,
    )
    if sim is None:
        return None
    task = next(
        (t for t in results.get("tasks") or [] if t.get("id") == task_id), {}
    )
    ri = sim.get("reward_info") or {}
    scenario = task.get("user_scenario") or {}

    messages: list[dict[str, Any]] = []
    for m in sim.get("messages") or []:
        entry: dict[str, Any] = {
            "role": m.get("role"),
            "content": m.get("content"),
            "turn_idx": m.get("turn_idx"),
        }
        calls = [
            {"name": tc.get("name"), "arguments": tc.get("arguments")}
            for tc in m.get("tool_calls") or []
        ]
        if calls:
            entry["tool_calls"] = calls
        if m.get("role") == "tool":
            entry["error"] = bool(m.get("error"))
        messages.append(entry)

    return {
        "task_id": task_id,
        "trial": trial,
        "suite": suite_of(task_id),
        "purpose": (task.get("description") or {}).get("purpose"),
        "user_instructions": scenario.get("instructions"),
        "persona": scenario.get("persona"),
        "scored": _sim_scored(sim),
        "passed": bool(_sim_scored(sim) and ri.get("reward") == 1.0),
        "reward": ri.get("reward") if _sim_scored(sim) else None,
        "termination_reason": sim.get("termination_reason"),
        "duration": sim.get("duration"),
        "env_assertions": [
            {
                "func_name": (a.get("env_assertion") or {}).get("func_name"),
                "arguments": (a.get("env_assertion") or {}).get("arguments"),
                "met": a.get("met"),
                "message": (a.get("env_assertion") or {}).get("message"),
            }
            for a in ri.get("env_assertions") or []
        ],
        "nl_assertions": [
            {
                "assertion": a.get("nl_assertion"),
                "met": a.get("met"),
                "justification": a.get("justification"),
            }
            for a in ri.get("nl_assertions") or []
        ],
        "messages": messages,
    }


def investigate_payload(
    params: dict[str, str], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    """Serve /api/investigate from query params. Returns (payload, status).

    The ``run`` param must name a registry file that actually exists in the
    loaded registry — the results path always comes from the record, never
    from the request, so the endpoint cannot be steered at arbitrary files.
    """
    run_file = params.get("run") or ""
    record = next(
        (r for r in records if r.get("registry_file") == run_file), None
    )
    if record is None:
        return {"error": f"no recorded run named {run_file!r}"}, 404
    if record.get("oos") and params.get("confirm_oos") != OOS_CONFIRM_TOKEN:
        return {
            "error": "OOS task contents require exact confirmation",
            "confirm_oos": OOS_CONFIRM_TOKEN,
        }, 403
    results_path = Path(record.get("results_path") or "")
    try:
        results = json.loads(results_path.read_text())
    except (OSError, ValueError) as exc:
        return {
            "error": f"results file for {run_file} is unreadable: {exc}",
            "results_path": str(results_path),
        }, 404

    task_id = params.get("task")
    if not task_id:
        return {"run": record, "cases": investigation_rows(results)}, 200

    try:
        trial = int(params.get("trial") or 0)
    except ValueError:
        return {"error": "trial must be an integer"}, 400
    detail = case_transcript(results, task_id, trial)
    if detail is None:
        return {
            "error": f"no simulation for task {task_id!r} trial {trial} in {run_file}"
        }, 404
    return detail, 200


# ---------------------------------------------------------------------------
# Request validation (pure, unit-testable without a server)
# ---------------------------------------------------------------------------


def record_oos_peek(run: BenchRun, status: str) -> None:
    """Record every OOS launch, including stopped and failed attempts."""
    if run.split not in experiments.OOS_SPLITS:
        return
    OOS_PEEKS_DIR.mkdir(parents=True, exist_ok=True)
    path = OOS_PEEKS_DIR / f"{run.run_name}.json"
    path.write_text(json.dumps({
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_id": run.version["id"],
        "run_name": run.run_name,
        "status": status,
    }, indent=2))


def oos_peek_count(version_id: Optional[str] = None) -> int:
    count = 0
    if not OOS_PEEKS_DIR.exists():
        return count
    for path in OOS_PEEKS_DIR.glob("*.json"):
        try:
            peek = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if version_id and peek.get("version_id") != version_id:
            continue
        count += 1
    return count


def validate_run_request(
    body: dict[str, Any],
    splits: dict[str, int],
    running: bool,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Returns (params, None) or (None, error message)."""
    if running:
        return None, "a benchmark run is already in progress"
    split = (body.get("split") or "").strip()
    if split not in splits:
        return None, f"unknown split {split!r}; choose one of {sorted(splits)}"
    try:
        trials = int(body.get("trials") or 1)
    except (TypeError, ValueError):
        return None, "trials must be an integer"
    if not 1 <= trials <= 5:
        return None, "trials must be between 1 and 5"
    if split in experiments.OOS_SPLITS and body.get("confirm_oos") != OOS_CONFIRM_TOKEN:
        return None, (
            "the oos split is the held-out final exam — confirm you mean to "
            f"spend a look at it (confirm_oos: {OOS_CONFIRM_TOKEN!r})"
        )
    agent_llm = (body.get("agent_llm") or "").strip() or DEFAULT_AGENT_LLM
    user_llm = (body.get("user_llm") or "").strip() or agent_llm
    return (
        {"split": split, "trials": trials, "agent_llm": agent_llm, "user_llm": user_llm},
        None,
    )


def build_state(current: Optional[BenchRun]) -> dict[str, Any]:
    """Everything the page needs to draw itself, in one round trip."""
    version = experiments.agent_version(
        current.agent_llm if current else DEFAULT_AGENT_LLM
    )
    runs = experiments.load_experiments()
    last = next(
        (r for r in reversed(runs) if (r.get("version") or {}).get("id")), None
    )
    splits = experiments.domain_split_counts()
    return {
        "version": version,
        "changed_since_last_run": (
            experiments.changed_files(version, last["version"]) if last else None
        ),
        "splits": splits,
        "seconds_per_case_estimate": SECONDS_PER_CASE_ESTIMATE,
        "oos_splits": sorted(experiments.OOS_SPLITS),
        "oos_peeks_this_version": oos_peek_count(version["id"]),
        "oos_peeks_total": oos_peek_count(),
        "oos_confirm_token": OOS_CONFIRM_TOKEN,
        "defaults": {"agent_llm": DEFAULT_AGENT_LLM, "trials": 1, "split": "smoke"},
        "experiments": runs,
        "running": current.describe() if current and current.running else None,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    bus: EventBus
    current: Optional[BenchRun] = None
    run_lock = threading.Lock()
    server_version = "medvoice-bench/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """Silence per-request logging; the page is the log."""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode(), "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, BENCH_HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(build_state(Handler.current))
        elif self.path.startswith("/api/investigate"):
            from urllib.parse import parse_qs, urlsplit  # noqa: PLC0415

            query = parse_qs(urlsplit(self.path).query)
            params = {k: v[0] for k, v in query.items() if v}
            payload, status = investigate_payload(
                params, experiments.load_experiments()
            )
            self._json(payload, status)
        elif self.path == "/events":
            self._stream()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/run":
            body = self._body()
            with Handler.run_lock:
                running = Handler.current is not None and Handler.current.running
                params, error = validate_run_request(
                    body, experiments.domain_split_counts(), running
                )
                if error:
                    self._json({"ok": False, "error": error}, 409 if running else 400)
                    return
                run = BenchRun(self.bus, **params)
                Handler.current = run
                record_oos_peek(run, "started")
                run.start()
            self._json({"ok": True, **run.describe()})
        elif self.path == "/stop":
            current = Handler.current
            if not current or not current.running:
                self._json({"ok": False, "error": "no run in progress"}, 409)
            else:
                current.stop()
                self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        history, subscriber = self.bus.subscribe()
        try:
            for event in history:
                self._event(event)
            self._event(
                {
                    "seq": -1,
                    "kind": "synced",
                    "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "replayed": len(history),
                }
            )
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self._event(event)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.bus.detach(subscriber)

    def _event(self, event: dict) -> None:
        self.wfile.write(f"data: {json.dumps(event, default=str)}\n\n".encode())
        self.wfile.flush()


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="bench")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    from rx_bench.live.web import allow_bind  # noqa: PLC0415

    if not allow_bind(args.host):
        return 2

    Handler.bus = EventBus()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    version = experiments.agent_version(DEFAULT_AGENT_LLM)
    print(f"experiment dashboard on http://{args.host}:{args.port}")
    print(f"  agent version {version['id']}  ({len(version['files'])} files + model id)")
    print("  ctrl-c here to stop the server; a killed benchmark run resumes")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
