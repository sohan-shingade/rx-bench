"""A browser front end for the same live call.

    python -m rx_bench.live.web          # then open http://127.0.0.1:8975

``talk.py`` gives you one conversation in a terminal. This gives you the same
conversation with everything that is normally invisible put on screen next to
it: every tool call the agent makes and what came back, the framework's own log
stream, and the flight-recorder verdict at the end. You can type or speak into
the same box, and you can freeze the call between turns.

Nothing about the agent changes. This is a third :class:`Channel` --- keystrokes
and speech arrive over HTTP instead of stdin --- wrapped around the identical
``run_single_task`` call that ``talk.py`` makes, so a session held here saves
the same ``results.json`` and is read by the same scorers.

Two seams are worth naming, because both are places where a demo could quietly
stop showing the truth:

* Tool calls never reach the caller. The orchestrator routes them to the
  environment, so a channel cannot see them. :func:`instrument_environment`
  wraps ``Environment.get_response`` --- the single point every tool call passes
  through --- rather than reconstructing them afterwards from the saved run.
* Speech recognition happens in the browser (the Web Speech API), which is not
  the Deepgram or whisper path the CLI measures. The status bar says so. A drug
  name mangled here was mangled by a different transcriber than the one the
  numbers in the docs describe.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root

# talk.py mutes loguru and litellm at import time; importing it first means this
# module inherits a quiet baseline and can then add its own sink deliberately.
from rx_bench.live import talk  # noqa: E402
from rx_bench.live.human_user import HumanUser, unwrap_envelope  # noqa: E402

UI_HTML = Path(__file__).with_name("ui.html")
DEFAULT_PORT = 8975
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # ~17 min of 16 kHz mono; a turn is seconds
REMOTE_ENV = "RXBENCH_ALLOW_REMOTE"
OOS_CONFIRM_TOKEN = "I_UNDERSTAND_OOS"


def is_loopback_host(host: str) -> bool:
    """Whether every address for a bind host is loopback."""
    if host.lower() == "localhost":
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    import ipaddress

    return bool(infos) and all(ipaddress.ip_address(info[4][0]).is_loopback for info in infos)


def allow_bind(host: str) -> bool:
    if is_loopback_host(host):
        return True
    if os.environ.get(REMOTE_ENV) == "1":
        print(f"WARNING: {REMOTE_ENV}=1; unauthenticated control UI is remotely reachable")
        return True
    print(
        f"refusing non-loopback bind {host!r}; set {REMOTE_ENV}=1 only if you accept "
        "remote unauthenticated access",
        file=sys.stderr,
    )
    return False


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class Transcriber:
    """The browser's microphone, transcribed by the project's own ASR.

    The obvious way to do speech in a web page is the Web Speech API, and it is
    the wrong one here for two reasons. It fails outright --- ``network`` ---
    whenever Chrome cannot reach Google's speech service, which is every
    Chromium build without Google API keys and every restricted network. And
    when it does work it is a *fourth* transcriber, so a drug name mangled in
    the browser tells you nothing about the pipeline the benchmark measures.

    Audio is captured in the page, encoded as 16 kHz mono WAV --- the format
    both backends already take --- and posted here, where it goes through the
    same :func:`resolve_asr` that ``talk-voice.sh`` uses. Deepgram when
    ``DEEPGRAM_API_KEY`` is set, local faster-whisper otherwise, named on every
    turn either way.
    """

    def __init__(self, prefer: Optional[str] = None):
        self.prefer = prefer
        self.name: Optional[str] = None
        self.error: Optional[str] = None
        self._fn: Any = None
        self._lock = threading.Lock()
        self._resolved = False

    def resolve(self) -> None:
        """Load the backend once, on first use rather than at startup.

        faster-whisper takes seconds to load its weights, and a demo that
        cannot serve its own page until the model is in memory looks broken.
        """
        with self._lock:
            if self._resolved:
                return
            self._resolved = True
            from rx_bench.live.channels import ASRUnavailable, resolve_asr

            try:
                self.name, self._fn = resolve_asr(self.prefer)
            except ASRUnavailable as exc:
                self.error = str(exc)

    @property
    def ready(self) -> bool:
        return self._fn is not None

    def transcribe(self, wav: bytes) -> str:
        self.resolve()
        if not self._fn:
            raise RuntimeError(self.error or "no ASR backend")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(wav)
            path = Path(handle.name)
        try:
            return self._fn(path)
        finally:
            path.unlink(missing_ok=True)



# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


@dataclass
class Bus:
    """Fan-out of session events to any number of connected browsers.

    Every event is kept. A browser that connects late, or reloads mid-call,
    replays the whole session rather than joining a conversation already in
    progress with no idea what was said.
    """

    history: list[dict] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, kind: str, **payload: Any) -> dict:
        event = {
            "seq": 0,
            "kind": kind,
            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **payload,
        }
        with self.lock:
            event["seq"] = len(self.history)
            self.history.append(event)
            for subscriber in list(self.subscribers):
                subscriber.put(event)
        return event

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        subscriber: queue.Queue = queue.Queue()
        with self.lock:
            return list(self.history), subscriber

    def attach(self, subscriber: queue.Queue) -> None:
        with self.lock:
            self.subscribers.append(subscriber)

    def detach(self, subscriber: queue.Queue) -> None:
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    def reset(self) -> None:
        with self.lock:
            self.history.clear()


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


class WebChannel:
    """The caller is a browser tab.

    ``listen`` blocks the orchestrator thread exactly as stdin does, which is
    what makes pause meaningful: a half-duplex conversation can only be frozen
    where it already waits, and that is here, at the caller's turn. Once the
    agent starts a step it runs to completion --- pausing mid-step would mean
    holding a half-issued tool call, which is worse than not pausing at all.
    """

    def __init__(self, bus: Bus):
        self.bus = bus
        self.inbox: queue.Queue = queue.Queue()
        self.resumed = threading.Event()
        self.resumed.set()
        self.waiting = False
        self.hungup = False
        self._held: Optional[str] = None

    # -- output --------------------------------------------------------------

    def say(self, text: str) -> None:
        spoken, envelope = unwrap_envelope(text)
        if envelope:
            self.bus.emit(
                "warning",
                text=f"agent replied inside a {envelope} envelope; unwrapped for display",
            )
        self.bus.emit("agent", text=spoken, raw=text)

    def note(self, text: str) -> None:
        self.bus.emit("note", text=text)

    # -- input ---------------------------------------------------------------

    def listen(self) -> str:
        self.waiting = True
        self.bus.emit("awaiting_caller", paused=not self.resumed.is_set())
        try:
            while True:
                if self.hungup:
                    self._held = None
                    return ""
                if self._held is None:
                    try:
                        self._held = self.inbox.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    if self._held is None:
                        return ""  # hang up
                # Pause is checked AFTER the message is in hand, not before.
                # Checking first loses the race that matters: a pause pressed
                # while a turn is already queued would let that turn through,
                # which is exactly when someone reaches for pause.
                if not self.resumed.is_set():
                    self.bus.emit("held", text=self._held)
                    self.resumed.wait()
                    continue
                text, self._held = self._held, None
                self.bus.emit("caller", text=text)
                return text
        finally:
            self.waiting = False

    def submit(self, text: str) -> None:
        self.inbox.put(text)

    def hangup(self) -> None:
        # Checked before the held slot, so hanging up while paused ends the call
        # instead of first delivering the turn the pause was meant to withhold.
        self.hungup = True
        self.resumed.set()
        self.inbox.put(None)

    def pause(self) -> None:
        self.resumed.clear()
        self.bus.emit("state", paused=True)

    def resume(self) -> None:
        self.resumed.set()
        self.bus.emit("state", paused=False)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def instrument_environment(bus: Bus) -> Any:
    """Stream every tool call and its result to the browser.

    ``Environment.get_response`` is the one place a tool call is executed, so
    wrapping it catches all of them, in order, with the arguments actually sent
    and the value actually returned. Reconstructing this from the saved run
    afterwards would show the same calls but not while they happen, and "what
    did it just do" is the question the screen is there to answer.

    It also catches calls the *evaluator* makes: ``Environment.set_state``
    replays every mutating call into a fresh gold environment to compare final
    state. Those are real executions but they are not part of the conversation,
    and letting them scroll past under the caller's last turn would read as the
    agent doing the work twice. They are tagged by environment identity --- the
    first environment to execute anything is the live one --- and reported as a
    separate phase rather than dropped, because "the evaluator replayed your
    call" is true and worth seeing.
    """
    from tau2.environment.environment import Environment

    original = Environment.get_response
    live: dict[str, int] = {}

    def traced(self, message):  # noqa: ANN001
        live.setdefault("id", id(self))
        phase = "call" if id(self) == live["id"] else "evaluation"
        bus.emit(
            "tool_call",
            phase=phase,
            name=message.name,
            arguments=message.arguments or {},
            requestor=getattr(message, "requestor", None),
        )
        response = original(self, message)
        body = response.content or ""
        bus.emit(
            "tool_result",
            phase=phase,
            name=message.name,
            error=bool(getattr(response, "error", False)),
            content=body if len(body) <= 4000 else body[:4000] + "…",
        )
        return response

    Environment.get_response = traced
    return original


def install_log_sink(bus: Bus, level: str) -> int:
    """Put the framework's own log stream on screen."""
    from loguru import logger

    def sink(message: Any) -> None:
        record = message.record
        bus.emit(
            "log",
            level=record["level"].name,
            source=f"{record['name']}:{record['function']}:{record['line']}",
            text=record["message"],
        )

    return logger.add(sink, level=level)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """One call, run on its own thread so the HTTP server stays responsive."""

    def __init__(self, bus: Bus):
        self.bus = bus
        self.channel: Optional[WebChannel] = None
        self.thread: Optional[threading.Thread] = None
        self.state = "idle"
        self.meta: dict = {}
        self._lock = threading.Lock()
        self.user: Optional[HumanUser] = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, options: dict) -> tuple[dict, int]:
        with self._lock:
            if self.running or self.state == "starting":
                return {"ok": False, "error": "a call is already in progress"}, 409
            self.state = "starting"

        self.bus.reset()
        self.channel = WebChannel(self.bus)
        args = talk.parse_args([])
        args.task = options.get("task") or None
        args.model = options.get("model") or talk.DEFAULT_AGENT_LLM
        args.max_steps = int(options.get("max_steps") or 40)
        args.retries = int(options.get("retries") or 6)
        args.evaluate = options.get("evaluate") or "all_with_nl_assertions"
        args.confirm_oos = options.get("confirm_oos")

        try:
            task = talk.load_task(args.task, args.confirm_oos)
        except SystemExit as exc:
            self.state = "idle"
            return {"ok": False, "error": str(exc)}, 400

        self.state = "running"
        self.meta = {
            "task": task.id,
            "graded": bool(task.evaluation_criteria),
            "model": args.model,
            "max_steps": args.max_steps,
        }
        self.thread = threading.Thread(
            target=self._run, args=(args, task), name="call", daemon=True
        )
        self.thread.start()
        return {"ok": True, **self.meta}, 200

    def _run(self, args: Any, task: Any) -> None:
        from tau2.data_model.simulation import Results
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.registry import registry
        from tau2.runner.batch import run_single_task
        from tau2.runner.helpers import get_info

        bus, channel = self.bus, self.channel
        started = datetime.now(timezone.utc)
        restore = instrument_environment(bus)
        try:
            name = f"web_human_user_{started.strftime('%H%M%S%f')}"
            session: dict[str, HumanUser] = {}
            registry.register_user(talk.bind_channel(channel, session), name)
            config = talk.build_config(args)
            config.user = name

            info = get_info(config)
            self.user = session.get("user")
            save_dir = talk.session_dir(args.save_root, task.id, started)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "results.json"

            bus.emit(
                "started",
                task=task.id,
                graded=bool(task.evaluation_criteria),
                model=args.model,
                policy=talk.policy_fingerprint(info.environment_info.policy),
                save_path=str(save_path),
                purpose=(task.description.purpose if task.description else None),
            )

            simulation = run_single_task(
                config,
                task,
                evaluation_type=EvaluationType(args.evaluate),
                save_dir=save_dir,
            )
            self.user = session.get("user")
            results = Results(info=info, tasks=[task], simulations=[simulation])
            save_path.write_text(results.model_dump_json(indent=2))
            self._report(task, simulation, save_path)
            self.state = "finished"
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            self.state = "failed"
            bus.emit(
                "error",
                text=f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc()[-2000:],
            )
            try:
                self.user = session.get("user")
                salvaged = len(self.user.transcript) if self.user else 0
                if salvaged:
                    bus.emit(
                        "note",
                        text=f"{salvaged} messages salvaged; no reward — the call "
                        "never reached evaluation",
                    )
            except Exception:  # noqa: BLE001, S110 - best effort only
                pass
        finally:
            from tau2.environment.environment import Environment

            Environment.get_response = restore
            bus.emit("ended", state=self.state)

    def _report(self, task: Any, simulation: Any, save_path: Path) -> None:
        reward_info = getattr(simulation, "reward_info", None)
        summary: dict = {
            "messages": len(simulation.messages or []),
            "save_path": str(save_path),
            "graded": bool(task.evaluation_criteria),
        }
        if reward_info is not None:
            summary["reward"] = reward_info.reward
            breakdown = getattr(reward_info, "reward_breakdown", None) or {}
            summary["breakdown"] = {
                getattr(k, "value", str(k)): v for k, v in breakdown.items()
            }
        try:
            from rx_bench.harness.scorers import score_readback  # noqa: PLC0415

            readback = score_readback(save_path)
            summary["writes"] = readback.get("total_writes", 0)
            summary["claimed"] = {
                k: v for k, v in (readback.get("certainty_counts") or {}).items() if v
            }
            summary["audited"] = {
                k: v for k, v in (readback.get("cross_check_counts") or {}).items() if v
            }
        except Exception as exc:  # noqa: BLE001 - reporting, never fatal
            summary["readback_error"] = str(exc)
        self.bus.emit("summary", **summary)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    bus: Bus
    session: Session
    transcriber: "Transcriber"
    server_version = "medvoice-live/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """Silence per-request logging; the page is the log."""

    # -- helpers -------------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/tasks":
            confirmation = self.headers.get("X-RXBench-OOS-Confirm")
            self._json({"tasks": _task_index(confirmation)})
        elif self.path == "/config":
            self._json(_config(self.transcriber))
        elif self.path == "/events":
            self._stream()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/transcribe":
            self._transcribe()
            return

        body = self._body()
        channel = self.session.channel

        if self.path == "/start":
            payload, code = self.session.start(body)
            self._json(payload, code)
        elif self.path == "/say":
            text = (body.get("text") or "").strip()
            if not channel or not self.session.running:
                self._json({"ok": False, "error": "no call in progress"}, 409)
            elif not text:
                self._json({"ok": False, "error": "empty"}, 400)
            else:
                channel.submit(text)
                self._json({"ok": True})
        elif self.path == "/control":
            action = body.get("action")
            if not channel:
                self._json({"ok": False, "error": "no call in progress"}, 409)
                return
            if action == "pause":
                channel.pause()
            elif action == "resume":
                channel.resume()
            elif action == "hangup":
                channel.hangup()
            else:
                self._json({"ok": False, "error": f"unknown action {action!r}"}, 400)
                return
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def _transcribe(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            self._json({"ok": False, "error": "no audio"}, 400)
            return
        if length > MAX_AUDIO_BYTES:
            self._json({"ok": False, "error": "audio too large"}, 413)
            return
        wav = self.rfile.read(length)
        try:
            text = self.transcriber.transcribe(wav)
        except Exception as exc:  # noqa: BLE001 - reported to the browser
            self._json({"ok": False, "error": str(exc)}, 503)
            return

        backend = self.transcriber.name or "unknown"
        # Echoed on every turn for the same reason the CLI echoes it: a drug
        # name that came out wrong is either the agent's fault or the
        # transcriber's, and this line is the only way to tell them apart.
        self.bus.emit("heard", backend=backend, text=text)
        self._json({"ok": True, "text": text, "backend": backend})

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        history, subscriber = self.bus.subscribe()
        self.bus.attach(subscriber)
        try:
            for event in history:
                self._event(event)
            # Marks the end of the replay. A reloading browser is catching up,
            # not listening to a call in progress; without this boundary the
            # page reads the whole previous conversation aloud on every reload.
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
                    # Keep-alive comment. Without it an idle call looks like a
                    # dropped connection to every proxy between here and there.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self._event(event)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.bus.detach(subscriber)

    def _event(self, event: dict) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


def _task_index(confirmation: Optional[str] = None) -> list[dict]:
    out = [{"id": "", "label": "free-form walk-in (no assertions)"}]
    oos = talk.oos_task_ids()
    confirmed = confirmation == talk.OOS_CONFIRM_TOKEN
    for task in talk.all_tasks():
        if task.id in oos and not confirmed:
            continue
        purpose = ""
        if task.description and task.description.purpose:
            purpose = task.description.purpose.strip().splitlines()[0]
        out.append({"id": task.id, "label": f"{task.id} — {purpose[:80]}"})
    return out


def _config(transcriber: "Transcriber") -> dict:
    """What the page needs to capture audio the way the CLI does.

    The endpoint detection constants are read from ``channels.py`` rather than
    duplicated in JavaScript. They were tuned against a real room --- ambient
    noise there measured 0.0215 RMS, well above the old hardcoded floor --- and
    a browser that guessed its own would cut people off mid-sentence while the
    terminal did not, for no reason anyone could see.
    """
    from rx_bench.live import channels

    transcriber.resolve()
    return {
        "asr": {
            "name": transcriber.name,
            "ready": transcriber.ready,
            "error": transcriber.error,
        },
        "vad": {
            "sample_rate": channels.SAMPLE_RATE,
            "silence_rms": channels.SILENCE_RMS,
            "calibration_headroom": channels.CALIBRATION_HEADROOM,
            "calibration_seconds": channels.CALIBRATION_SECONDS,
            "speech_onset_blocks": channels.SPEECH_ONSET_BLOCKS,
            "silence_seconds": channels.SILENCE_SECONDS,
            "max_turn_seconds": channels.MAX_TURN_SECONDS,
            "min_turn_seconds": channels.MIN_TURN_SECONDS,
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="talk-web")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="framework log level streamed to the page (default INFO)",
    )
    parser.add_argument(
        "--asr",
        choices=["deepgram", "whisper"],
        default=None,
        help="force a transcription backend (default: Deepgram if keyed, else whisper)",
    )
    args = parser.parse_args(argv)
    if not allow_bind(args.host):
        return 2

    bus = Bus()
    install_log_sink(bus, args.log_level)
    Handler.bus = bus
    Handler.session = Session(bus)
    Handler.transcriber = Transcriber(args.asr)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"live call UI on http://{args.host}:{args.port}")
    print("  type or speak; ctrl-c here to stop the server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
