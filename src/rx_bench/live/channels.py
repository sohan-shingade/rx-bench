"""I/O channels for a live conversation with the agent.

A channel is the seam between a person and the orchestrator. Everything above it
--- the agent, its tools, ``policy.md``, the flight recorder, the scorers --- is
identical whether the words arrive as keystrokes or as speech, which is exactly
the property worth having: a voice demo that ran a different agent would prove
nothing about the numbers in the scorecard.

    TextChannel()    stdin/stdout
    VoiceChannel()   microphone -> ASR -> agent -> macOS `say`

:class:`VoiceChannel` reports which ASR backend it resolved and refuses to
silently downgrade. That matters more here than usual: the whole project is
about uncertainty that gets laundered on the way into a record, and an ASR that
quietly fell back to a weaker model is precisely that failure happening to the
evidence itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

SAY = "/usr/bin/say"

#: Mic capture rate. 16 kHz mono matches the rx_bench.audio corpus, so a
#: turn captured here can be dropped straight into that pipeline.
SAMPLE_RATE = 16_000

#: Silence detection. A turn ends after this much quiet, so the speaker does not
#: have to press anything to hand the floor back.
#:
#: Fallback speech threshold, used only if calibration cannot run. Real rooms
#: vary by more than an order of magnitude --- a measured 0.0215 ambient floor
#: in an ordinary office is already above this, which would make every silence
#: read as speech and turn-taking never work. See :meth:`VoiceChannel.calibrate`.
SILENCE_RMS = 0.012

#: Ambient floor is multiplied by this to get the speech threshold. Applied to
#: the LOUDEST calibration block, not the median: a room's median is quiet and
#: its transients are not, and a single keyboard tap above threshold is enough
#: to open a turn that then never closes.
CALIBRATION_HEADROOM = 2.0
CALIBRATION_SECONDS = 1.0

#: Consecutive above-threshold blocks (0.1 s each) required to call it speech.
#: Transients are loud and short; syllables are not.
SPEECH_ONSET_BLOCKS = 3

SILENCE_SECONDS = 1.4
MAX_TURN_SECONDS = 45.0
MIN_TURN_SECONDS = 0.4

#: Give up waiting for a turn that never starts. Without this, a muted or
#: permission-denied microphone looks exactly like a person thinking, for the
#: full MAX_TURN_SECONDS, which is a miserable way to discover the problem.
NO_SPEECH_SECONDS = 8.0


class Channel(Protocol):
    """How the human side of the conversation is read and written."""

    def say(self, text: str) -> None:
        """Deliver an agent turn to the person."""

    def listen(self) -> str:
        """Block until the person produces a turn. Empty string means hang up."""

    def note(self, text: str) -> None:
        """Out-of-band commentary (tool calls, status). Never spoken aloud."""

    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")


@dataclass
class TextChannel:
    """Keyboard in, terminal out."""

    colour: bool = field(default_factory=_supports_colour)
    prompt: str = "you> "

    def _paint(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.colour else text

    def say(self, text: str) -> None:
        label = self._paint("agent", CYAN + BOLD)
        body = "\n       ".join(text.strip().splitlines())
        print(f"\n[{label}] {body}\n")

    def listen(self) -> str:
        try:
            return input(self._paint(self.prompt, BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""

    def note(self, text: str) -> None:
        print(self._paint(f"       · {text}", DIM))

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------


class ASRUnavailable(RuntimeError):
    """No transcription backend could be resolved."""


def _deepgram_backend() -> Optional[tuple[str, Any]]:
    """Deepgram nova-3-medical, if a key is present. The production path."""
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        return None
    import requests  # noqa: PLC0415  -- optional, only on this path

    model = os.environ.get("DEEPGRAM_MODEL", "nova-3-medical")
    url = (
        f"https://api.deepgram.com/v1/listen?model={model}"
        "&smart_format=true&punctuate=true"
    )

    def transcribe(path: Path) -> str:
        response = requests.post(
            url,
            headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
            data=path.read_bytes(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        alternatives = payload["results"]["channels"][0]["alternatives"]
        return alternatives[0]["transcript"] if alternatives else ""

    return f"deepgram/{model}", transcribe


def _whisper_backend() -> Optional[tuple[str, Any]]:
    """Local faster-whisper. Offline, no key, materially worse on drug names.

    Measured on the project's own corpus (``src/rx_bench/audio/corpus/renders``),
    against the ground truth "I take hydroxyzine, twenty-five milligrams, at
    bedtime":

        profile       base.en                       small.en
        clean         "hydroxazine 25 milligrams"   "hydroxazine, 25 milligrams"
        phone         "hydroxzines, 25 milligrams"  "hydroxyzines, 25 milligrams"
        noisy_phone   total loss                    total loss
        bad_line      total loss                    total loss

    So: on clean speech it hears the drug name approximately and the dose
    exactly, and on a degraded line it invents a sentence. ``small.en`` is the
    default because the drug name is the one token in the utterance that
    matters, and the extra ~0.7 s per turn is invisible in conversation. Set
    ``WHISPER_MODEL`` to trade back.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError:
        return None

    size = os.environ.get("WHISPER_MODEL", "small.en")
    model = WhisperModel(size, device="cpu", compute_type="int8")

    def transcribe(path: Path) -> str:
        segments, _ = model.transcribe(str(path), beam_size=5, language="en")
        return " ".join(segment.text for segment in segments).strip()

    return f"faster-whisper/{size}", transcribe


def resolve_asr(prefer: Optional[str] = None) -> tuple[str, Any]:
    """Pick a transcription backend, best first, and name the one chosen.

    The name is surfaced in the banner and recorded on the run. A number quoted
    from a whisper-transcribed conversation is not the number Deepgram would
    have produced, and nothing downstream can tell the difference unless the
    backend travels with the transcript.
    """
    builders = {"deepgram": _deepgram_backend, "whisper": _whisper_backend}
    order = [prefer] if prefer else ["deepgram", "whisper"]
    tried = []
    for name in order:
        builder = builders.get(name)
        if builder is None:
            raise ASRUnavailable(f"unknown ASR backend {name!r}")
        tried.append(name)
        resolved = builder()
        if resolved:
            return resolved
    raise ASRUnavailable(
        "no ASR backend available (tried: " + ", ".join(tried) + ").\n"
        "  Deepgram:  export DEEPGRAM_API_KEY=...\n"
        "  offline:   uv pip install faster-whisper\n"
        "  or run the text channel instead: python -m rx_bench.live.talk"
    )


@dataclass
class VoiceChannel:
    """Microphone in, synthesised speech out.

    Turn-taking is silence-based rather than push-to-talk, so a demo does not
    require a free hand. Recording stops after ``SILENCE_SECONDS`` of quiet.
    """

    voice: str = "Samantha"
    rate: int = 175
    asr_backend: Optional[str] = None
    keep_audio: Optional[Path] = None
    echo_transcript: bool = True
    #: Force the speech threshold instead of measuring the room. Calibration
    #: takes one second and cannot see a noise burst that arrives later, so a
    #: loud venue may need this set by hand -- the per-turn "peak" figure in the
    #: no-speech message is what to set it above.
    threshold_override: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            import sounddevice  # noqa: F401, PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ASRUnavailable(
                "microphone capture needs sounddevice:\n"
                "  uv pip install sounddevice\n"
                "or run the text channel instead: python -m rx_bench.live.talk"
            ) from exc
        self._text = TextChannel()
        # First use downloads model weights. Without this line the program just
        # sits there and the operator assumes the microphone is broken.
        self._text.note("loading transcription backend…")
        self.asr_name, self._transcribe = resolve_asr(self.asr_backend)
        if self.asr_name.startswith("faster-whisper"):
            self._text.note(
                "local whisper will mangle drug names (measured: 'hydroxyzine' "
                "→ 'hydroxazine' on clean speech). Every turn prints what it "
                "heard — read that line before blaming the agent."
            )
        self._turn = 0
        self._peak = 0.0
        self.threshold = SILENCE_RMS
        if self.keep_audio:
            self.keep_audio.mkdir(parents=True, exist_ok=True)
        self.calibrate()

    def calibrate(self) -> float:
        """Measure the room and set the speech threshold from it.

        A fixed threshold does not survive contact with a real room. Measured
        ambient floor in an ordinary office was 0.0215 --- nearly twice the
        hardcoded default --- which makes silence indistinguishable from speech
        and turn-taking simply stop working. One second of listening fixes it.
        """
        if self.threshold_override is not None:
            self.threshold = self.threshold_override
            self._text.note(f"speech threshold forced to {self.threshold:.4f}")
            return self.threshold
        try:
            import numpy as np  # noqa: PLC0415
            import sounddevice as sd  # noqa: PLC0415

            block = int(SAMPLE_RATE * 0.1)
            levels = []
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block
            ) as stream:
                for _ in range(int(CALIBRATION_SECONDS / 0.1)):
                    chunk, _overflowed = stream.read(block)
                    levels.append(float(np.sqrt(np.mean(np.square(chunk)))))
            floor = max(levels)
        except Exception as exc:  # noqa: BLE001 - falls back, never fatal
            self._text.note(f"could not calibrate ({exc}); using default threshold")
            return self.threshold

        self.threshold = max(SILENCE_RMS, floor * CALIBRATION_HEADROOM)
        self._text.note(
            f"loudest ambient block {floor:.4f} → speech threshold {self.threshold:.4f}"
        )
        return self.threshold

    # -- output --------------------------------------------------------------

    def say(self, text: str) -> None:
        self._text.say(text)
        if not shutil.which(SAY):  # pragma: no cover - macOS only
            return
        # Spoken separately from the print so the transcript is on screen even
        # if TTS fails; a demo that goes silent should still be readable.
        subprocess.run(
            [SAY, "-v", self.voice, "-r", str(self.rate), text],
            check=False,
            timeout=180,
        )

    def note(self, text: str) -> None:
        self._text.note(text)

    # -- input ---------------------------------------------------------------

    def _record(self) -> Optional[Path]:
        """Capture one turn, ending on silence. None if nothing was said."""
        import numpy as np  # noqa: PLC0415
        import sounddevice as sd  # noqa: PLC0415

        frames: list[Any] = []
        silence_run = 0.0
        speech_seen = False
        onset_run = 0
        peak = 0.0
        block = int(SAMPLE_RATE * 0.1)

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block
        ) as stream:
            elapsed = 0.0
            while elapsed < MAX_TURN_SECONDS:
                chunk, _overflowed = stream.read(block)
                frames.append(chunk.copy())
                elapsed += block / SAMPLE_RATE
                level = float(np.sqrt(np.mean(np.square(chunk))))
                if level > self.threshold:
                    onset_run += 1
                    if onset_run >= SPEECH_ONSET_BLOCKS:
                        speech_seen = True
                    silence_run = 0.0
                elif speech_seen:
                    onset_run = 0
                    silence_run += block / SAMPLE_RATE
                    if silence_run >= SILENCE_SECONDS:
                        break
                else:
                    onset_run = 0
                    if elapsed >= NO_SPEECH_SECONDS:
                        self._peak = peak
                        return None
                peak = max(peak, level)

        self._peak = peak
        if not speech_seen:
            return None
        audio = np.concatenate(frames, axis=0)
        if len(audio) / SAMPLE_RATE < MIN_TURN_SECONDS:
            return None

        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        self._turn += 1
        if self.keep_audio:
            path = self.keep_audio / f"turn-{self._turn:03d}.wav"
        else:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            path = Path(handle.name)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(pcm.tobytes())
        return path

    def listen(self) -> str:
        while True:
            self._text.note("listening… (speak, then pause; ctrl-c to hang up)")
            try:
                path = self._record()
            except KeyboardInterrupt:
                print()
                return ""
            if path is None:
                # The peak level separates "nobody spoke" from "the microphone
                # is muted or was never granted permission" -- indistinguishable
                # otherwise, and the second one never fixes itself.
                if self._peak < self.threshold / 4:
                    self._text.note(
                        f"heard nothing at all (peak {self._peak:.4f}) — check the "
                        "input device and macOS microphone permission"
                    )
                else:
                    self._text.note("heard nothing — still listening")
                continue

            try:
                text = self._transcribe(path).strip()
            except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
                self._text.note(f"transcription failed ({exc}) — say that again")
                continue
            finally:
                if not self.keep_audio:
                    path.unlink(missing_ok=True)

            if not text:
                self._text.note("transcribed to nothing — say that again")
                continue
            if self.echo_transcript:
                # Showing what the ASR heard is not a nicety. When the agent
                # mishandles a drug name, this line is the only way to tell a
                # policy failure from a transcription failure.
                self._text.note(f"heard [{self.asr_name}]: {text}")
            return text

    def close(self) -> None:
        pass
