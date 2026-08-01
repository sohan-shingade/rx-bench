"""The speech channel of the pipeline agent class: TTS -> degrade -> ASR.

This module is the audio seam and nothing else. Given one user turn as text it

    1. renders it with macOS ``say``            (reused: rx_bench.audio.render)
    2. degrades it with a named channel profile (reused: rx_bench.audio.degrade)
    3. transcribes it with Deepgram             (reused: rx_bench.audio.asr_eval)

and returns the ASR transcript plus a full per-turn record (original text, WAV
paths, WER, timings, achieved SNR). The tau2 integration lives in
``pipeline_user.py``; the CLI in ``run_pipeline.py``.

Reuse notes
-----------
* ``tau2.voice.synthesis.audio_effects`` was evaluated and does NOT import
  without scipy (not a dependency of this package), so the project's own
  pure-numpy ``rx_bench.audio.degrade`` is used instead. It was
  built for exactly this corpus (16 kHz mono WAV, G.711 mu-law telephony
  chain) and its profiles are already referenced by the S1 case metadata.
* ``tau2.voice.transcription.transcribe`` was evaluated: it works but drags
  in the tau2 AudioData/TranscriptionConfig plumbing and resamples to 24 kHz
  for the OpenAI realtime path. The Deepgram reference transcriber in
  ``rx_bench.audio.asr_eval`` posts the WAV directly and is what
  the Tier-A corpus numbers were produced with, so the pipeline uses that,
  extended here with word-level confidences (the harness contract asks for
  them; ``asr_eval`` only returns the transcript string).

Determinism
-----------
TTS is deterministic given (voice, rate, text). Degradation is deterministic
given the profile seed, which is derived from (run seed, conversation key,
turn index). ASR is a network service and is NOT deterministic in principle;
the manifest records the model id and the response is logged verbatim so a run
is at least fully auditable.

Self-test (the LASA unit check)::

    python -m rx_bench.voice_pipeline.pipeline --selftest --profile noisy_phone
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root

from rx_bench.audio import degrade, render
from rx_bench.audio.asr_eval import classify_drug, wer

#: Run outputs live with the other run outputs, under the gitignored
#: data/simulations/ tree (see .gitignore; tau2 writes its runs there too).
RESULTS_ROOT = PROJECT_ROOT / "data" / "simulations" / "voice_pipeline"

DEFAULT_ASR_MODEL = "nova-3-medical"
PROFILES = sorted(degrade.PROFILES)  # clean | phone | noisy_phone | bad_line
#: Friendly alias used by the CLI: "realistic" is the telephony+noise profile.
PROFILE_ALIASES = {"realistic": "noisy_phone"}


def resolve_profile(name: str) -> str:
    resolved = PROFILE_ALIASES.get(name, name)
    if resolved not in degrade.PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; have {PROFILES} (alias: realistic=noisy_phone)"
        )
    return resolved


# ---------------------------------------------------------------------------
# Environment: the Deepgram key lives in the repo-root .env
# ---------------------------------------------------------------------------


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, never overrides the real env."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Deepgram with word confidences
# ---------------------------------------------------------------------------


class ASRError(RuntimeError):
    pass


def deepgram_transcribe_words(
    wav_path: str | Path,
    model: str = DEFAULT_ASR_MODEL,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    retries: int = 3,
) -> dict:
    """Transcribe one WAV. Returns ``{transcript, confidence, words, model}``.

    Same request shape as ``asr_eval.deepgram_transcriber`` (raw WAV POST,
    stdlib urllib), extended to keep the word-level confidences the harness
    contract calls for. Retries transient failures with linear backoff; a
    hard failure raises :class:`ASRError` rather than silently returning an
    empty transcript, because an empty transcript is itself a meaningful ASR
    outcome and must not be conflated with an outage.
    """
    key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise ASRError(
            "DEEPGRAM_API_KEY is not set (export it or put it in the repo-root .env)"
        )

    params = {
        "model": model,
        "smart_format": "false",
        "punctuate": "true",
        "language": "en",
    }
    url = "https://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)
    data = Path(wav_path).read_bytes()

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            alt = payload["results"]["channels"][0]["alternatives"][0]
            return {
                "transcript": alt.get("transcript") or "",
                "confidence": alt.get("confidence"),
                "words": [
                    {
                        "word": w.get("word"),
                        "confidence": w.get("confidence"),
                        "start": w.get("start"),
                        "end": w.get("end"),
                    }
                    for w in (alt.get("words") or [])
                ],
                "model": model,
            }
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode()[:200]
            except Exception:  # noqa: BLE001
                pass
            # 4xx will not get better on retry; say why immediately.
            if 400 <= exc.code < 500:
                raise ASRError(f"deepgram rejected the request ({exc.code}): {body}") from exc
            last = exc
        except Exception as exc:  # noqa: BLE001 - network layer, retried
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise ASRError(f"deepgram failed after {retries} attempts: {last!r}")


# ---------------------------------------------------------------------------
# The channel
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    profile: str = "noisy_phone"  # degradation profile (post-alias)
    asr_model: str = DEFAULT_ASR_MODEL
    audio_dir: Path = RESULTS_ROOT / "_audio"
    keep_audio: bool = True
    seed: int = 0  # run-level seed; per-turn degradation seeds derive from it
    voices: Optional[list[str]] = None  # archetype keys to rotate; None = all

    def __post_init__(self) -> None:
        self.profile = resolve_profile(self.profile)
        self.audio_dir = Path(self.audio_dir)


@dataclass
class TurnRecord:
    """Everything one user turn did on its way through the channel."""

    conversation: str
    turn_index: int
    original_text: str
    transcript: str
    profile: str
    archetype: str
    voice: str
    rate_wpm: int
    degrade_seed: int
    clean_wav: Optional[str]
    degraded_wav: Optional[str]
    asr_model: str
    asr_confidence: Optional[float]
    words: list = field(default_factory=list)
    wer: Optional[dict] = None
    achieved_snr_db: Optional[float] = None
    tts_s: float = 0.0
    degrade_s: float = 0.0
    asr_s: float = 0.0
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> dict:
        """Compact form embedded in the UserMessage's raw_data."""
        return {
            "original_text": self.original_text,
            "transcript": self.transcript,
            "profile": self.profile,
            "archetype": self.archetype,
            "voice": self.voice,
            "rate_wpm": self.rate_wpm,
            "asr_model": self.asr_model,
            "asr_confidence": self.asr_confidence,
            "words": self.words,
            "wer": None if self.wer is None else self.wer.get("wer"),
            "degraded_wav": self.degraded_wav,
        }


class SpeechChannel:
    """TTS -> degrade -> ASR for a whole run. One instance per run.

    Thread-safe by construction: :meth:`process` touches only per-call local
    state plus append-only filesystem writes with unique paths.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.roster = render.resolve_roster()
        keys = config.voices or sorted(self.roster)
        unknown = [k for k in keys if k not in self.roster]
        if unknown:
            raise KeyError(f"unknown archetypes {unknown}; have {sorted(self.roster)}")
        self.archetype_keys = keys

    # -- persona rotation --------------------------------------------------

    def archetype_for(self, conversation: str) -> str:
        """Deterministic per-conversation persona voice.

        Hash of (run seed, conversation key) -> archetype, so voices rotate
        across cases and trials but a re-run of the same (case, trial, seed)
        speaks with the same voice.
        """
        import hashlib

        digest = hashlib.sha256(
            f"{self.config.seed}:{conversation}".encode()
        ).hexdigest()
        return self.archetype_keys[int(digest[:8], 16) % len(self.archetype_keys)]

    # -- the pipe ----------------------------------------------------------

    def process(self, text: str, conversation: str, turn_index: int) -> TurnRecord:
        """Run one user turn through TTS -> degrade -> ASR."""
        cfg = self.config
        archetype = self.archetype_for(conversation)
        spec = self.roster[archetype]
        voice, rate = spec["voice"], spec["rate"]

        convo_dir = cfg.audio_dir / conversation
        convo_dir.mkdir(parents=True, exist_ok=True)
        stem = f"turn-{turn_index:03d}"
        clean_path = convo_dir / f"{stem}.clean.wav"
        deg_path = convo_dir / f"{stem}.{cfg.profile}.wav"

        t0 = time.monotonic()
        meta = render.render(text, clean_path, voice=voice, rate=rate, overwrite=True)
        t1 = time.monotonic()

        # Per-turn degradation seed: reproducible, distinct across turns.
        import hashlib

        degrade_seed = int(
            hashlib.sha256(
                f"{cfg.seed}:{conversation}:{turn_index}".encode()
            ).hexdigest()[:8],
            16,
        )
        result = degrade.degrade_file(str(clean_path), str(deg_path), cfg.profile, degrade_seed)
        t2 = time.monotonic()

        asr = deepgram_transcribe_words(deg_path, model=cfg.asr_model)
        t3 = time.monotonic()

        record = TurnRecord(
            conversation=conversation,
            turn_index=turn_index,
            original_text=text,
            transcript=asr["transcript"],
            profile=cfg.profile,
            archetype=archetype,
            voice=voice,
            rate_wpm=rate,
            degrade_seed=degrade_seed,
            clean_wav=str(clean_path),
            degraded_wav=str(deg_path),
            asr_model=asr["model"],
            asr_confidence=asr["confidence"],
            words=asr["words"],
            wer=wer(text, asr["transcript"]),
            achieved_snr_db=result.achieved_snr_db,
            tts_s=round(t1 - t0, 3),
            degrade_s=round(t2 - t1, 3),
            asr_s=round(t3 - t2, 3),
            duration_s=meta.get("duration_s", 0.0),
        )

        if not cfg.keep_audio:
            clean_path.unlink(missing_ok=True)
            deg_path.unlink(missing_ok=True)
            record.clean_wav = None
            record.degraded_wav = None
        return record

    def manifest(self) -> dict:
        """Reproducibility record for the run manifest."""
        return {
            "profile": self.config.profile,
            "profile_stages": degrade.PROFILE_DOC[self.config.profile],
            "asr_model": self.config.asr_model,
            "seed": self.config.seed,
            "keep_audio": self.config.keep_audio,
            "archetype_rotation": self.archetype_keys,
            "roster": self.roster,
            "tts": "macOS say, 16 kHz mono 16-bit PCM (rx_bench.audio.render)",
            "degrade": "rx_bench.audio.degrade (pure numpy; tau2 audio_effects needs scipy)",
        }


# ---------------------------------------------------------------------------
# Self-test: the LASA utterance, end to end through the channel
# ---------------------------------------------------------------------------


def selftest(profile: str, asr_model: str, out_dir: Path) -> int:
    """Synthesize -> degrade -> ASR one LASA utterance and show the damage."""
    load_env_file()
    text = "the doctor prescribed hydroxyzine 25 milligrams"
    cfg = PipelineConfig(profile=profile, asr_model=asr_model, audio_dir=out_dir, seed=0)
    channel = SpeechChannel(cfg)
    rec = channel.process(text, conversation="selftest", turn_index=0)

    outcome = classify_drug(rec.transcript, "hydroxyzine", "hydralazine")
    print(f"profile     {rec.profile}   asr {rec.asr_model}")
    print(f"voice       {rec.voice} ({rec.archetype}, {rec.rate_wpm} wpm)")
    print(f"original    {rec.original_text}")
    print(f"transcript  {rec.transcript}")
    print(f"wer         {rec.wer['wer']:.3f}  (S{rec.wer['sub']}/D{rec.wer['del']}/I{rec.wer['ins']})")
    print(f"confidence  {rec.asr_confidence}")
    low = [w for w in rec.words if (w.get('confidence') or 1.0) < 0.85]
    if low:
        print("low-conf    " + ", ".join(f"{w['word']}@{w['confidence']:.2f}" for w in low))
    print(f"drug check  {outcome['outcome']} (matched={outcome['matched']})")
    print(f"wavs        {rec.clean_wav}\n            {rec.degraded_wav}")
    print(f"timing      tts {rec.tts_s}s  degrade {rec.degrade_s}s  asr {rec.asr_s}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Speech channel: TTS -> degrade -> ASR")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--profile", default="noisy_phone",
                    help=f"{PROFILES} or 'realistic' (= noisy_phone)")
    ap.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT / "_selftest")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest(a.profile, a.asr_model, a.out))
    ap.print_help()
