"""TTS rendering for the Tier-A (audio) eval corpus.

Wraps macOS ``/usr/bin/say``. No cloud TTS, no API keys, no ffmpeg.

Output format
-------------
Every render is **16 kHz, mono, 16-bit signed PCM WAV**. ``say`` can emit that
directly via ``--data-format=LEI16@16000``, so there is no conversion step in
the happy path. 16 kHz mono is the standard input for telephony-adjacent ASR
(Deepgram, Whisper, Google all accept it natively) and it is above the 8 kHz
Nyquist limit of a real phone line, which matters because ``degrade.py`` does
its band-pass in the wide band and then lets the codec do the damage.

If ``--data-format`` ever fails on some macOS version, we fall back to
``say -o file.aiff`` followed by ``afconvert`` (built into macOS; ffmpeg is not
installed and is not assumed).

Voices
------
The roster is resolved **at runtime** from ``say -v '?'`` output. Nothing is
hardcoded that might not exist. Each persona archetype lists candidate voices in
preference order plus a speaking rate in words per minute; the first installed
candidate wins, and if none is installed the archetype falls back to the first
available en_* voice. Speaking rate is part of the archetype because rate is one
of the strongest real drivers of ASR error -- a 130 wpm elderly caller and a 240
wpm rushed caller are different acoustic problems even with identical text.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

SAY = "/usr/bin/say"
AFCONVERT = "/usr/bin/afconvert"

TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_WIDTH = 2  # bytes -> 16-bit


class RenderError(RuntimeError):
    pass


# ==========================================================================
# Voice enumeration
# ==========================================================================

_VOICE_LINE = re.compile(r"^(.+?)\s+([a-z]{2}(?:_[A-Z]{2})?)\s+#")

# macOS ships a pile of joke voices (Bells, Zarvox, Bubbles...). They are not
# speech, and including them would produce numbers that mean nothing.
NOVELTY_VOICES = {
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos",
    "Good News", "Jester", "Organ", "Superstar", "Trinoids", "Whisper",
    "Wobble", "Zarvox", "Junior", "Ralph", "Fred", "Kathy", "Princess",
    "Deranged", "Hysterical", "Pipe Organ",
}


@dataclass(frozen=True)
class Voice:
    name: str
    locale: str


def list_voices(english_only: bool = True, exclude_novelty: bool = True) -> list[Voice]:
    """Enumerate voices actually installed on this machine via ``say -v '?'``."""
    if not os.path.exists(SAY):
        return []
    out = subprocess.run([SAY, "-v", "?"], capture_output=True, text=True, timeout=30).stdout
    voices: list[Voice] = []
    for line in out.splitlines():
        m = _VOICE_LINE.match(line)
        if not m:
            continue
        name, locale = m.group(1).strip(), m.group(2)
        if english_only and not locale.startswith("en"):
            continue
        if exclude_novelty and name in NOVELTY_VOICES:
            continue
        voices.append(Voice(name, locale))
    return voices


# ==========================================================================
# Persona archetypes
# ==========================================================================

@dataclass(frozen=True)
class Archetype:
    key: str
    description: str
    candidates: tuple[str, ...]
    rate: int  # words per minute passed to `say -r`


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        "elderly_female", "Older woman, slow and deliberate, some vocal tremor in the synth model.",
        ("Grandma (English (US))", "Grandma", "Shelley (English (US))", "Samantha", "Kathy"), 130),
    Archetype(
        "elderly_male", "Older man, slow, lower pitch.",
        ("Grandpa (English (US))", "Grandpa", "Reed (English (US))", "Daniel", "Alex"), 125),
    Archetype(
        "brisk_professional", "Clear standard American, fast and clipped -- a caregiver or office manager.",
        ("Samantha", "Ava (Premium)", "Ava", "Allison", "Kathy"), 215),
    Archetype(
        "hurried_caller", "Rushed, running the words together. Near the top of say's usable rate.",
        ("Karen", "Sandy (English (US))", "Samantha", "Kathy"), 255),
    Archetype(
        "accented_indian", "Indian English. Accent variation, NOT non-native L2 speech -- see README.",
        ("Rishi", "Tara", "Aman", "Veena", "Sangeeta"), 175),
    Archetype(
        "accented_british", "British English, moderate pace.",
        ("Daniel", "Reed (English (UK))", "Serena", "Kate"), 165),
    Archetype(
        "accented_irish_sa", "Irish / South African English -- vowel-space variation.",
        ("Moira", "Tessa", "Fiona", "Karen"), 170),
    Archetype(
        "soft_low_energy", "Quiet, low-energy delivery; unwell or exhausted caller.",
        ("Shelley (English (US))", "Sandy (English (US))", "Flo (English (US))", "Samantha"), 145),
)

ARCHETYPES_BY_KEY = {a.key: a for a in ARCHETYPES}


def resolve_roster(voices: list[Voice] | None = None) -> dict[str, dict]:
    """Map each archetype -> the voice actually installed here, plus rate.

    Returns ``{archetype_key: {"voice": name, "rate": wpm, "locale": ..,
    "matched": bool, "description": ..}}``. ``matched`` is False when we had to
    fall back, so the manifest and README can be honest about substitutions.
    """
    voices = voices if voices is not None else list_voices()
    installed = {v.name: v for v in voices}
    if not installed:
        raise RenderError("no usable English voices found from `say -v '?'`")
    default = voices[0]

    roster: dict[str, dict] = {}
    for arch in ARCHETYPES:
        chosen, matched = None, False
        for cand in arch.candidates:
            if cand in installed:
                chosen, matched = installed[cand], True
                break
        if chosen is None:
            chosen = default
        roster[arch.key] = {
            "voice": chosen.name,
            "locale": chosen.locale,
            "rate": arch.rate,
            "matched": matched,
            "description": arch.description,
        }
    return roster


# ==========================================================================
# Rendering
# ==========================================================================

def text_hash(text: str, voice: str | None, rate: int | None) -> str:
    """Stable short hash over everything that determines the audio."""
    key = f"{text}\x00{voice or ''}\x00{rate or ''}\x00{TARGET_SR}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def slugify(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:maxlen] or "utt"


def default_out_path(out_dir: str | Path, slug: str, text: str,
                     voice: str | None, rate: int | None) -> Path:
    """Deterministic filename: <slug>__<voiceslug>_r<rate>__<hash>.wav"""
    vs = slugify(voice or "default", 24)
    return Path(out_dir) / f"{slugify(slug)}__{vs}_r{rate or 0}__{text_hash(text, voice, rate)}.wav"


def wav_info(path: str | Path) -> dict:
    with wave.open(str(path), "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sample_rate": w.getframerate(),
            "sample_width": w.getsampwidth(),
            "frames": w.getnframes(),
            "duration_s": w.getnframes() / float(w.getframerate()),
        }


def _say_to_wav(text: str, out_path: Path, voice: str | None, rate: int | None) -> None:
    cmd = [SAY, "--data-format=LEI16@16000", "-o", str(out_path)]
    if voice:
        cmd += ["-v", voice]
    if rate:
        cmd += ["-r", str(int(rate))]
    cmd += ["--", text]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not out_path.exists():
        raise RenderError(f"say failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")


def _say_via_aiff(text: str, out_path: Path, voice: str | None, rate: int | None) -> None:
    """Fallback: AIFF from say, then afconvert to 16 kHz mono 16-bit WAV."""
    if not os.path.exists(AFCONVERT):
        raise RenderError("say --data-format failed and afconvert is not available")
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "tmp.aiff"
        cmd = [SAY, "-o", str(aiff)]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", str(int(rate))]
        cmd += ["--", text]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if p.returncode != 0 or not aiff.exists():
            raise RenderError(f"say(aiff) failed: {p.stderr.strip()[:300]}")
        p2 = subprocess.run(
            [AFCONVERT, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(out_path)],
            capture_output=True, text=True, timeout=180)
        if p2.returncode != 0 or not out_path.exists():
            raise RenderError(f"afconvert failed: {p2.stderr.strip()[:300]}")


def render(text: str, out_path: str | Path, voice: str | None = None,
           rate: int | None = None, overwrite: bool = False) -> dict:
    """Render ``text`` to a 16 kHz mono 16-bit WAV at ``out_path``.

    Skip-if-exists: if the file is already there and is a valid WAV, nothing is
    re-synthesised (paths are content-addressed via ``default_out_path``, so an
    existing file with the right name has the right content). Pass
    ``overwrite=True`` to force.

    Returns a metadata dict: path, text, voice, rate, sample_rate, duration_s,
    frames, and ``skipped``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        try:
            info = wav_info(out_path)
            if info["frames"] > 0:
                return {"path": str(out_path), "text": text, "voice": voice,
                        "rate": rate, "skipped": True, **info}
        except Exception:
            out_path.unlink(missing_ok=True)  # corrupt, re-render

    tmp = out_path.with_suffix(".partial.wav")
    tmp.unlink(missing_ok=True)
    try:
        _say_to_wav(text, tmp, voice, rate)
    except RenderError:
        tmp.unlink(missing_ok=True)
        _say_via_aiff(text, tmp, voice, rate)

    info = wav_info(tmp)
    if (info["sample_rate"], info["channels"], info["sample_width"]) != (
            TARGET_SR, TARGET_CHANNELS, TARGET_WIDTH):
        conv = out_path.with_suffix(".conv.wav")
        conv.unlink(missing_ok=True)
        p = subprocess.run([AFCONVERT, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                            str(tmp), str(conv)], capture_output=True, text=True, timeout=180)
        if p.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RenderError(f"could not normalise to 16k mono: {p.stderr[:300]}")
        tmp.unlink(missing_ok=True)
        tmp = conv
        info = wav_info(tmp)

    if info["frames"] == 0:
        tmp.unlink(missing_ok=True)
        raise RenderError(f"empty render for voice={voice!r} text={text[:40]!r}")

    shutil.move(str(tmp), str(out_path))
    return {"path": str(out_path), "text": text, "voice": voice, "rate": rate,
            "skipped": False, **info}


def render_archetype(text: str, out_dir: str | Path, archetype: str, slug: str,
                     roster: dict | None = None, overwrite: bool = False) -> dict:
    """Render using a persona archetype resolved against installed voices."""
    roster = roster or resolve_roster()
    if archetype not in roster:
        raise KeyError(f"unknown archetype {archetype!r}; have {sorted(roster)}")
    spec = roster[archetype]
    path = default_out_path(out_dir, slug, text, spec["voice"], spec["rate"])
    meta = render(text, path, spec["voice"], spec["rate"], overwrite=overwrite)
    meta["archetype"] = archetype
    meta["locale"] = spec["locale"]
    return meta


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    ap = argparse.ArgumentParser(description="macOS `say` renderer -> 16 kHz mono WAV")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--roster", action="store_true", help="show resolved archetype->voice map")
    ap.add_argument("--text")
    ap.add_argument("--out")
    ap.add_argument("--voice")
    ap.add_argument("--rate", type=int)
    a = ap.parse_args()

    if a.list_voices:
        for v in list_voices():
            print(f"{v.name:<28} {v.locale}")
    elif a.roster:
        print(json.dumps(resolve_roster(), indent=2))
    elif a.text and a.out:
        print(json.dumps(render(a.text, a.out, a.voice, a.rate), indent=2))
    else:
        ap.print_help()
