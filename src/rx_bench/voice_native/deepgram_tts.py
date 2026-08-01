"""Deepgram Aura-2 TTS provider for the tau2 voice user simulator.

Env-gated alternative to ElevenLabs (the upstream default). Activated only
when ``TAU2_TTS_PROVIDER=deepgram`` (see
:mod:`rx_bench.voice_native.tau2_patches`); nothing changes when the var is
unset. Reuses the deepgram-sdk that the ``[voice]`` extra already installs for
transcription, so no new API keys are needed beyond ``DEEPGRAM_API_KEY``.

Caveats vs. ElevenLabs (disclose on any leaderboard row):

- Voices are Deepgram Aura-2 stock voices, not the tau2 Voice-Design personas.
  Accent identity is approximate at best (Aura-2 English voices are largely
  American), so realistic-tier accent diversity is NOT reproduced.
- ElevenLabs v3 audio tags (``[cough]``, ``[sneeze]``, ``[sniffle]``) are
  stripped; ``[pause]`` degrades to "..." (same as ElevenLabs non-v3 models).

This module is a straight port of the fork-side
``tau2/voice/utils/deepgram_tts_utils.py``; rx-bench depends on upstream tau2
at a pinned commit and cannot ship tau2 source edits, so the provider lives
here and is wired in by monkeypatch at ``import rx_bench`` time.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from loguru import logger

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.data_model.voice_personas import get_persona_name_by_voice_id

# Matches the user-sim synthesis rate (DEFAULT_PCM_SAMPLE_RATE).
AURA_SAMPLE_RATE = 16000
AURA_OUTPUT_AUDIO_FORMAT = AudioFormat(
    encoding=AudioEncoding.PCM_S16LE,
    sample_rate=AURA_SAMPLE_RATE,
)

# Persona -> Aura-2 voice. Only the two control (Clean) personas are carefully
# matched on age/gender/register; the regular-tier rows are best-effort
# stand-ins and do NOT reproduce the accents the τ-voice personas specify.
PERSONA_TO_AURA_VOICE = {
    "matt_delaney": "aura-2-orion-en",  # adult male, calm/approachable
    "lisa_brenner": "aura-2-thalia-en",  # adult female, clipped/confident
    "mildred_kaplan": "aura-2-athena-en",  # mature female
    "arjun_roy": "aura-2-apollo-en",
    "wei_lin": "aura-2-luna-en",
    "mamadou_diallo": "aura-2-zeus-en",
    "priya_patil": "aura-2-amalthea-en",
}
DEFAULT_AURA_VOICE = "aura-2-thalia-en"

_PAUSE_TAG_PATTERN = re.compile(r"\[pause\]", re.IGNORECASE)
_BRACKET_TAG_PATTERN = re.compile(r"\[[^\[\]]{1,40}\]")
_WS_PATTERN = re.compile(r"\s+")


def _clean_text_for_aura(text: str) -> str:
    """Strip ElevenLabs-style bracket tags; degrade [pause] to an ellipsis."""
    cleaned = _PAUSE_TAG_PATTERN.sub("...", text)
    if _BRACKET_TAG_PATTERN.search(cleaned):
        logger.debug("Deepgram Aura TTS: stripping ElevenLabs-style audio tags")
        cleaned = _BRACKET_TAG_PATTERN.sub(" ", cleaned)
    cleaned = _WS_PATTERN.sub(" ", cleaned).strip()
    if not cleaned:
        # e.g. the whole utterance was a vocal-tic tag; Aura rejects empty text.
        cleaned = "mm-hm"
    return cleaned


def resolve_aura_voice(elevenlabs_voice_id: Optional[str]) -> str:
    """Map the persona (via its ElevenLabs voice id) to an Aura-2 voice.

    TAU2_DEEPGRAM_TTS_MODEL forces a single Aura model for all personas.
    """
    forced = os.environ.get("TAU2_DEEPGRAM_TTS_MODEL")
    if forced:
        return forced
    persona_name = (
        get_persona_name_by_voice_id(elevenlabs_voice_id)
        if elevenlabs_voice_id
        else None
    )
    if persona_name and persona_name in PERSONA_TO_AURA_VOICE:
        return PERSONA_TO_AURA_VOICE[persona_name]
    logger.warning(
        f"Deepgram Aura TTS: no persona mapping for voice_id={elevenlabs_voice_id}; "
        f"falling back to {DEFAULT_AURA_VOICE}"
    )
    return DEFAULT_AURA_VOICE


def tts_deepgram_aura(
    text: str,
    config: ElevenLabsTTSConfig,
) -> AudioData:
    """Text to speech using the Deepgram Aura-2 REST API.

    Args:
        text: The text to synthesize.
        config: The (ElevenLabs-shaped) provider config; only ``voice_id`` is
            used, to recover the persona for Aura voice selection.

    Returns:
        AudioData in PCM_S16LE @ 16 kHz (same contract as tts_elevenlabs).
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY not found in environment")

    from deepgram import DeepgramClient

    model = resolve_aura_voice(getattr(config, "voice_id", None))
    cleaned_text = _clean_text_for_aura(text)

    text_preview = cleaned_text[:50] + "..." if len(cleaned_text) > 50 else cleaned_text
    logger.debug(
        f"Deepgram Aura TTS: calling API for text '{text_preview}' (model={model})"
    )

    client = DeepgramClient(api_key=api_key)
    try:
        audio_iter = client.speak.v1.audio.generate(
            text=cleaned_text,
            model=model,
            encoding="linear16",
            sample_rate=AURA_SAMPLE_RATE,
            container="none",
        )
        audio_bytes = b"".join(audio_iter)
    except Exception as e:
        logger.debug(
            f"Deepgram Aura TTS API call failed: {type(e).__name__}: {e} "
            f"(text='{text_preview}', model={model})"
        )
        raise

    if len(audio_bytes) == 0:
        raise ValueError(
            f"Deepgram Aura TTS returned empty audio for text: '{cleaned_text}'"
        )

    logger.debug(f"Deepgram Aura TTS: received {len(audio_bytes)} bytes of audio")
    return AudioData(
        data=audio_bytes,
        format=AURA_OUTPUT_AUDIO_FORMAT.model_copy(deep=True),
    )
