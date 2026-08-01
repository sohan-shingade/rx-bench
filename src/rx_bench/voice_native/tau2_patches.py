"""Runtime patches applied to the pinned upstream tau2 voice stack.

rx-bench depends on upstream tau2-bench at a fixed commit and cannot ship
edits to tau2 source, so two small fork-side changes needed for reproducible
Class B (audio-native) runs land here as monkeypatches instead. ``apply()``
is called from ``rx_bench.__init__`` — same pattern as the config patch there
— so anything that imports ``rx_bench`` before tau2 gets both patches.

1. **TTS provider gate** (``TAU2_TTS_PROVIDER=deepgram``) — reroutes the voice
   user simulator's caller TTS from ElevenLabs to Deepgram Aura-2
   (:mod:`rx_bench.voice_native.deepgram_tts`), so no ElevenLabs key or
   custom voice IDs are needed. Unset (or ``elevenlabs``), behavior is
   byte-identical to upstream. Diverges from upstream τ-voice caller audio —
   disclose on any leaderboard row.

2. **Turn-taker LLM override** (``TAU2_VOICE_USER_LLM``) — upstream hardcodes
   the voice user simulator's turn-taking/backchannel decision model to
   ``gpt-4.1`` (``VOICE_USER_SIMULATOR_DECISION_MODEL`` in ``tau2.config``),
   so a voice run against a non-OpenAI agent still dies on a missing
   OPENAI_API_KEY. Same pattern as ``TAU2_LLM_NL_ASSERTIONS``: env override,
   byte-identical default when unset. Overriding changes the turn-taking
   model away from upstream's gpt-4.1 — disclose on any leaderboard row.

Both patches are guarded so plain ``import rx_bench`` never crashes when
tau2's ``[voice]`` extra is not installed: the modules are patched only when
they import cleanly.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

#: Attribute set on patched tau2 modules so apply() is idempotent.
_PATCH_MARKER = "_RX_BENCH_PATCHED"


def _patch_voice_user_llm() -> None:
    """Make VOICE_USER_SIMULATOR_DECISION_MODEL overridable via TAU2_VOICE_USER_LLM.

    ``tau2.user.user_simulator_streaming`` binds the constant with a
    ``from tau2.config import ...`` at its own import time, so rebinding the
    attribute on ``tau2.config`` is only enough when that module has not been
    imported yet. We therefore also rebind the module-level name directly
    (importing the module if the voice deps allow it; if they don't, any
    later successful import will pick the patched config value up anyway).
    """
    import tau2.config as _cfg

    if not hasattr(_cfg, "VOICE_USER_SIMULATOR_DECISION_MODEL"):
        # Pinned upstream always has it; be safe against future bumps.
        logger.debug(
            "rx-bench: tau2.config has no VOICE_USER_SIMULATOR_DECISION_MODEL; "
            "skipping TAU2_VOICE_USER_LLM patch"
        )
        return

    model = os.environ.get(
        "TAU2_VOICE_USER_LLM", _cfg.VOICE_USER_SIMULATOR_DECISION_MODEL
    )
    _cfg.VOICE_USER_SIMULATOR_DECISION_MODEL = model

    uss = sys.modules.get("tau2.user.user_simulator_streaming")
    if uss is None:
        try:
            import tau2.user.user_simulator_streaming as uss  # noqa: F401
        except ImportError:
            # Voice deps unavailable; nothing has bound the old value.
            return
    uss.VOICE_USER_SIMULATOR_DECISION_MODEL = model


def _patch_tts_provider() -> None:
    """Env-gate the user-sim TTS provider behind TAU2_TTS_PROVIDER.

    Upstream ``synthesize_voice`` only supports ElevenLabs; it looks
    ``tts_elevenlabs`` up in its module globals at call time, so replacing
    that name with a dispatcher reaches every caller regardless of how or
    when they imported ``synthesize_voice`` (which is decorated and often
    imported ``from``-style, so it cannot itself be swapped reliably).
    """
    try:
        import tau2.voice.synthesis.synthesize as _synth
    except ImportError as exc:
        # tau2's [voice] extra (elevenlabs, ...) is not installed.
        logger.debug(f"rx-bench: skipping TAU2_TTS_PROVIDER patch ({exc})")
        return

    if getattr(_synth, _PATCH_MARKER, False):
        return

    from rx_bench.voice_native.deepgram_tts import tts_deepgram_aura

    _original_tts_elevenlabs = _synth.tts_elevenlabs

    def _tts_dispatch(text, config):
        """rx-bench dispatcher: TAU2_TTS_PROVIDER=deepgram -> Aura-2, else upstream."""
        tts_override = os.environ.get("TAU2_TTS_PROVIDER", "").strip().lower()
        if tts_override == "deepgram":
            return tts_deepgram_aura(text=text, config=config)
        if tts_override not in ("", "elevenlabs"):
            raise ValueError(f"Unsupported TAU2_TTS_PROVIDER: {tts_override}")
        return _original_tts_elevenlabs(text=text, config=config)

    _synth.tts_elevenlabs = _tts_dispatch
    setattr(_synth, _PATCH_MARKER, True)


def apply() -> None:
    """Apply both voice patches. Idempotent; never raises on missing voice deps."""
    _patch_voice_user_llm()
    _patch_tts_provider()
