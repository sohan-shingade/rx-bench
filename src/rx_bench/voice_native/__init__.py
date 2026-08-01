"""Class B (full-duplex audio-native) support for rx-bench.

This package holds the small pieces rx-bench adds on top of the pinned
upstream tau2-bench voice stack:

- :mod:`rx_bench.voice_native.deepgram_tts` — a Deepgram Aura-2 TTS provider
  for the voice user simulator (env-gated alternative to ElevenLabs).
- :mod:`rx_bench.voice_native.tau2_patches` — runtime monkeypatches applied to
  the pinned upstream tau2 at ``import rx_bench`` time (rx-bench cannot ship
  edits to tau2 source, so fork-side changes land here instead).

See ``README.md`` in this directory for how to run Class B evaluations.
"""
