"""Tier-A audio corpus tooling: TTS rendering, channel degradation, corpus
manifest, and ASR scoring hooks.

The corpus WAVs themselves are distributed as a release asset (or rebuilt
locally with ``python -m rx_bench.audio.manifest build``); only the manifest
ships in git.
"""
