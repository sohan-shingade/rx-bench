#!/usr/bin/env bash
# Voice-pipeline agent-under-test runner (ASR -> text LLM -> [TTS]).
#
#   ./src/rx_bench/voice_pipeline/run_pipeline.sh --cases S1-lasa-001 --trials 1 \
#       --profile realistic --run-name pipeline_dryrun
#
# Any litellm-routable text model works via --agent-llm. Results are standard
# tau2 results.json, scored by `rxbench score`. Keep --concurrency 1 (default).
#
# Requires the rx-bench environment (`uv sync` / `pip install -e .`) and a
# DEEPGRAM_API_KEY (exported or in the repo-root .env).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1 && [ -f "$ROOT/uv.lock" ]; then
    exec uv run python -m rx_bench.voice_pipeline.run_pipeline "$@"
fi
exec python -m rx_bench.voice_pipeline.run_pipeline "$@"
