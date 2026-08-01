#!/usr/bin/env bash
# Full-duplex voice runner for the medical_reception domain (τ-voice audio-native).
#
#   ./src/rx_bench/voice_native/run_voice.sh clean --smoke              # 2 tasks, control, 1 trial
#   ./src/rx_bench/voice_native/run_voice.sh realistic --smoke
#   ./src/rx_bench/voice_native/run_voice.sh clean --task-ids S7-seek-001 --trials 1
#   ./src/rx_bench/voice_native/run_voice.sh realistic                  # full base split
#
# Modes:
#   clean      -> --speech-complexity control  (no noise, US accents, patient user)
#   realistic  -> --speech-complexity regular  (noise, bursts, accents, interruptions)
#
# Flags:
#   --smoke                small fixed subset (S1-lasa-001 S7-seek-001)
#   --task-ids ID [ID..]   explicit task ids (until next --flag)
#   --trials N             trials per task (default 1)
#   --save-to NAME         run dir name under data/simulations/
#   --dry-run              print the tau2 command without executing
#
# Env overrides:
#   PROVIDER      openai | gemini | xai | livekit   (default openai)
#   MODEL         --audio-native-model override
#   MAX_SECONDS   --max-steps-seconds (default 600; tau2 default is 1200)
#   CONCURRENCY   --max-concurrency (default 1; realtime APIs rate-limit hard)
#   SEED          --seed (default 42)
#   TAU2_ALLOW_DEFAULT_VOICES=1   skip the TAU2_VOICE_ID_* check (Sierra-internal
#                                 default voice IDs only work on Sierra accounts)
#
# This script never prints secret values — only variable NAMES.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

# Run inside the rx-bench environment: `python -m rx_bench.cli run` registers
# the medical_reception domain and points TAU2_DATA_DIR at $ROOT/data before
# forwarding to the stock `tau2 run` CLI.
cd "$ROOT"
if command -v uv >/dev/null 2>&1 && [ -f "$ROOT/uv.lock" ]; then
  PYRUN=(uv run python)
else
  PYRUN=(python)
fi
DATA_DIR="${TAU2_DATA_DIR:-$ROOT/data}"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
MODE="${1:-}"
case "$MODE" in
  clean)     COMPLEXITY="control" ;;
  realistic) COMPLEXITY="regular" ;;
  *) echo "usage: $0 {clean|realistic} [--smoke] [--task-ids ID..] [--trials N] [--save-to NAME] [--dry-run]" >&2
     exit 2 ;;
esac
shift

TRIALS=1
SAVE_TO=""
DRY_RUN=0
TASK_IDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)    TASK_IDS=(S1-lasa-001 S7-seek-001); shift ;;
    --task-ids) shift
                while [ $# -gt 0 ] && [[ "$1" != --* ]]; do TASK_IDS+=("$1"); shift; done ;;
    --trials)   TRIALS="$2"; shift 2 ;;
    --save-to)  SAVE_TO="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-}"
MAX_SECONDS="${MAX_SECONDS:-600}"
CONCURRENCY="${CONCURRENCY:-1}"
SEED="${SEED:-42}"
if [ -z "$SAVE_TO" ]; then
  SAVE_TO="voice_${COMPLEXITY}_${PROVIDER}_$(date +%Y%m%d_%H%M%S)"
fi

# ---------------------------------------------------------------------------
# Load keys from the repo-root .env (values are never echoed)
# ---------------------------------------------------------------------------
# Parse KEY=VALUE lines, strip carriage returns and surrounding quotes, export.
# (Tolerates CRLF files, unlike plain `source`.) Values are never echoed.
load_env_file() {
  local f="$1" line key val
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    key="${line%%=*}"
    val="${line#*=}"
    # strip matching surrounding quotes
    if [[ "$val" == \"*\" && "$val" == *\" ]]; then val="${val:1:${#val}-2}"; fi
    if [[ "$val" == \'*\' && "$val" == *\' ]]; then val="${val:1:${#val}-2}"; fi
    # do not clobber values already set in the environment
    if [ -z "${!key:-}" ]; then export "$key=$val"; fi
  done < "$f"
}
load_env_file "$ROOT/.env"

# ---------------------------------------------------------------------------
# Preflight: python deps
# ---------------------------------------------------------------------------
fail=0
if ! "${PYRUN[@]}" -c "import elevenlabs, deepgram, websockets" >/dev/null 2>&1; then
  echo "ERROR: voice dependencies are not installed in the rx-bench environment." >&2
  echo "  Fix:" >&2
  echo "    cd $ROOT && uv sync --extra voice   # or: pip install -e '.[voice]'" >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# Preflight: API keys (names only, never values)
# ---------------------------------------------------------------------------
missing=()
# Always required in voice mode:
#  - OPENAI_API_KEY: user-sim turn-taking decision model is hard-coded to gpt-4.1
#    (src/tau2/config.py VOICE_USER_SIMULATOR_DECISION_MODEL) and the default
#    user LLM is gpt-4.1-2025-04-14.
#  - ELEVENLABS_API_KEY: user-sim TTS (only synthesis provider).
#  - DEEPGRAM_API_KEY: transcription for evaluation (default nova-3).
[ -n "${OPENAI_API_KEY:-}" ]     || missing+=(OPENAI_API_KEY)
[ -n "${ELEVENLABS_API_KEY:-}" ] || missing+=(ELEVENLABS_API_KEY)
[ -n "${DEEPGRAM_API_KEY:-}" ]   || missing+=(DEEPGRAM_API_KEY)

case "$PROVIDER" in
  openai)  ;; # covered by OPENAI_API_KEY above
  gemini)  [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] \
             || [ -n "${GOOGLE_SERVICE_ACCOUNT_KEY:-}" ] || missing+=("GEMINI_API_KEY (or GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_SERVICE_ACCOUNT_KEY)") ;;
  xai)     [ -n "${XAI_API_KEY:-}" ] || missing+=(XAI_API_KEY) ;;
  livekit) [ -n "${ANTHROPIC_API_KEY:-}" ] || missing+=("ANTHROPIC_API_KEY (livekit cascaded LLM)") ;;
  *) echo "ERROR: unknown PROVIDER '$PROVIDER' (openai|gemini|xai|livekit)" >&2; exit 2 ;;
esac

if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: missing required API keys for voice mode (provider=$PROVIDER):" >&2
  for k in "${missing[@]}"; do echo "  - $k" >&2; done
  echo "Add them to $ROOT/.env (see src/rx_bench/voice_native/README.md). Refusing to run." >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# Preflight: ElevenLabs voice IDs
# The voice IDs baked into src/tau2/data_model/voice_personas.py are
# Sierra-internal and do not work on external ElevenLabs accounts.
# ---------------------------------------------------------------------------
if [ "$COMPLEXITY" = "control" ]; then
  personas=(MATT_DELANEY LISA_BRENNER)
else
  personas=(MILDRED_KAPLAN ARJUN_ROY WEI_LIN MAMADOU_DIALLO PRIYA_PATIL)
fi
missing_voices=()
for p in "${personas[@]}"; do
  v="TAU2_VOICE_ID_${p}"
  [ -n "${!v:-}" ] || missing_voices+=("$v")
done
if [ ${#missing_voices[@]} -gt 0 ] && [ "${TAU2_ALLOW_DEFAULT_VOICES:-0}" != "1" ]; then
  echo "ERROR: no ElevenLabs voice-ID overrides set for complexity '$COMPLEXITY':" >&2
  for v in "${missing_voices[@]}"; do echo "  - $v" >&2; done
  echo "The built-in defaults are Sierra-internal and will be rejected by ElevenLabs." >&2
  echo "Create voices (see docs/voice-personas.md in sierra-research/tau2-bench)" >&2
  echo "and export the IDs, or set TAU2_ALLOW_DEFAULT_VOICES=1 to try anyway. Refusing to run." >&2
  fail=1
fi

# ---------------------------------------------------------------------------
# Preflight: background-noise assets (needed for realistic mode)
# ---------------------------------------------------------------------------
NOISE_DIR="$DATA_DIR/voice/background_noise_audio_pcm_mono_verified"
if [ "$COMPLEXITY" = "regular" ] && [ ! -f "$NOISE_DIR/continuous/people_talking.wav" ]; then
  echo "ERROR: background-noise assets missing at $NOISE_DIR" >&2
  echo "  Expected continuous/*.wav and bursts/*.wav (see src/tau2/voice_config.py;" >&2
  echo "  copy them from the sierra-research/tau2-bench data/voice/ assets)." >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry-run: preflight FAILED — showing the command that would run)" >&2
  else
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cmd=("${PYRUN[@]}" -m rx_bench.cli run
  --domain medical_reception
  --audio-native
  --audio-native-provider "$PROVIDER"
  --speech-complexity "$COMPLEXITY"
  --num-trials "$TRIALS"
  --max-concurrency "$CONCURRENCY"
  --max-steps-seconds "$MAX_SECONDS"
  --seed "$SEED"
  --save-to "$SAVE_TO"
  --auto-resume
  --verbose-logs)
[ -n "$MODEL" ] && cmd+=(--audio-native-model "$MODEL")
[ ${#TASK_IDS[@]} -gt 0 ] && cmd+=(--task-ids "${TASK_IDS[@]}")

echo "mode=$MODE (speech-complexity=$COMPLEXITY) provider=$PROVIDER trials=$TRIALS"
echo "results -> $DATA_DIR/simulations/$SAVE_TO/"
printf '%q ' "${cmd[@]}"; echo
if [ "$DRY_RUN" -eq 1 ]; then
  echo "(dry-run: not executing)"
  exit 0
fi
exec "${cmd[@]}"
