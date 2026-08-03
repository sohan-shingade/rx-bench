#!/usr/bin/env bash
# One-command runner for the medical_reception suite.
#
#   ./scripts/run.sh smoke              # ~10 cases, fast sanity check
#   ./scripts/run.sh safety             # S1 S2 S3 S4 S6 S7
#   ./scripts/run.sh base 4             # everything, 4 trials (for pass^k)
#   ./scripts/run.sh lasa               # one suite
#
# Env overrides:
#   AGENT_LLM / USER_LLM     LiteLLM model ids (default gpt-4.1-2025-04-14)
#   TAU2_LLM_NL_ASSERTIONS   NL-assertion judge (default claude-gpt-5-6-luna)
#   CONCURRENCY              parallel simulations (default 2)
#   MAX_STEPS                turn cap per call (default 40; tau2 ships 200)
#   MAX_STEP_SECONDS         per-simulation wall clock cap (default 300)
#   SAVE_TO                  name the run dir so a killed run can be resumed
#   MEDICAL_POLICY_MUTANT    run against a weakened policy from data/v1/mutants/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

SPLIT="${1:-smoke}"
TRIALS="${2:-1}"

AGENT_LLM="${AGENT_LLM:-gpt-4.1-2025-04-14}"
USER_LLM="${USER_LLM:-gpt-4.1-2025-04-14}"

# Keep the judge independent of the agent seat. The pilot matrix pins this model.
export TAU2_LLM_NL_ASSERTIONS="${TAU2_LLM_NL_ASSERTIONS:-claude-gpt-5-6-luna}"

# Raise only if you have confirmed your provider/gateway holds up under
# parallel load; failed simulations are recorded as infrastructure_error and
# quietly shrink every denominator.
CONCURRENCY="${CONCURRENCY:-2}"

# tau2 defaults to 200 steps. A front-desk phone call that needs 200 turns has
# already failed the ease-of-use bar the E3 suite exists to measure.
MAX_STEPS="${MAX_STEPS:-40}"
MAX_STEP_SECONDS="${MAX_STEP_SECONDS:-300}"
SAVE_TO="${SAVE_TO:-}"

RXBENCH=(uv run --project "$ROOT" rxbench)
cd "$ROOT"

echo "==> merging case files into tasks.json"
"${RXBENCH[@]}" merge

echo
echo "==> checking for vacuous / unsatisfiable cases"
"${RXBENCH[@]}" vacuity

echo
echo "==> running split '$SPLIT' x$TRIALS trials"
[ -n "${MEDICAL_POLICY_MUTANT:-}" ] && echo "    (MUTANT POLICY: $MEDICAL_POLICY_MUTANT)"
RUN_ARGS=(
  --domain medical_reception
  --task-split-name "$SPLIT"
  --agent-llm "$AGENT_LLM"
  --user-llm "$USER_LLM"
  --num-trials "$TRIALS"
  --max-steps "$MAX_STEPS"
  --max-steps-seconds "$MAX_STEP_SECONDS"
  --max-concurrency "$CONCURRENCY"
  --auto-resume
)
[ -n "$SAVE_TO" ] && RUN_ARGS+=(--save-to "$SAVE_TO")
"${RXBENCH[@]}" run "${RUN_ARGS[@]}"

# tau2 saves each run as data/simulations/<run_name>/results.json.
LATEST="$(ls -td "$ROOT"/data/simulations/*/ 2>/dev/null | head -1 || true)"
LATEST="${LATEST:+${LATEST}results.json}"
if [ -n "$LATEST" ] && [ -f "$LATEST" ]; then
  echo
  echo "==> scorecard for $LATEST"
  "${RXBENCH[@]}" score "$LATEST" \
    --baseline "$ROOT/results/baseline_scorecard.json" \
    || echo "    (scorecard step failed — raw results are still at $LATEST)"
fi
