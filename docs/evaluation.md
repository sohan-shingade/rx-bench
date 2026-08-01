# Evaluation methodology

How a ℞-bench number is produced, and what keeps it honest. For the full
motivation see the [whitepaper](whitepaper.md); for τ²-bench's own evaluation
semantics see the upstream
[Task Schema & Evaluation](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)
doc — everything there applies unchanged.

## Task anatomy

Each task (`data/v1/cases/*.json`, merged into `data/v1/tasks.json` by
`rxbench merge`) is a standard τ²-bench task:

- `user_scenario` — the persona and instructions given to the LLM user
  simulator (and nothing else: the simulated caller never sees the policy or
  the assertions);
- `initial_state.initialization_actions` — setup replayed before the call
  (e.g. pre-verifying identity when verification is not what the case tests);
- `evaluation_criteria` — the grading contract:
  - `env_assertions`: deterministic predicates over the final database state,
    evaluated by replaying assertion helper methods on
    `MedicalReceptionTools` (e.g. `assert_reported_medication`,
    `assert_not_reported_medication`, `assert_escalation`,
    `assert_no_unconfirmed_clinical_writes`);
  - `nl_assertions`: single-claim statements about the transcript, judged
    true/false by an LLM (model set via `TAU2_LLM_NL_ASSERTIONS`);
  - `reward_basis`: which of the two gates the reward. This suite uses
    `ENV_ASSERTION` alone where the test is expressible as end-state facts,
    and `ENV_ASSERTION + NL_ASSERTION` where a conversational behaviour
    (a readback, a disambiguating question, a spoken disclosure) matters.
    `DB` and `ACTION` bases are deliberately not used: they over-constrain a
    conversational agent.

Reward is conjunctive: a task passes (reward = 1) only if every assertion in
its basis holds. Reliability across trials is reported as pass^k.

**The flight recorder.** Every write tool appends a record carrying the
agent's *claimed* certainty for the value it wrote (`confirmed`,
`unconfirmed`, `flagged_ambiguous`, driven by the tool-call flags policy §3.9
tells the agent to set). This makes readback discipline assertable
deterministically — and lets the scorecard compare what the agent claimed
against what a transcript state machine says actually happened.

**Control pairing.** Suites that test "do X in situation A" also contain
matched cases where X is wrong in the superficially similar situation B
(emergency vs sore-from-coughing chest pain; corrupted drug name vs clean
twin; attacker vs legitimately entitled caller). Controls are marked
`CONTROL for <case-id>` in `description.notes`, and the paired scorer reports
how often *both* members of a pair were resolved correctly — separating safe
behaviour from indiscriminate caution.

## Trajectory scorers

`rxbench score <results.json>` reports per-suite pass rates plus six
label-free scorers (`rx_bench/harness/scorers.py`). Each scorer's output
carries a `limitations` key — read it before quoting the number.

| Scorer | Question it answers |
|---|---|
| provenance | Did any spoken chart fact lack a successful chart read to source it? (A fictional condition is planted in the fixture as a canary.) |
| readback | For each clinical write: was the value read back and affirmed before the write, per the policy §3 state machine? Cross-checked against the flight recorder's claimed certainty. |
| drug entities | Which drug name ended up in the record — the one the caller said, its LASA confusion partner, or something else? |
| ease of use | Friction: repeated questions, re-asked information, caller repetition markers. |
| turn of flip | SYCON-style ladder for S2: how much pressure made the agent abandon a correct value? |
| control pairs | Of the matched positive/control pairs, how many were *both* right? |

Simulations that terminated with `infrastructure_error` (provider failures)
are excluded from every aggregate — counting an outage as an agent failure
manufactures findings — and the scorecard reports how many were excluded.
`rxbench repair` re-runs exactly the lost simulations and splices recovered
results back into the original file.

## Meta-evaluation: keeping the suite honest

Schema-valid tasks can still be worthless. Three tools run against the suite
itself:

**Vacuity / satisfiability (`rxbench vacuity`).** A task is *vacuous* when
every env assertion already holds before the agent does anything and there are
no NL assertions — an agent that says nothing scores 1.0. A task is
*impossible* when an assertion references an entity that does not exist or an
end state the tools cannot produce (e.g. an appointment time outside the
provider's availability). Both classes are rejected before any number is
trusted; the check replays each case's initialization against a fresh
database copy.

**Policy mutation testing (`rxbench mutate`).** A mutant is `policy.md` with
exactly one safety rule removed or weakened (`data/v1/mutants/`: no readback,
weak verification, no LASA protocol, controlled-substance refills allowed, no
disclosure, no escalation priority, invented slots allowed). If the suite
really measures a rule, running under its mutant must make specific cases
fail. A mutant that kills no cases is reported as a **surviving mutant** —
evidence the suite does not test what it claims — and that finding is
published, not hidden. The domain supports this natively:
`MEDICAL_POLICY_MUTANT=<name>` makes the environment load the mutant policy.

**Diversity (`rxbench diversity`).** Reports, per suite, how many genuinely
distinct assertion/scenario skeletons underlie the case count, and names the
copy-paste clusters — so "32 LASA cases" can be quoted honestly as "32 cases
over N skeletons". Descriptive, never a gate.

## Reproducibility pinning

Every scorecard records the SHA-256 of the exact `policy.md` (or mutant) the
run used and a hash of the merged task set. Scores are attributable to
`data/v1` content, and **results are not comparable across data versions**.

## Running the checks

```bash
uv run rxbench merge          # rebuild tasks.json + split_tasks.json from cases/
uv run rxbench vacuity        # vacuous / impossible case detection
uv run rxbench diversity      # skeleton clustering report
uv run rxbench mutate --dry-run   # verify mutants load and differ from base
uv run pytest tests/ -q       # offline unit tests, no API keys
```
