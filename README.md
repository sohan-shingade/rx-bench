# ℞-bench: A Benchmark for Medical AI Voice Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Whitepaper](https://img.shields.io/badge/📄_Whitepaper-draft_v0.2-B31B1B.svg)](docs/whitepaper.md)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-pilot-orange.svg)](https://sohan-shingade.github.io/rx-bench/)

> The name "℞-bench" (ASCII: `rx-bench`) is a provisional working name.

℞-bench evaluates **patient-facing medical AI voice agents**: systems that
talk to patients, use tools, and write to the medical record. Voice agents
are already answering clinic phone lines and intake desks today — scheduling
appointments, routing refills, recording reported medications, triaging
emergencies. The distinctive risk of the voice modality is that speech is
uncertain and medical vocabulary is adversarially confusable — and an agent
that mishears, guesses, and commits the guess to the chart has converted a
low-confidence acoustic token into a trusted clinical fact. We call this
failure mode **uncertainty laundering**, and ℞-bench exists to measure it.
(The benchmark's motivating application is the development of an emergency
department intake agent; the same failure modes govern every patient-facing
voice workflow.)

The design principle throughout: **score the chart, not the conversation.** A
transcript can read as fluent and careful while the committed record is wrong.
Every task is graded against the final environment state — which patient's
chart was touched, what medication name and dose were written, whether an
escalation was filed — with natural-language assertions on the dialogue as a
secondary, conjunctive check.

℞-bench is built as a family of domains on
[τ²-bench](https://github.com/sierra-research/tau2-bench) and inherits its
orchestrator, task schema, user simulator, and evaluators unmodified (the
τ³ voice/full-duplex stack included). The first domain, `medical_reception`,
ships **111 tasks across 8 suites**, a synthetic FHIR-mapped practice
database, a 19-tool API, a nine-section front-desk policy, six deterministic
safety scorers, and the meta-evaluation machinery (vacuity checks, policy
mutation testing, diversity reporting) that keeps the suite honest.

## Initial results (pilot, 2026-08-01)

Full rows and disclosures on the [leaderboard](https://sohan-shingade.github.io/rx-bench/); scorecards in `results/`. These are pilot-scale historical results (n ≤ 2), with no confidence intervals or statistical separation established. The August 2026 audit found and fixed scorer and evaluation-parity defects in the pre-audit harness that produced them; all rows will be re-run before being treated as comparable.

`claude-gpt-5-6-luna` served as the NL judge and user simulator throughout and is also a ranked agent below. Its Class A row is therefore self-graded; all other rows share a judge/simulator from the same model family, except the environment-only telephony score.

| Agent | Class | Split | Pass^1 |
|---|---|---|---|
| claude-gpt-5-6-sol | A (text) | full (111) | 0.658 |
| claude-gpt-5-6-luna | A (text; self-graded) | 110/111 (F-func-025 unscored) | 0.564 (pass^2 0.491) |
| claude-haiku-4-5 | A (text) | full (111) | 0.414 |
| Lacuna ED-intake pipeline (custom) | C (voice pipeline; external) | S1 LASA (31/32 scored) | 0.581 |
| claude-gpt-5-6-luna over telephony | C (voice pipeline; env-only, same model in agent/user seats) | S1 LASA (32) | 0.375 |

**Smoke / not ranked.** `gemini-3.1-flash-live` scored 0.5 on a two-case Class B smoke run. No backing scorecard is present in `results/`, so this is retained only as an unverified execution smoke result, not a benchmark comparison.

Two preliminary observations so far:

- **The telephony run needs an evaluation-parity re-run.** Its 0.375 score on
  32 sound-alike-drug tasks used environment-only grading, while the reported
  0.641 text reference included environment + NL assertions; they are not
  directly comparable, and the previously reported WER has no backing artifact
  in `results/`. The telephony scorecard does support zero wrong-drug charting:
  20 failures were missed records rather than substitutions.
- **Reviewer layers may guard the chart more than the conversation.** The external
  Lacuna scorecard reports 18/31 (0.581) on native environment + NL grading and
  29/30 supported cases under a post-hoc FHIR cross-grader. The 31/32-case run cannot be reproduced from this repository: its
  runnable bridge covers only 24 transcript-injection cases.

## Domains

| Domain | Status | Scope |
|---|---|---|
| `medical_reception` | **v1, shipping** | front-office phone workflows: scheduling, refills, messages, identity verification, escalation |
| `ed_intake` | roadmap | emergency-department / urgent-care intake: chief complaint capture, acuity-relevant history, red-flag escalation |
| `triage` | roadmap | symptom triage and disposition over the phone |
| `post_discharge` | roadmap | follow-up calls: medication reconciliation, warning-sign checks |

τ-bench launched with retail and airline and grew from there; ℞-bench follows
the same pattern — one rigorously validated domain first, with the task
schema, scorers, and voice stack designed to carry the roadmap domains.

## Task suites (`medical_reception`)

| Suite | Prefix | Tasks | What it measures |
|---|---|---:|---|
| LASA drug fidelity | `S1` | 32 | look-alike/sound-alike confusions, corrupted vs clean twins, noisy-transcript variants |
| Functional | `F` | 25 | happy paths and ordinary variations (booking, refills, messages) |
| Planted ASR error | `S2` | 14 | readback under pressure, sycophantic capitulation to a wrong "correction" |
| Emergency escalation | `S4` | 12 | red flags **and matched non-emergency controls** |
| Drug seeking | `S7` | 12 | controlled-substance routing, manipulation resistance |
| Adversarial | `S3` | 8 | social engineering, PHI extraction, identity attacks |
| Repetition / effort | `E3` | 5 | caller friction, redundant questions |
| AI disclosure | `S6` | 3 | disclosure ceremony (e.g. CA AB 3030-style rules) |
| **Total** | | **111** | |

Splits: `base` (all 111, default), `safety` (81), `functional` (30), `smoke`
(10), plus one split per suite (`lasa`, `planted_error`, `attacks`,
`emergency`, `disclosure`, `drug_seeking`, `repetition`). See
`data/v1/split_tasks.json`.

## Quick start

### 1. Install

```bash
git clone https://github.com/sohan-shingade/rx-bench
cd rx-bench
uv sync                 # installs rx_bench + tau2 (pinned) into .venv
# uv sync --extra voice # optional: local voice loop + audio tooling
```

### 2. Set up API keys

```bash
cp .env.example .env
# Fill in the provider keys your chosen models need (LiteLLM ids).
# Strongly recommended: set TAU2_LLM_NL_ASSERTIONS to a model on the SAME
# provider as your run, or grading fails after the simulation tokens are spent.
```

### 3. Run

```bash
# fast sanity check: 10 cases
uv run rxbench run --domain medical_reception --task-split-name smoke \
  --agent-llm gpt-4.1 --user-llm gpt-4.1 --max-steps 40 --max-concurrency 2

# score the run (per-suite pass rates + all six safety scorers)
uv run rxbench score data/simulations/<run_name>/results.json

# or the one-command wrapper (merge + vacuity-check + run + score):
./scripts/run.sh smoke
```

`rxbench` is a thin wrapper over the stock `tau2` CLI: it registers the
domain, applies the env-var config knobs, and forwards everything else — so
`rxbench run`, `rxbench view`, etc. behave exactly as documented for τ²-bench.

Suite maintenance commands: `rxbench merge` (rebuild `tasks.json` from
`data/v1/cases/`), `rxbench vacuity` (reject tasks a silent agent would pass),
`rxbench mutate` (policy mutation testing), `rxbench diversity` (copy-paste
cluster report), `rxbench repair` (recover simulations lost to infrastructure
errors). See [docs/evaluation.md](docs/evaluation.md).

## Agent classes

The same 111 cases, tools, policy, and scorers evaluate three classes of agent
— what differs is only how the caller's speech reaches the model:

| Class | Under test | How audio enters | Run |
|---|---|---|---|
| **A — text / standard scaffold** | `llm_agent` on clean text | none | `uv run rxbench run --domain medical_reception --task-split-name smoke --agent-llm <model> --user-llm <model>` |
| **B — voice / audio-native** | a speech-to-speech model (OpenAI Realtime, Gemini Live, xAI) | model consumes audio directly (τ-voice full-duplex) | `./src/rx_bench/voice_native/run_voice.sh clean --smoke` |
| **C — voice / pipeline** | ASR → any text LLM with tools → TTS | Deepgram hears a degraded TTS rendering of each caller turn | `uv run python -m rx_bench.voice_pipeline.run_pipeline --cases S1-lasa-001 --trials 1 --profile realistic` |

Class B needs the `[voice]` extra plus OpenAI/ElevenLabs/Deepgram keys (see
[src/rx_bench/voice_native/README.md](src/rx_bench/voice_native/README.md));
class C needs macOS `say` and a `DEEPGRAM_API_KEY` (see
[src/rx_bench/voice_pipeline/README.md](src/rx_bench/voice_pipeline/README.md)).
All three emit a standard tau2 `results.json` that `rxbench score` reads
unchanged. Scores are comparable across classes only when task sets, trial
counts, and evaluation modes are matched.

## Metrics

**Reward.** Each task's reward is computed by the τ²-bench evaluator from its
`reward_basis` — environment assertions over the final database state
(`ENV_ASSERTION`, deterministic, no judge) optionally conjoined with
LLM-judged natural-language assertions over the transcript (`NL_ASSERTION`).
A task passes when reward = 1. Tasks in this suite deliberately use only these
two bases; per-task detail is in each case file's `evaluation_criteria`.

**pass^k.** Reliability over k trials of the same task, as in τ-bench:

$$\text{pass}^k = \mathbb{E}_{\text{task}}\left[\binom{c}{k} \Big/ \binom{n}{k}\right]$$

where n is the number of trials run and c the number passed. pass^1 is the
average pass rate; pass^k for k>1 rewards *consistent* success and is the
number that matters for anything answering a phone in a clinic.

**Safety scorers.** Alongside reward, `rxbench score` reports six
deterministic scorers over the trajectory: provenance (chart-fact canary),
readback discipline (policy §3 state machine vs the agent's own claimed
certainty), drug-entity substitution (LASA vs generic ASR error), ease-of-use
(friction/repetition), turn-of-flip (pressure-induced concession ladder), and
paired positive/control outcomes. Each reports its own documented
limitations. See [docs/evaluation.md](docs/evaluation.md).

**Infrastructure honesty.** Simulations that died on provider/infrastructure
errors are excluded from every rate (never counted as failures), and the
scorecard says how many there were. `rxbench repair` re-runs exactly the lost
cases and splices them back.

## What's new / versioning

- **data/v1** — current data version: 111 tasks / 8 suites, including the new
  S7 drug-seeking suite. **Results are not comparable across data versions**;
  every scorecard records the policy file's SHA-256 and the task-set hash so a
  number is always attributable to the exact data that produced it.
- Run outputs land in `data/simulations/` (gitignored). Reference scorecards
  for the baseline runs live in `results/`.

## Repository layout

```
src/rx_bench/domain/    # tau2 domain: DB model, 19 tools, environment, Medplum mirror
src/rx_bench/harness/   # merge, vacuity, scorers, scorecard, mutate, diversity, repair
src/rx_bench/live/      # talk to the agent yourself (terminal, browser, voice)
src/rx_bench/audio/     # Tier-A audio corpus tooling (render/degrade/manifest/ASR eval)
src/rx_bench/voice_pipeline/  # agent class C: ASR -> text LLM -> (TTS) runner
src/rx_bench/voice_native/    # agent class B: tau-voice full-duplex wrapper

data/v1/                # db.json, policy.md, tasks.json, cases/, mutants/
docs/                   # evaluation methodology, submission guide, whitepaper
results/                # reference scorecards
tests/                  # offline unit tests (no API keys needed)
```

## Citation

If you use ℞-bench, please cite it, and please also cite the τ-bench line of
work it builds on:

```bibtex
@misc{rxbench2026,
  title={Rx-bench: A Benchmark for Medical AI Voice Agents},
  author={Sohan Shingade},
  year={2026},
  note={Working draft. Repository: https://github.com/sohan-shingade/rx-bench}
}

@misc{yao2024tau,
  title={$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
  author={Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan},
  year={2024}, eprint={2406.12045}, archivePrefix={arXiv}, primaryClass={cs.AI},
  url={https://arxiv.org/abs/2406.12045},
}

@misc{barres2025tau2,
  title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment},
  author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
  year={2025}, eprint={2506.07982}, archivePrefix={arXiv}, primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.07982},
}

@misc{ray2026tauvoice,
  title={$\tau$-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains},
  author={Soham Ray and Keshav Dhandhania and Victor Barres and Karthik Narasimhan},
  year={2026}, eprint={2603.13686}, archivePrefix={arXiv}, primaryClass={cs.SD},
  url={https://arxiv.org/abs/2603.13686},
}
```

## License and acknowledgments

MIT — see [LICENSE](LICENSE).

℞-bench is built on Sierra Research's
[τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT), whose
orchestrator, simulator, and evaluation stack it reuses wholesale. LASA drug
pairs are informed by the FDA's Tall Man lettering list (public domain). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). All patients, providers, and
clinical data in this benchmark are synthetic.
