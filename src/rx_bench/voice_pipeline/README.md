# Voice-pipeline agent class (ASR → text LLM → [TTS])

The third agent class in the taxonomy:

| class | what is under test | how audio enters |
|---|---|---|
| **text-standard-scaffold** | `llm_agent` on clean text | none (Tier T; `rxbench run`) |
| **voice-audio-native** | a speech-to-speech model | model consumes audio directly (`src/rx_bench/voice_native/`) |
| **voice-pipeline** (this dir) | ASR → **any text LLM with tools** → TTS | Deepgram hears a degraded TTS rendering of every caller turn |

The point of this class: it lets **any text model** be evaluated on the voice
tier. The model under test is exactly the stock tau2 `llm_agent` (same tools,
`policy.md`, environment, evaluator as every text run) — what changes is what
its "ears" receive. That makes scores directly comparable across the three
classes: same cases, same scorers, different ear.

## Architecture (v1, turn-based — Tier A of the two-tier voice design)

```
user-sim LLM ──text──► macOS `say` (persona voice, per-conversation rotation)
                            │ 16 kHz mono WAV        (rx_bench.audio.render)
                            ▼
                    degradation profile              (rx_bench.audio.degrade)
                    clean | phone | noisy_phone | bad_line   ("realistic" = noisy_phone)
                            │ 300–3400 Hz band-pass, G.711 µ-law, babble, packet loss…
                            ▼
                    Deepgram ASR (default nova-3-medical, word confidences kept)
                            │ TRANSCRIPT, not the original text
                            ▼
                    stock tau2 orchestrator ──► llm_agent (agent under test)
                            │ agent reply passes through as text (see convention)
                            ▼
                    user-sim LLM (sees clean conversation from its own side)
```

Implementation: `PipelineUser` subclasses tau2's `UserSimulator` and wraps the
USER→AGENT channel, following the proven pattern of `rx_bench.live.talk` +
`rx_bench.live.human_user` (substitute the producer of user turns via
`registry.register_user`; touch nothing else). One tau2 `results.json` comes
out the other end and `rxbench score` and all scorers read it unchanged.

## The transcript-injection convention

τ-voice itself evaluates pipelines by injecting the transcript on the user
side, and this class follows that convention:

* **User → agent**: the message that enters the shared conversation (agent
  context, `results.json`, every scorer) carries the **ASR transcript** of the
  degraded audio. If ASR yields an empty string the agent receives
  `[inaudible]` (documented sentinel; a real IVR turn would surface the same).
* **User's own memory**: the simulator's private state keeps the **original
  text** — a caller knows what they said, not what the phone line did to it.
  Without this the user-sim would "remember" the garbled version of its own
  speech and drift out of character.
* **Control tokens** (`###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###`)
  and tool-call turns bypass the channel: they are orchestration signals, not
  speech.
* **Agent → user**: passes through as text. The agent's TTS leg exists in the
  architecture but is not audibly rendered in v1 (see Limitations) — which is
  convention-compliant, since the user-sim of a pipeline eval reads text either
  way.

## Running

```bash
python -m rx_bench.voice_pipeline.run_pipeline --cases S1-lasa-001,S1-lasa-002 \
    --trials 1 --profile realistic --agent-llm gpt-4.1-2025-04-14 \
    --run-name my_run
```

(or `./src/rx_bench/voice_pipeline/run_pipeline.sh`, which wraps the same
module through `uv run`). Requires `DEEPGRAM_API_KEY` — exported or in the
repo-root `.env` — plus macOS `say` for TTS.

Flags: `--cases` (comma-separated ids; `--list-cases` to browse), `--trials`,
`--profile` (`clean|phone|noisy_phone|bad_line|realistic`), `--agent-llm`
(**any** litellm-routable text model), `--user-llm`, `--asr-model`
(`nova-3-medical` default; `nova-2`, `nova-3` etc. accepted), `--voices`
(archetype subset), `--seed`, `--no-keep-audio`, `--evaluate`
(`env` default; `all_with_nl_assertions` adds the judge). Keep
`--concurrency 1` (default): `say` is serial.

Output under `data/simulations/voice_pipeline/<run-name>/` (gitignored):

* `results.json` — standard tau2 Results; feed it to `rxbench score`, diff it
  against text-tier runs, everything just works (run_pipeline invokes the
  scorecard automatically).
* `manifest.json` — reproducibility record: model ids, ASR model, profile and
  its exact stage list, voice roster actually resolved on this machine,
  seeds, git rev.
* `pipeline_turns.jsonl` — per turn: original text, clean + degraded WAV
  paths, ASR transcript, word confidences, WER vs. the original, achieved SNR,
  stage timings. This is the L1 attribution feed: "did the correct entity
  survive the ear?" is answerable per turn without re-running anything.
* `audio/<case>-s<seed>/turn-NNN.{clean,<profile>}.wav` — the actual audio
  (deleted if `--no-keep-audio`).

Each pipelined `UserMessage` in `results.json` also carries a compact copy in
`raw_data.voice_pipeline` (original text, transcript, WER, voice, confidences),
so a results file is self-describing even without the sidecar log.

### Unit self-test (no tau2, no LLM)

```bash
python -m rx_bench.voice_pipeline.pipeline --selftest --profile noisy_phone
```

Synthesizes "the doctor prescribed hydroxyzine 25 milligrams", degrades it,
transcribes it, and classifies the drug outcome against the LASA twin
(hydralazine) — the LASA laundering risk exercised in one command.

## Reproducibility

* Voice per conversation: `sha256(run_seed, case_id, trial_seed)` → archetype
  from `rx_bench.audio.render`'s roster (resolved at runtime against installed
  voices; recorded in the manifest, including fallbacks).
* Degradation: seeded per `(run_seed, conversation, turn)` — bit-reproducible.
* TTS: deterministic given (voice, rate, text).
* Non-deterministic in principle: the LLMs (temperature pinned at 0, seed
  recorded) and Deepgram (a network service; model id recorded, response
  logged verbatim). A re-run is auditable turn-for-turn even where it cannot
  be bit-identical.

## What is reused vs. new

Reused as libraries (all read-only):
* `rx_bench.audio.render` — `say` TTS, persona archetypes, voice roster.
* `rx_bench.audio.degrade` — the four channel profiles (pure numpy G.711 chain).
* `rx_bench.audio.asr_eval` — WER, LASA drug-outcome classifier; the Deepgram
  request shape (re-implemented here only to add word confidences).
* tau2: `UserSimulator`, orchestrator, `llm_agent`, registry, `run_tasks`
  batch runner (checkpointing/resume for free), evaluator.
* the harness scorecard (`rxbench score`) — consumed as-is on the output.

Considered and not used: `tau2.voice.synthesis.audio_effects` (needs scipy,
which is not a dependency of this package — verified) and
`tau2.voice.transcription.transcribe` (works, but drags in AudioData
plumbing and a 24 kHz resample tuned for the OpenAI realtime path; the
project's own Deepgram path is what the Tier-A corpus numbers used).

New here: `pipeline.py` (the TTS→degrade→ASR seam + per-turn records),
`pipeline_user.py` (the tau2 user wrapper), `run_pipeline.py` / `.sh` (CLI).

## Limitations (v1)

* **Turn-based, not full-duplex.** This is Tier A of the two-tier design, not
  τ-voice's streaming tier: no barge-in, no backchannels, no latency-induced
  overlap. Timing metrics from these runs measure the pipeline stages, not
  conversational timing under interruption.
* **Agent TTS is not audibly rendered.** The agent's replies reach the
  user-sim as text. Consequences: (a) E-suite responsiveness numbers exclude
  the agent-TTS leg; (b) failure modes where the *agent's* speech is
  misheard by the *caller* are out of scope in v1. Rendering the agent leg
  (say/ElevenLabs) is a straight extension of `SpeechChannel`.
* **`say` is accent variation, not L2 speech** — same honesty note as the
  audio corpus docs (`src/rx_bench/audio/`); equity findings from these
  voices are about synthetic accent proxies.
* **No paralinguistics.** `say` cannot do distress; distress cases remain the
  audio-native class's job.
* **User-sim wording is model-generated**, so exact utterances vary run to
  run even at temperature 0 across model versions; the S1 corpus's fixed
  carrier sentences live in `src/rx_bench/audio/`, and this class complements
  — not replaces — that fixed-utterance ASR measurement.
