# medical_reception × τ-voice (full-duplex audio-native)

How to run the `medical_reception` domain through tau2-bench's full-duplex voice
loop — a real-time audio-native agent (OpenAI Realtime / Gemini Live / xAI Grok
Voice) talking to a voice user simulator (ElevenLabs TTS + audio effects), with
Deepgram transcription for evaluation.

Everything here is additive: the standard tau2 voice machinery drives the run,
and the same env/DB/action/NL-assertion evaluator scores it. This is the
**voice-audio-native** agent class (Class B) — see the taxonomy in
`src/rx_bench/voice_pipeline/README.md`.

---

## TL;DR — the commands

`medical_reception` is registered when `rx_bench` is imported, so voice mode is
just the standard `run` invocation with `--audio-native`:

```bash
# Clean (control): no noise, American accents, patient user
rxbench run \
  --domain medical_reception \
  --audio-native \
  --audio-native-provider openai \
  --speech-complexity control \
  --task-ids S7-seek-001 \
  --num-trials 1 \
  --max-concurrency 1 \
  --save-to voice_dryrun \
  --verbose-logs

# Realistic (regular): background noise, bursts, accents, interruptions, frame drops
rxbench run \
  --domain medical_reception \
  --audio-native \
  --audio-native-provider openai \
  --speech-complexity regular \
  --num-trials 1 \
  --max-concurrency 1 \
  --save-to voice_regular_base \
  --verbose-logs
```

Or use the wrapper in this directory (does key/dep/asset preflight for you):

```bash
./src/rx_bench/voice_native/run_voice.sh clean --smoke          # 2 tasks, control, 1 trial
./src/rx_bench/voice_native/run_voice.sh realistic --smoke
./src/rx_bench/voice_native/run_voice.sh clean --task-ids S1-lasa-001 --trials 1
./src/rx_bench/voice_native/run_voice.sh realistic             # full base split
PROVIDER=gemini ./src/rx_bench/voice_native/run_voice.sh clean --smoke
```

Results land in `data/simulations/<save_to>/` (gitignored; directory format:
`results.json` + `simulations/sim_*.json` + `artifacts/task_*/sim_*/audio/both.wav`,
Audacity label tracks, and `llm_debug/` when `--verbose-logs` is on). Browse with
`rxbench view`.

Useful task IDs for single-task smoke runs (from `data/v1/tasks.json`, 111
tasks total): `S1-lasa-001`, `S7-seek-001`, `E3-rep-001`, `F-func-001`.

---

## How voice mode is wired (what the flags do)

Flag plumbing lives in tau2-bench's `src/tau2/cli.py`. `--audio-native`
switches `run_domain()` from `TextRunConfig` to `VoiceRunConfig` with an
`AudioNativeConfig`:

| Flag | Default | Meaning |
|---|---|---|
| `--audio-native` | off | Enable full-duplex voice (DiscreteTimeAudioNativeAgent + VoiceStreamingUserSimulator) |
| `--audio-native-provider` | `openai` | `openai` \| `gemini` \| `xai` \| `livekit` (cascaded STT→LLM→TTS pipeline) |
| `--audio-native-model` | per-provider | openai: `gpt-realtime-1.5`, gemini: `gemini-3.1-flash-live-preview`, xai: `xai-realtime` |
| `--speech-complexity` | `regular` | `control` (Clean) or `regular` (Realistic); ablations: `control_audio`, `control_accents`, `control_behavior`, and pairwise combos |
| `--cascaded-config` | — | livekit provider only: `default`, `openai-thinking` |
| `--tick-duration` | 0.2 s | Simulation timestep |
| `--max-steps-seconds` | 1200 s | Max conversation duration (cap this for smoke runs) |
| `--verbose-logs` | off | Save `both.wav`, per-tick data, LLM logs |
| `--task-ids A B` / `--num-tasks N` | all | Subset selection |
| `--save-to NAME` | timestamped | Run dir under `data/simulations/` |

There is no separate "voice harness" — reward evaluation is the same
env/DB/action/NL-assertion machinery as the text tier, driven from the
transcribed conversation.

### Clean vs Realistic (per-task user-sim voice config)

Presets are defined in tau2-bench's `src/tau2/user_simulation_voice_presets.py`:

- **`control` (Clean)** — personas `matt_delaney` / `lisa_brenner` (American
  accents), no background noise, no bursts, no frame drops, no vocal tics, no
  muffling. Telephony conversion (G.711 μ-law 8 kHz) is always applied.
- **`regular` (Realistic)** — personas `mildred_kaplan`, `arjun_roy`, `wei_lin`,
  `mamadou_diallo`, `priya_patil` (accents/age variation), plus an environment
  sampled per task: `indoor` (people talking / TV news + ringing phone / dog
  bark bursts) or `outdoor` (busy street / street & metro + car horn / engine /
  siren), SNR ~15 dB with drift, ~1 burst/min, 1% Gilbert–Elliott frame drops,
  out-of-turn speech, vocal tics, dynamic muffling (constants in
  tau2-bench's `src/tau2/voice_config.py`).

Per-task configs are sampled **deterministically on the fly** from
`seed + hash(task_id)` — no domain file is required. Other domains ship a
pre-sampled `tasks_voice.json`; `medical_reception` does not, so each run logs
a "sampling on the fly" warning, which is harmless and reproducible for a
fixed `--seed`. To pin configs permanently you can run
`python -m tau2.user_simulation_voice_presets medical_reception` (writes a
`tasks_voice.json` into the domain data directory).

The user-sim persona/behavior (verbosity, interrupt tendency) comes with the
complexity preset; `--user-persona` JSON can override behavior knobs.

---

## What each provider/key is used for

| Env var | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI Realtime agent; **also always required**: the voice user simulator's turn-taking decision model is hard-coded to `gpt-4.1` (`VOICE_USER_SIMULATOR_DECISION_MODEL`, tau2 `src/tau2/config.py`; used in `src/tau2/user/user_simulator_streaming.py`), and the default user-sim LLM is `gpt-4.1-2025-04-14` |
| `ELEVENLABS_API_KEY` | User simulator TTS (the only synthesis provider — tau2 `src/tau2/voice/synthesis/synthesize.py`) — required for **every** provider, every complexity |
| `DEEPGRAM_API_KEY` | Transcription for evaluation (default `nova-3`); also livekit cascaded STT |
| `GEMINI_API_KEY` (AI Studio) or `GOOGLE_SERVICE_ACCOUNT_KEY`/`GOOGLE_APPLICATION_CREDENTIALS` (Vertex) | Gemini Live agent (tau2's Gemini provider reads `GEMINI_API_KEY`, not `GOOGLE_API_KEY`) |
| `XAI_API_KEY` | xAI Grok Voice agent |
| `ANTHROPIC_API_KEY` | Only used in voice mode if you point the livekit cascaded LLM or the NL-assertion judge (`TAU2_LLM_NL_ASSERTIONS=anthropic/...`) at Claude |
| `TAU2_VOICE_ID_<PERSONA>` | ElevenLabs voice-ID overrides. The IDs baked into tau2's `src/tau2/data_model/voice_personas.py` are Sierra-internal and **will not work** on an external ElevenLabs account |

Put keys in the repo-root `.env` (see `.env.example`; `run_voice.sh` loads it
without echoing values) or export them in your shell.

## Setup checklist

1. **Voice dependencies** — the `[voice]` extra (`elevenlabs`, `deepgram-sdk`,
   `websockets`, `google-genai`, `livekit-agents`, …):
   ```bash
   uv sync --extra voice        # or: pip install -e '.[voice]'
   ```
2. **Keys** — add to the repo-root `.env` (names in `.env.example`):
   - `OPENAI_API_KEY` — required for any voice run (user-sim decision model),
     and it is the agent key for the default `openai` provider.
   - `ELEVENLABS_API_KEY` — required for any voice run (user-sim TTS).
   - `DEEPGRAM_API_KEY` — transcription for evaluation.
   - Optional extra providers: `GEMINI_API_KEY` (Gemini Live), `XAI_API_KEY`
     (Grok Voice).
3. **ElevenLabs voices** — create voices in your ElevenLabs account (guide:
   `docs/voice-personas.md` in sierra-research/tau2-bench) and export their IDs:
   - Clean minimum: `TAU2_VOICE_ID_MATT_DELANEY`, `TAU2_VOICE_ID_LISA_BRENNER`
   - Realistic additionally: `TAU2_VOICE_ID_MILDRED_KAPLAN`,
     `TAU2_VOICE_ID_ARJUN_ROY`, `TAU2_VOICE_ID_WEI_LIN`,
     `TAU2_VOICE_ID_MAMADOU_DIALLO`, `TAU2_VOICE_ID_PRIYA_PATIL`
   - Without overrides the run may start but ElevenLabs will reject the
     Sierra-internal voice IDs at synthesis time (`run_voice.sh` refuses to
     run unless `TAU2_ALLOW_DEFAULT_VOICES=1`).
4. **Background-noise assets** (realistic mode only) — WAVs at
   `data/voice/background_noise_audio_pcm_mono_verified/` (`continuous/`:
   people_talking, TV news, busy street, street & metro; `bursts/`: ringing
   phone, dog bark, car horn, engine idling, siren). They ship with
   sierra-research/tau2-bench's data assets; they are not vendored here.

With those in place, `run_voice.sh clean --smoke` is the recommended first run
(1 trial, `S1-lasa-001` + `S7-seek-001`), then scale to `realistic` and the
full split.
