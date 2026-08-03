# Leaderboard submission conventions

> The leaderboard publishes historical pilot rows and accepts submissions under
> these conventions, which mirror
> [τ²-bench's leaderboard guide](https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md)
> so that results produced now remain submittable later.

## Requirements

1. **All tasks.** Run the full `base` split (all 111 tasks). Do not filter
   with `--task-ids` or `--num-tasks`; partial runs are not rankable.
2. **≥ 4 trials.** We strongly prefer at least 4 trials per task
   (`--num-trials 4`) so pass^2..pass^4 can be computed. Single-trial results
   may be listed but will be marked as such.
3. **Consistent configuration.** The same agent model, user-simulator model,
   and arguments across the whole run. Both models are disclosed on the
   leaderboard — the user simulator materially affects difficulty.
4. **Disclose the judge.** Report the `TAU2_LLM_NL_ASSERTIONS` model used for
   NL-assertion grading.
5. **No lost simulations.** Repair infrastructure casualties
   (`rxbench repair`) or report the run as incomplete. Scorecards state the
   excluded-simulation count; a submission with silent holes will be rejected.
6. **Fixed step budget.** Use `--max-steps 40` (the reference budget used by
   `scripts/run.sh` and the baselines). A front-desk call that needs 200
   turns has already failed the ease-of-use bar.

Repository-owned pre-audit pilot rows are historical exemptions from these
requirements and are labeled preliminary; they are not precedents for new
ranked submissions.

## Standard vs custom scaffold

**Standard** submissions evaluate a general-purpose LLM with the default
τ²-bench agent scaffold, the shipped tool set, the shipped `policy.md`, and
unmodified prompts. If that describes your run, it is standard by default.

**Custom** submissions are anything else — multi-model routing, extra tools,
modified prompts or orchestration, or models trained/fine-tuned on medical
front-office data or on this benchmark's tasks. Custom submissions must:

- be labeled `custom`;
- include methodology notes describing what was changed and why;
- link an implementation reference (repo, paper, or blog post).

Training on ℞-bench task data and submitting as standard is
misrepresentation and will be removed.

## What to submit

Produce runs with `rxbench run` (results land in
`data/simulations/<run_name>/results.json`) and score them with
`rxbench score`. A submission bundle is:

```
submission/
├── submission.json          # model ids + args, judge model, trials, scaffold type, contact
├── results.json             # raw tau2 results (full trajectories)
└── scorecard.json           # rxbench score output (records policy + task-set hashes)
```

Keep the raw `results.json`: scores are re-verified from trajectories, and
the recorded policy hash must match the `data/v1` release you ran against.
Results from different data versions are never compared.

Open a GitHub issue titled
`[submission] <model> on data/v1` with the bundle attached or linked.
