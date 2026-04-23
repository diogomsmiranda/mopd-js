# On-Policy Multi-Teacher Implementation Plan

## Progress

- [x] Phase 1 point 1: create `trl/experimental/multi_teacher_gkd/__init__.py`
- [x] Phase 1 point 2: create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`
- [x] Phase 1 point 3: create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`
- [ ] Phase 1 point 4: add `tests/experimental/test_multi_teacher_gkd_trainer.py`

## Goal

Add an experimental trainer for on-policy multi-teacher distillation inside TRL, reusing the existing on-policy GKD pattern and extending it to support multiple frozen teacher LMs with token-level aggregation.

## Core Decision

Implement this as a new experimental trainer under `trl/experimental/`, not in `trl/trainer/`.

- Keep it experimental first, consistent with `AGENTS.md:7`.
- Keep the trainer self-contained, consistent with `AGENTS.md:22`.
- Use `trl/experimental/gkd/gkd_trainer.py:53` as the main source to copy/adapt, because it already has:
  - on-policy generation via `generate_on_policy_outputs` at `trl/experimental/gkd/gkd_trainer.py:394`
  - on-policy switching in `training_step` at `trl/experimental/gkd/gkd_trainer.py:416`
  - JSD-based KD loss in `compute_loss` at `trl/experimental/gkd/gkd_trainer.py:292`

## Scope For V1

Target the specialized-track setup first:

- student: one causal LM
- teachers: multiple causal LMs
- assumption: shared tokenizer / same token space
- objective: fused-teacher JSD on student rollouts or fused-teacher rollouts
- aggregation: uniform or static weighted average over teacher probabilities
- generation: if `seq_kd` is used, generation must already come from the fused teacher policy, not from a single teacher shortcut

Defer to later phases:

- heterogeneous tokenizers
- adaptive token-level routing
- N-way consensus JSD
- hybrid GRPO-style scalar reward functions

## Exact Files To Add

### 1. `trl/experimental/multi_teacher_gkd/__init__.py`

Add:
- `MultiTeacherGKDConfig`
- `MultiTeacherGKDTrainer`

Pattern to follow:
- `trl/experimental/gkd/__init__.py`
- `trl/experimental/minillm/__init__.py`

### 2. `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`

Add:
- `class MultiTeacherGKDConfig(GKDConfig)`

Base file to mirror:
- `trl/experimental/gkd/gkd_config.py:21`

Fields for V1:
- `teacher_model_names_or_paths: list[str] | None = None`
- `teacher_weights: list[float] | None = None`
- `teacher_aggregation: str = "uniform"`
- reuse inherited:
  - `temperature`
  - `lmbda`
  - `beta`
  - `max_new_tokens`
  - `teacher_model_init_kwargs`
  - `disable_dropout`
  - `seq_kd`

`__post_init__` validations:
- at least one teacher is provided
- if `teacher_weights` is set, its length matches number of teachers
- weight sum is positive
- `teacher_aggregation` is one of:
  - `"uniform"`
  - `"static_weighted"`

Notes:
- do not add tokenizer-alignment options yet
- do not add max-margin routing yet

### 3. `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`

Add:
- `class MultiTeacherGKDTrainer(SFTTrainer)`

Important: this should be a self-contained copy/adaptation of `GKDTrainer`, not a deep abstraction layer.

Source file to copy from:
- `trl/experimental/gkd/gkd_trainer.py`

Methods to include:

- `__init__(...)`
  - same shape as `GKDTrainer.__init__`
  - replace `teacher_model` with `teacher_models: list[PreTrainedModel | nn.Module | str]`
  - load all teacher models
  - prepare all teachers with Accelerate / DeepSpeed
  - normalize static teacher weights

- `_load_teacher_models(...)`
  - instantiate teachers from strings using `AutoModelForCausalLM.from_pretrained`
  - use shared `teacher_model_init_kwargs`

- `_prepare_teacher_models(...)`
  - prepare each teacher in eval mode
  - store as `self.teacher_models`

- `_normalize_teacher_weights(...)`
  - convert weights to normalized tensor
  - default to uniform if none provided

- `generalized_jsd_loss(...)`
  - copy from `GKDTrainer`
  - keep the same behavior and signature style if possible

- `_aggregate_teacher_log_probs(...)`
  - run all teachers on the same student rollout
  - compute per-teacher log-probs for the generated region
  - aggregate with weighted log-sum-exp over teacher probabilities

- `compute_loss(...)`
  - same flow as `GKDTrainer.compute_loss`
  - get student logits
  - get aggregated teacher distribution
  - slice to generated tokens only
  - compute fused-teacher JSD loss

- `generate_on_policy_outputs(...)`
  - copy from `trl/experimental/gkd/gkd_trainer.py:394`

- `_generate_from_fused_teachers(...)`
  - generate tokens autoregressively from the fused teacher policy
  - compute each teacher next-token distribution at every decoding step
  - aggregate them with static weights in probability space
  - sample the next token from the fused distribution

- `training_step(...)`
  - keep the student rollout branch from `trl/experimental/gkd/gkd_trainer.py:416`
  - if `seq_kd` is enabled, use fused-teacher generation instead of `teacher_models[0]`

Implementation detail for V1:
- aggregate teachers in probability space, not logit space
- use:
  - `teacher_log_probs_k = log_softmax(teacher_logits_k / temperature)`
  - `ensemble_log_probs = logsumexp(log(w_k) + teacher_log_probs_k, dim=teachers)`
- reuse the same probability-space fusion rule during both loss computation and fused-teacher decoding

Do not add in V1:
- custom token alignment helpers
- extra utility modules
- GRPO reward plumbing

## Exact Files To Add For Tests

### 4. `tests/experimental/test_multi_teacher_gkd_trainer.py`

Base test to mirror:
- `tests/experimental/test_gkd_trainer.py`

Tests to implement:

- `test_multi_teacher_gkd_train_smoke`
  - tiny dataset
  - tiny student
  - two tiny teachers
  - one short train run
  - assert train loss exists
  - assert model params changed

- `test_teacher_weight_validation`
  - mismatched length raises
  - zero-sum weights raises

- `test_single_teacher_parity_with_gkd`
  - one teacher in `MultiTeacherGKDTrainer`
  - same config and same batch as `GKDTrainer`
  - losses should match or be very close

- `test_identical_teachers_equal_single_teacher`
  - two identical teachers with uniform weights
  - aggregated loss should match single-teacher case

- `test_on_policy_generation_path`
  - inherited/copied rollout path still returns valid tensors
  - mirror `tests/experimental/test_gkd_trainer.py:28`

## Existing Files To Update

### 5. `docs/source/multi_teacher_gkd_trainer.md`

Create a new docs page modeled on:
- `docs/source/gkd_trainer.md:1`

Contents:
- overview
- how multi-teacher fused JSD works
- usage example
- expected dataset type
- autodoc blocks for:
  - `experimental.multi_teacher_gkd.MultiTeacherGKDTrainer`
  - `experimental.multi_teacher_gkd.MultiTeacherGKDConfig`

### 6. `docs/source/index.md`

Add:
- `MultiTeacherGKDTrainer` under the `Knowledge distillation` section

Reference section:
- `docs/source/index.md:52`

### 7. `docs/source/dataset_formats.md`

Add one row:
- `experimental.multi_teacher_gkd.MultiTeacherGKDTrainer` -> `Prompt-completion`

Reference table:
- `docs/source/dataset_formats.md:378`

## Optional But Recommended Later

### 8. `examples/scripts/multi_teacher_gkd.py`

Mirror:
- `examples/scripts/gkd.py:24`

Purpose:
- reproducible training entrypoint for thesis experiments
- easier to run baselines and ablations

This is optional for the first code pass, but useful before running real experiments.

## Files That Do Not Need Changes In V1

- `trl/__init__.py`
- `trl/trainer/__init__.py`
- `trl/trainer/grpo_trainer.py`
- `trl/trainer/rloo_trainer.py`

Reason:
- this stays experimental
- no top-level stable export needed yet
- no need to touch main trainer code for the first thesis version

## Phased Delivery Plan

### Phase 1: Minimal Working Trainer

Files:
- `trl/experimental/multi_teacher_gkd/__init__.py`
- `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`
- `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`
- `tests/experimental/test_multi_teacher_gkd_trainer.py`

Behavior:
- multiple teachers
- shared tokenizer only
- uniform or static weighted teacher fusion
- fused-teacher JSD on student rollouts
- fused-teacher decoding for `seq_kd` rollouts

Success criteria:
- smoke test passes
- single-teacher parity passes
- two-teacher aggregation passes
- `seq_kd` uses fused-teacher generation rather than a single teacher shortcut

### Phase 2: Thesis-Specific Diagnostics

Extend:
- `multi_teacher_gkd_config.py`
- `multi_teacher_gkd_trainer.py`
- `test_multi_teacher_gkd_trainer.py`

Add:
- per-teacher metrics
- ensemble entropy metrics
- teacher weight logging
- prompt/completion diagnostics
- ablation flags for:
  - `uniform`
  - `static_weighted`

### Phase 3: Adaptive Teacher Aggregation

Extend same files, no new abstraction layer.

Add config options:
- `teacher_aggregation = "max_margin"` or `"confidence_weighted"`

Add trainer methods:
- `_compute_teacher_token_weights(...)`
- `_aggregate_teacher_log_probs(...)` updated for dynamic per-token weights

This is where the thesis-specific adaptive routing starts.

### Phase 4: Multi-Consensus Loss

Only do this after Phase 1 static fusion and Phase 3 adaptive routing are stable.

Goal:
- add a Multi-Consensus objective as a second multi-teacher loss family, separate from fused-teacher JSD

Likely changes:
- extend `multi_teacher_gkd_config.py` with a loss/objective selector
- add a new consensus loss path in `multi_teacher_gkd_trainer.py`
- compare fused-teacher supervision against consensus-style supervision in tests and experiments

Suggested work:
- add `loss_type = "fused_jsd" | "multi_consensus"`
- implement an N-way consensus objective over teacher/student distributions
- add ablations comparing:
  - static fused teacher
  - adaptive teacher aggregation
  - Multi-Consensus loss

Why it comes after adaptive routing:
- adaptive routing is the smaller extension of the current Phase 1 implementation
- Multi-Consensus is an objective-level change and is easier to evaluate once the fused baseline is trusted

### Phase 5: Heterogeneous Teacher Support

Only do this after Phase 1-4 are stable.

Likely changes:
- extend config with teacher processing/alignment fields
- adapt trainer to compare aligned distributions across tokenizers

Reference only:
- `trl/experimental/gold/gold_trainer.py`
- `trl/experimental/gold/gold_config.py`

This should not be part of the MVP.

## Optimisations

These optimisations are inspired by `distilling-100b-models-40x-faster-with-trl.pdf` and should be treated as
performance work after the Phase 1 trainer is correct and tested.

### 1. Cached Fused-Teacher Decoding

Current bottleneck:
- `_generate_from_fused_teachers(...)` currently recomputes each teacher over the full prefix at every decoding step.

Planned optimisation:
- keep `past_key_values` for each teacher during fused generation
- feed only the newly generated token after the first step
- preserve the same fused probability-space decoding rule

Why it matters:
- this is the biggest Phase 1 runtime bottleneck
- it should reduce fused-teacher `seq_kd` rollout cost substantially

### 2. Generation Buffer For Student Rollouts

Current bottleneck:
- student rollout generation still follows the usual per-step training flow and does not yet exploit a buffer across
  gradient accumulation steps.

Planned optimisation:
- accumulate prompts across a generation window
- batch rollout generation while keeping model weights fixed before the optimizer step
- feed buffered completions back into the sequential train loop

Why it matters:
- this is the main throughput improvement highlighted in the TRL distillation paper
- it improves generation efficiency without breaking the on-policy setting

### 3. Optional Top-k Distillation Approximation

Current bottleneck:
- the current fused multi-teacher implementation uses full-vocabulary distributions for aggregation and JSD.

Planned optimisation:
- support an optional top-k approximation for teacher and/or student log-prob comparisons
- preserve the exact full-vocab path as the correctness baseline

Why it matters:
- reduces memory and compute cost for long sequences and multiple teachers
- becomes increasingly valuable as the number of teachers grows

### 4. External Teacher Server

Current bottleneck:
- all teachers are currently colocated with training.

Planned optimisation:
- move teacher scoring to an external vLLM-style service when model scale or teacher count makes colocated inference
  inefficient
- batch concurrent requests on the server side
- consider compact/binary payload encoding for returned log-prob data

Why it matters:
- useful once teacher scale grows beyond the comfortable colocated setup
- likely more relevant for later thesis phases than for the initial 1.5B shared-tokenizer experiments

## Recommended Class And Package Names

- package: `multi_teacher_gkd`
- config: `MultiTeacherGKDConfig`
- trainer: `MultiTeacherGKDTrainer`

Reason:
- matches existing TRL naming
- makes the relationship to GKD explicit
- keeps the first implementation narrow and publishable inside the repo

## First Implementation Order

1. Create `multi_teacher_gkd_config.py`
2. Create `multi_teacher_gkd_trainer.py` by copying/adapting `GKDTrainer`
3. Add `__init__.py`
4. Add smoke and parity tests
5. Add docs page
6. Update docs index and dataset table
7. Add example script only after tests pass

## Non-Goals For First PR

- no separate reward models
- no GRPO hybrid loss
- no cross-tokenizer support
- no vLLM-specific teacher serving
- no extra shared utility module

The first PR should establish one clean experimental trainer that works for the specialized Qwen-family track.
