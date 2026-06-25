# On-Policy Multi-Teacher Implementation Plan

## Progress

- [x] Phase 1 point 1: create `trl/experimental/multi_teacher_gkd/__init__.py`
- [x] Phase 1 point 2: create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`
- [x] Phase 1 point 3: create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`
- [x] Phase 1 point 4: add `tests/experimental/test_multi_teacher_gkd_trainer.py`
- [x] Phase 2: add thesis-specific diagnostics, metric logging, and ablation docs
- [x] Phase 3A: add `confidence_weighted` adaptive aggregation
- [x] Phase 3B: add `max_margin` adaptive aggregation
- [x] Phase 3C: add FuseLLM-style `min_ce` and `avg_ce` aggregation
- [x] Phase 3D: add dataset-domain routing with `domain_routed`

## Current Experiment Status

- The strongest completed MT-GKD method so far is `min_ce` with `lmbda=0.9`.
- `domain_routed` and `min_ce` off-policy ablations with `lmbda=0.0` have been added and evaluated.
- `max_margin` currently acts as a negative ablation: it is implemented, but its measured results are weak.
- Domain-filtered single-teacher GKD baselines have been added for instruct/general, math/math, and coder/code.
- Single-teacher math and coder baselines show weak specialist transfer so far; evaluate intermediate checkpoints before increasing epochs again.
- The current dataset has `general`, `math`, and `code` domains, but no explicit `instruct` domain. Treat IFEval as diagnostic until an instruction-following domain is added.
- Eval wrappers now cover `no_chat`, `code`, `chat_hellaswag`, and `chat_ifeval` modes for a single student checkpoint path.

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
- N-way consensus JSD
- hybrid GRPO-style scalar reward functions

Status update:
- adaptive teacher routing is no longer deferred; `confidence_weighted`, `max_margin`, `min_ce`, `avg_ce`, and `domain_routed` are implemented.

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

- structure the test file into three layers:
  - copied/shared-behavior tests mirrored from `tests/experimental/test_gkd_trainer.py`
  - parity tests showing the multi-teacher trainer reduces to GKD in the right special cases
  - new multi-teacher-specific tests for aggregation and fused-teacher generation

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

- `test_generate_from_fused_teachers`
  - directly test `_generate_from_fused_teachers(...)`
  - check shape, prompt-prefix preservation, labels, and attention mask
  - prefer deterministic generation settings where possible

- `test_aggregate_teacher_log_probs_uniform`
  - check uniform probability-space fusion behavior for multiple teachers

- `test_aggregate_teacher_log_probs_static_weighted`
  - check that static weights bias the fused distribution toward the heavier teacher

- mirror the existing GKD utility tests where behavior should stay identical:
  - deterministic `generate_on_policy_outputs(...)`
  - shape/type checks for `generate_on_policy_outputs(...)`

Test data/model strategy:
- use Hugging Face Hub internal test assets referenced as `trl-internal-testing/...`
- these are model and dataset repo IDs fetched through `from_pretrained(...)` and `load_dataset(...)`, not local folders inside this repository

Status:
- [x] created `tests/experimental/test_multi_teacher_gkd_trainer.py`
- [x] mirrored shared `generate_on_policy_outputs(...)` tests from `tests/experimental/test_gkd_trainer.py`
- [x] added adapted JSD/log-prob loss unit tests
- [x] added config validation tests for teacher weights and aggregation mode
- [x] added trainer smoke test
- [x] added single-teacher parity against `GKDTrainer`
- [x] added identical-teacher equivalence tests
- [x] added fused-teacher `seq_kd` generation tests
- [x] added uniform and static-weighted aggregation tests

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

Status:
- [x] added per-teacher selected-logprob, entropy, confidence, and weight metrics
- [x] added fused-teacher selected-logprob, entropy, and confidence metrics
- [x] added metric assertions to `tests/experimental/test_multi_teacher_gkd_trainer.py`
- [x] documented metric meanings in `docs/source/multi_teacher_gkd_trainer.md`
- [x] documented `uniform` vs `static_weighted` ablation guidance
- [ ] add prompt/completion diagnostics if needed for experiment debugging

### Phase 3: Adaptive Teacher Aggregation

Extend same files, no new abstraction layer.

Add config options:
- `teacher_aggregation = "confidence_weighted"`
- `teacher_aggregation = "max_margin"`
- `teacher_aggregation = "min_ce"`
- `teacher_aggregation = "avg_ce"`
- `teacher_aggregation = "domain_routed"`

Add trainer methods:
- `_aggregate_teacher_log_probs(...)` updated for dynamic per-token weights

Implementation order:
- Phase 3A: implement `confidence_weighted` first
- Phase 3B: implement `max_margin` after the confidence-weighted path is stable
- Phase 3C: implement FuseLLM-style CE aggregation with `min_ce` and `avg_ce`
- Phase 3D: implement dataset-domain routing with `domain_routed`

Status:
- [x] Phase 3A: `confidence_weighted`
- [x] Phase 3B: `max_margin`
- [x] Phase 3C: `min_ce` and `avg_ce`
- [x] Phase 3D: `domain_routed`

`confidence_weighted` definition from the thesis document:
- use selected-token teacher confidence, not entropy, as the primary implementation
- for teacher `k` and supervised/generated token `y_t`, compute:
  - `selected_logprob_k = log P_Tk(y_t | x)`
  - `C_k = 1 / (-selected_logprob_k + eps)`
  - `pi_k = softmax(C_k over teachers)`
- apply these dynamic `pi_k` weights per token before constructing the fused teacher distribution

`max_margin` definition from the thesis document:
- use winner-takes-all routing based on the selected-token probability margin between teacher and student
- for teacher `k`, compute:
  - `margin_k = abs(P_Tk(y_t | x) - Q(y_t | x))`
  - `pi_k = 1` for the teacher with the largest margin and `0` for all others
- this path requires student selected-token probabilities in the aggregation path

FuseLLM-style CE aggregation definitions:
- use the source-model cross-entropy against the selected supervised/generated tokens as a sequence-level quality score
- compute one average CE per teacher and example over valid target tokens
- `min_ce` selects the teacher distribution with the lowest sequence-level CE for that example
- `avg_ce` computes a weighted average of teacher distributions using FuseLLM-style rewards, where lower CE produces a larger reward
- these are online, shared-tokenizer adaptations of FuseLLM's fusion functions; heterogeneous tokenizer alignment remains deferred to Phase 5

Domain-routed aggregation definition:
- use a dataset-provided `domain` field to choose the teacher for each example
- expected initial domains are `general`, `math`, and `code`
- initial routing should map `general` to the instruction/general teacher, `math` to the math teacher, and `code` to the coder teacher
- this requires preserving the `domain` column through prompt-completion preparation and collation

Memory requirement:
- preserve the streaming aggregation design
- do not materialize `[num_teachers, batch_size, sequence_length, vocab_size]` tensors
- for adaptive modes, compute small per-teacher selected-token score tensors first, then stream-fuse full-vocabulary teacher distributions with dynamic token weights

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

## Experimental Plan

The implementation plan supports a staged thesis evaluation. Experiments should isolate one variable at a time:
static fusion, on-policy rollouts, sequence-level KD, adaptive aggregation, and multi-consensus objectives.

### Experiment 1: Static Uniform On-Policy Multi-Teacher GKD

Goal:
- validate the core On-Policy FuseLLM-style baseline
- test whether student rollouts plus fused-teacher JSD improve over static/off-policy imitation

Setup:
- student: `Qwen/Qwen2.5-1.5B`
- teachers:
  - `Qwen/Qwen2.5-1.5B-Instruct`
  - `Qwen/Qwen2.5-Math-1.5B`
  - `Qwen/Qwen2.5-Coder-1.5B`
- aggregation: `teacher_aggregation="uniform"`
- objective: fused-teacher JSD
- `seq_kd=False`
- sweep `lmbda` over values such as `0.25`, `0.5`, and `0.9`

Primary question:
- does increasing the student on-policy rollout fraction improve reasoning recovery and generalization?

### Experiment 2: Static Sequence-Level KD Ablation

Goal:
- isolate the effect of fused-teacher sequence generation

Setup:
- same Qwen2.5 1.5B student and teachers
- aggregation: `teacher_aggregation="uniform"`
- objective: fused-teacher JSD
- `seq_kd=True`
- `lmbda=0.0`

Primary question:
- does training on fused-teacher-generated completions help compared with student-generated on-policy rollouts?

Note:
- keep this separate from the main on-policy runs because `seq_kd` and `lmbda` control different rollout sources
- in the current implementation, the `lmbda` branch can overwrite the `seq_kd` branch, so mixed runs should be treated as later ablations

### Experiment 3: Mixed Rollout Ablation

Goal:
- test whether combining fused-teacher sequence KD with student rollouts improves stability or final quality

Setup:
- `seq_kd=True`
- sweep `lmbda`, with emphasis on `0.9` for mostly student rollouts
- compare against pure `seq_kd=True, lmbda=0.0` and pure on-policy `seq_kd=False, lmbda=0.9`

Primary question:
- is there value in occasional fused-teacher-generated trajectories once the student is mostly trained on its own rollouts?

Implementation note:
- this is conceptually valid but currently inefficient because fused-teacher generation runs before potential student-rollout overwrite

### Experiment 4: Static Weighted Aggregation

Goal:
- test whether fixed prior beliefs about teacher relevance outperform uniform fusion

Setup:
- same Qwen2.5 specialized teacher team
- aggregation: `teacher_aggregation="static_weighted"`
- example weights:
  - balanced: `[0.333, 0.333, 0.333]`
  - instruction-heavy: `[0.5, 0.25, 0.25]`
  - math/code-heavy: `[0.2, 0.4, 0.4]`

Primary question:
- can simple prior weighting reduce interference between instruction, math, and code teachers before adaptive routing is introduced?

### Experiment 5: Adaptive Aggregation With Fused-Teacher JSD

Goal:
- implement and evaluate adaptive teacher routing while keeping the current fused-teacher objective

Setup:
- add adaptive `teacher_aggregation` modes:
  - `"max_margin"`
  - `"confidence_weighted"`
  - `"min_ce"`
  - `"avg_ce"`
  - `"domain_routed"`
- objective: fused-teacher JSD
- apply dynamic teacher weights before constructing the fused teacher distribution

Primary question:
- does adaptive teacher weighting outperform uniform/static fusion by prioritizing the most relevant expert?

Implementation note:
- `confidence_weighted` should follow the thesis formula: `C_k = 1 / (-log P_Tk(y_t | x) + eps)`, then `softmax(C_k)` over teachers
- `max_margin` should follow the thesis formula: winner-takes-all on `abs(P_Tk(y_t | x) - Q(y_t | x))`
- `max_margin` requires student selected-token probabilities in the aggregation path
- `min_ce` and `avg_ce` should follow FuseLLM's sequence-level CE scoring rather than token-level routing
- `domain_routed` should use the dataset domain metadata to avoid off-domain teacher noise
- adaptive aggregation must preserve streaming full-vocabulary fusion to avoid multi-teacher OOMs

### Experiment 6: Multi-Distribution Consensus Objective

Goal:
- compare pre-fused teacher supervision against holistic N-way consensus supervision

Setup:
- add objective selector such as `loss_type="fused_jsd"` or `loss_type="multi_consensus"`
- keep the same teacher aggregation modes
- compute a shared mixture over student and all teachers: `M = pi_S * Q + sum_k pi_Tk * P_Tk`

Primary question:
- does the N-way generalized JSD objective improve robustness compared with first collapsing teachers into one fused pseudo-teacher?

Implementation note:
- this path cannot reuse only the pre-fused teacher distribution
- it must keep per-teacher log-probs and compute separate KL terms from student and each teacher to the shared mixture

### Experiment 7: Adaptive Sequence-Level KD Generation

Goal:
- decide whether adaptive teacher aggregation should also control fused-teacher `seq_kd` generation

Setup:
- treat this as a later ablation after adaptive loss-path routing is stable
- for `confidence_weighted`, weight teachers by next-token confidence or entropy at each decoding step
- for `max_margin`, decide between distribution-level student-teacher disagreement and confidence-margin routing

Primary question:
- should adaptive aggregation decide only how to supervise tokens, or also which teacher policy writes synthetic trajectories?

Open decision:
- `max_margin` generation is less direct because the next token is not known before generation
- likely options are distribution-level disagreement against the student or confidence-margin routing

### Baseline Experiments

Compare against:
- individual source models
- parameter-level model merging on the Qwen2.5 specialized track
- linear/model soup merging
- task arithmetic
- TIES merging
- DARE variants
- off-policy FuseLLM-style distillation
- off-policy InfiFusion-style distillation
- InfiFPO-style preference fusion if time allows

Primary question:
- does on-policy multi-teacher distillation provide gains beyond cheaper model merging and static/off-policy distillation?

### Hardware Plan

Preferred first hardware target:
- 4 x RTX A6000 46GB
- `fp16=True`
- `bf16=False`
- microbatch size 1
- gradient accumulation 8 or 16
- `max_length=512`
- `max_new_tokens=64` for on-policy runs
- `max_new_tokens=32` for first `seq_kd` runs

Fallback or scale-up target:
- 4 x H100 80GB
- `bf16=True`
- larger context length or longer `seq_kd` generations

Resource principle:
- start with 4 A6000s for demos
- use H100s only when A6000 memory or runtime becomes limiting
- avoid 8 A6000 runs until the training configuration is stable

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
