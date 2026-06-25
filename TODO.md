# TODO

## Phase Overview

- Phase 1: Minimal working multi-teacher on-policy trainer
- Phase 2: Thesis-specific diagnostics and ablations
- Phase 3: Adaptive token-level teacher aggregation
- Phase 4: Multi-Consensus loss
- Phase 5: Heterogeneous-teacher support and advanced hybrids

## Phase 1: Minimal Working Trainer

- [x] Create `trl/experimental/multi_teacher_gkd/__init__.py`
- [x] Create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`
- [x] Add `MultiTeacherGKDConfig(GKDConfig)`
- [x] Add config fields:
  - [x] `teacher_model_names_or_paths`
  - [x] `teacher_weights`
  - [x] `teacher_aggregation`
- [x] Add config validation in `__post_init__`
- [x] Create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`
- [x] Copy/adapt `GKDTrainer` into `MultiTeacherGKDTrainer`
- [x] Implement `_load_teacher_models(...)`
- [x] Implement `_prepare_teacher_models(...)`
- [x] Implement `_normalize_teacher_weights(...)`
- [x] Implement `_aggregate_teacher_log_probs(...)`
- [x] Reuse/copy `generate_on_policy_outputs(...)`
- [x] Implement `_generate_from_fused_teachers(...)` for `seq_kd`
- [x] Update `training_step(...)` so `seq_kd` uses fused-teacher generation
- [x] Override `compute_loss(...)` for fused multi-teacher JSD
- [x] Verify there is no `teacher_models[0]` single-teacher shortcut in the Phase 1 rollout path

### Tests

- [x] Create `tests/experimental/test_multi_teacher_gkd_trainer.py`
- [x] Mirror shared utility tests from `tests/experimental/test_gkd_trainer.py` where behavior should remain identical
- [x] Add adapted JSD/loss unit tests for `generalized_jsd_loss_from_log_probs(...)`
- [x] Add smoke train test
- [x] Add invalid teacher weight test
- [x] Add single-teacher parity test against `GKDTrainer`
- [x] Add identical-teachers equivalence test
- [x] Add on-policy generation path test
- [x] Add fused-teacher `seq_kd` generation test
- [x] Add direct `_generate_from_fused_teachers(...)` test
- [x] Add uniform aggregation test for `_aggregate_teacher_log_probs(...)`
- [x] Add static-weighted aggregation test for `_aggregate_teacher_log_probs(...)`
- [x] Use `trl-internal-testing/...` Hub assets for tiny model/dataset coverage rather than local test folders

### Docs

- [x] Create `docs/source/multi_teacher_gkd_trainer.md`
- [x] Add `MultiTeacherGKDTrainer` to `docs/source/index.md`
- [x] Add dataset-format row to `docs/source/dataset_formats.md`

## Phase 2: Thesis-Specific Diagnostics

- [x] Add per-teacher metrics and logging
- [x] Log fused-teacher entropy / confidence diagnostics
- [x] Document `static_weighted` vs `uniform` ablations
- [x] Add trainer tests for metric logging
- [x] Extend docs with ablation guidance and experiment notes

## Phase 3: Adaptive Teacher Aggregation

- [x] Add config options for adaptive aggregation
- [x] Add `confidence_weighted` teacher routing first
- [x] Implement `confidence_weighted` with the thesis formula `C_k = 1 / (-log P_Tk(y_t | x) + eps)` and `softmax(C_k)` over teachers
- [x] Add tests showing `confidence_weighted` favors the teacher with higher selected-token confidence
- [x] Preserve streaming aggregation for adaptive modes; do not stack `[num_teachers, batch_size, sequence_length, vocab_size]`
- [x] Update `_aggregate_teacher_log_probs(...)` for dynamic per-token weights
- [x] Add tests proving `uniform` and `static_weighted` behavior remains unchanged after adaptive changes
- [x] Add `max_margin` teacher routing after `confidence_weighted` is stable
- [x] Implement `max_margin` with the thesis formula `argmax_k abs(P_Tk(y_t | x) - Q(y_t | x))`
- [x] Pass student selected-token probabilities into the aggregation path for `max_margin`
- [x] Add tests showing `max_margin` selects the teacher with largest teacher-student selected-token probability gap
- [x] Add FuseLLM-style `min_ce` teacher routing
- [x] Implement `min_ce` as sequence-level teacher selection by lowest average selected-token CE
- [x] Add FuseLLM-style `avg_ce` teacher routing
- [x] Implement `avg_ce` as sequence-level weighted fusion using CE-derived teacher rewards
- [x] Add tests showing `min_ce` selects the lowest-CE teacher per example
- [x] Add tests showing `avg_ce` gives larger sequence-level weights to lower-CE teachers
- [x] Add `domain_routed` teacher routing
- [x] Preserve dataset `domain` metadata through prompt-completion preparation and collation
- [x] Implement initial domain mapping for `general`, `math`, and `code`
- [x] Add tests showing `domain_routed` selects the expected teacher for each domain
- [x] Add docs for implemented adaptive aggregation modes
- [x] Add docs for `domain_routed` after implementation

## Phase 4: Advanced Thesis Extensions

- [ ] Add Multi-Consensus loss objective
- [ ] Add config option for `fused_jsd` vs `multi_consensus`
- [ ] Implement N-way consensus loss over teacher/student distributions
- [ ] Add fused-teacher vs consensus objective ablation
- [ ] Add adaptive-routing vs Multi-Consensus ablation

## Experiment And Evaluation Tracking

- [x] Add Slurm wrapper for domain-filtered single-teacher GKD baselines
- [x] Add dataset splitting script for `general`, `math`, and `code` domain subsets
- [x] Add single-teacher eval wrappers for instruct/general, math/math, and coder/code baselines
- [x] Update eval defaults to cover `no_chat`, `code`, `chat_hellaswag`, and `chat_ifeval`
- [x] Document dataset provenance, domain counts, split sizes, and length statistics
- [x] Evaluate `domain_routed` and `min_ce` off-policy (`lmbda=0.0`) ablations
- [x] Evaluate domain-filtered single-teacher GKD baselines
- [x] Inspect local W&B logs for MT-GKD and single-GKD training-curve diagnostics
- [x] Update result interpretation so `Specialist Rel.` tracks GSM8K/math, HumanEval/code, and general-core retention
- [ ] Evaluate intermediate checkpoints for weak single-teacher math/code runs before trying longer training
- [ ] Add an explicit `instruct` domain to the dataset if IFEval becomes a primary specialist metric
- [x] Audit `BAAI/Infinity-Instruct` `Gen` labels with `mopd-slurm/audit_infinity_instruct.py` before reconstructing the dataset
- [x] Add exact `ability_en` / `cate_ability_en` filters for the clean `instruct` domain rebuild
- [ ] Reconstruct the Qwen3 teacher dataset from source with `instruct`, `math`, and `code` domains
- [ ] Run a targeted `max_length=2048` ablation for math/code specialist transfer if checkpoint diagnostics justify it
- [ ] Rebuild result tables after any new checkpoint or length-ablation evals
- [ ] Use denser eval/checkpointing for future diagnostic runs: `eval_steps=100`, `save_steps=100`, `save_total_limit=6`

## Phase 5: Heterogeneous Teachers And Hybrids

- [ ] Investigate heterogeneous tokenizer support
- [ ] Investigate GRPO-style reward hybridization
- [ ] Evaluate whether cross-tokenizer alignment should live in this trainer or a separate experimental trainer

## Non-Critical Later Work

- [ ] Add a `docs/source/paper_index.md` entry for the multi-teacher/fused-teacher distillation method
- [ ] Add `examples/scripts/multi_teacher_gkd.py`

## Optimisations

- [ ] Add cached fused-teacher decoding with per-teacher `past_key_values`
- [ ] Benchmark cached vs uncached fused-teacher `seq_kd` rollout speed
- [ ] Add a generation buffer for student on-policy rollouts across gradient accumulation steps
- [ ] Benchmark buffered vs unbuffered rollout generation
- [ ] Add optional top-k distillation approximation for fused teacher aggregation / comparison
- [ ] Benchmark full-vocab vs top-k distillation cost and quality
- [ ] Investigate external teacher server support for larger teachers or more concurrent teachers
- [ ] Investigate request batching for external teacher scoring
- [ ] Investigate compact/binary encoding for transferred teacher log-probs
