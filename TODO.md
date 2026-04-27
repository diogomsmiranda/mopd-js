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

### Optional After MVP

- [ ] Add `examples/scripts/multi_teacher_gkd.py`

## Phase 2: Thesis-Specific Diagnostics

- [x] Add per-teacher metrics and logging
- [x] Log fused-teacher entropy / confidence diagnostics
- [x] Document `static_weighted` vs `uniform` ablations
- [x] Add trainer tests for metric logging
- [x] Extend docs with ablation guidance and experiment notes

## Phase 3: Adaptive Token-Level Aggregation

- [ ] Add config options for adaptive aggregation
- [ ] Implement token-level teacher weighting
- [ ] Add `max_margin` teacher routing
- [ ] Add `confidence_weighted` teacher routing
- [ ] Update `_aggregate_teacher_log_probs(...)` for dynamic weights
- [ ] Add tests for adaptive routing behavior
- [ ] Add docs for adaptive aggregation modes

## Phase 4: Advanced Thesis Extensions

- [ ] Add Multi-Consensus loss objective
- [ ] Add config option for `fused_jsd` vs `multi_consensus`
- [ ] Implement N-way consensus loss over teacher/student distributions
- [ ] Add fused-teacher vs consensus objective ablation
- [ ] Add adaptive-routing vs Multi-Consensus ablation

## Phase 5: Heterogeneous Teachers And Hybrids

- [ ] Investigate heterogeneous tokenizer support
- [ ] Investigate GRPO-style reward hybridization
- [ ] Evaluate whether cross-tokenizer alignment should live in this trainer or a separate experimental trainer

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
