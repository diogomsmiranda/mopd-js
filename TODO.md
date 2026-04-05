# TODO

## Phase Overview

- Phase 1: Minimal working multi-teacher on-policy trainer
- Phase 2: Thesis-specific diagnostics and ablations
- Phase 3: Adaptive token-level teacher aggregation
- Phase 4: Heterogeneous-teacher support and advanced hybrids

## Phase 1: Minimal Working Trainer

- [x] Create `trl/experimental/multi_teacher_gkd/__init__.py`
- [x] Create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`
- [x] Add `MultiTeacherGKDConfig(GKDConfig)`
- [x] Add config fields:
  - [x] `teacher_model_names_or_paths`
  - [x] `teacher_weights`
  - [x] `teacher_aggregation`
- [x] Add config validation in `__post_init__`
- [ ] Create `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`
- [ ] Copy/adapt `GKDTrainer` into `MultiTeacherGKDTrainer`
- [ ] Implement `_load_teacher_models(...)`
- [ ] Implement `_prepare_teacher_models(...)`
- [ ] Implement `_normalize_teacher_weights(...)`
- [ ] Implement `_aggregate_teacher_log_probs(...)`
- [ ] Reuse/copy `generate_on_policy_outputs(...)`
- [ ] Reuse/copy `training_step(...)`
- [ ] Override `compute_loss(...)` for fused multi-teacher JSD

### Tests

- [ ] Create `tests/experimental/test_multi_teacher_gkd_trainer.py`
- [ ] Add smoke train test
- [ ] Add invalid teacher weight test
- [ ] Add single-teacher parity test against `GKDTrainer`
- [ ] Add identical-teachers equivalence test
- [ ] Add on-policy generation path test

### Docs

- [ ] Create `docs/source/multi_teacher_gkd_trainer.md`
- [ ] Add `MultiTeacherGKDTrainer` to `docs/source/index.md`
- [ ] Add dataset-format row to `docs/source/dataset_formats.md`

### Optional After MVP

- [ ] Add `examples/scripts/multi_teacher_gkd.py`

## Phase 2: Thesis-Specific Diagnostics

- [ ] Add per-teacher metrics and logging
- [ ] Log fused-teacher entropy / confidence diagnostics
- [ ] Add `static_weighted` vs `uniform` ablations
- [ ] Add trainer tests for metric logging
- [ ] Extend docs with ablation guidance and experiment notes

## Phase 3: Adaptive Token-Level Aggregation

- [ ] Add config options for adaptive aggregation
- [ ] Implement token-level teacher weighting
- [ ] Add `max_margin` teacher routing
- [ ] Add `confidence_weighted` teacher routing
- [ ] Update `_aggregate_teacher_log_probs(...)` for dynamic weights
- [ ] Add tests for adaptive routing behavior
- [ ] Add docs for adaptive aggregation modes

## Phase 4: Advanced Thesis Extensions

- [ ] Add N-way consensus JSD objective
- [ ] Add fused-teacher vs consensus objective ablation
- [ ] Investigate heterogeneous tokenizer support
- [ ] Investigate GRPO-style reward hybridization
- [ ] Evaluate whether cross-tokenizer alignment should live in this trainer or a separate experimental trainer
