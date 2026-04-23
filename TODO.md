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

- [ ] Create `tests/experimental/test_multi_teacher_gkd_trainer.py`
- [ ] Add smoke train test
- [ ] Add invalid teacher weight test
- [ ] Add single-teacher parity test against `GKDTrainer`
- [ ] Add identical-teachers equivalence test
- [ ] Add on-policy generation path test
- [ ] Add fused-teacher `seq_kd` generation test

### Docs

- [x] Create `docs/source/multi_teacher_gkd_trainer.md`
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
