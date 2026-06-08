# Multi-Teacher GKD Trainer

This document explains what is currently implemented inside `trl/experimental/multi_teacher_gkd/`.

The package currently contains three files:

- `trl/experimental/multi_teacher_gkd/__init__.py`
- `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`
- `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`

The goal of this package is to extend TRL's single-teacher GKD setup into a multi-teacher version where:

- multiple frozen teacher language models are loaded
- their token distributions are fused in probability space
- the student is trained against that fused target
- `seq_kd` can generate from a fused teacher policy instead of a single teacher

## Expected dataset type

The trainer supports conversational datasets with a `messages` column and prompt-completion datasets with `prompt` and
`completion` columns. Raw prompt-completion datasets are tokenized by the trainer using the same prompt plus
prompt-completion pattern as `SFTTrainer`; the default collator then batches the normalized examples into full
`input_ids`, supervised `labels`, prompt-only `prompts`, and `prompt_attention_mask` for on-policy generation.

## File overview

### `trl/experimental/multi_teacher_gkd/__init__.py`

This file is the package entrypoint. It re-exports the public objects that should be importable from the package.

Current code:

```python
from .multi_teacher_gkd_config import MultiTeacherGKDConfig
from .multi_teacher_gkd_trainer import MultiTeacherGKDTrainer


__all__ = ["MultiTeacherGKDConfig", "MultiTeacherGKDTrainer"]
```

What it does:

- exposes `MultiTeacherGKDConfig`
- exposes `MultiTeacherGKDTrainer`
- defines the public API of the folder

### `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`

This file defines the training configuration specific to the multi-teacher version.

It inherits from `GKDConfig`, so it reuses the normal GKD settings and adds only the multi-teacher-specific fields.

### `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_trainer.py`

This is the main implementation file.

It contains the trainer that:

- loads multiple teachers
- prepares them for evaluation
- combines teacher distributions with static weights
- computes fused-teacher JSD distillation loss
- supports student rollouts and fused-teacher rollouts

## `MultiTeacherGKDConfig`

Main class:

```python
@dataclass
class MultiTeacherGKDConfig(GKDConfig):
```

This extends `GKDConfig` rather than starting from scratch. That means the new trainer still inherits the usual GKD behavior, including things like `lmbda`, `beta`, `temperature`, `max_new_tokens`, `seq_kd`, and `teacher_model_init_kwargs`.

### Fields

#### `teacher_model_names_or_paths`

```python
teacher_model_names_or_paths: list[str] | None = field(
    default=None,
    metadata={"help": "Model names or paths of the teacher models."},
)
```

What it means:

- a list of teacher checkpoints
- these are the teachers the trainer will load if teacher models are not passed directly as instantiated model objects
- this can be `None` when instantiated teacher models are passed through `MultiTeacherGKDTrainer(teacher_models=...)`

#### `teacher_weights`

```python
teacher_weights: list[float] | None = field(
    default=None,
    metadata={"help": "Static weights used to aggregate the teacher distributions."},
)
```

What it means:

- optional fixed weights for teacher fusion
- if omitted, the trainer defaults to uniform weights

#### `teacher_aggregation`

```python
teacher_aggregation: str = field(
    default="uniform",
    metadata={
        "help": "Strategy used to aggregate teacher distributions. Supported values are 'uniform', "
        "'static_weighted', 'confidence_weighted', and 'max_margin'."
    },
)
```

What it means:

- controls how teacher distributions are fused
- current options are:
  - `"uniform"`
  - `"static_weighted"`
  - `"confidence_weighted"`
  - `"max_margin"`

`confidence_weighted` is the Phase 3A adaptive routing mode. It uses the thesis confidence formula based on the
selected supervised/generated token: `C_k = 1 / (-log P_Tk(y_t | x) + eps)`, followed by a softmax over teachers.

`max_margin` is the Phase 3B adaptive routing mode. It computes the selected-token probability gap between each
teacher and the student, `abs(P_Tk(y_t | x) - Q(y_t | x))`, and routes each token to the teacher with the largest gap.

### Validation in `__post_init__`

Code:

```python
def __post_init__(self):
    super().__post_init__()

    if self.teacher_model_names_or_paths is not None and len(self.teacher_model_names_or_paths) == 0:
        raise ValueError("teacher_model_names_or_paths must contain at least one teacher model when provided.")

    if self.teacher_aggregation not in ["uniform", "static_weighted", "confidence_weighted", "max_margin"]:
        raise ValueError(
            "teacher_aggregation must be one of "
            "['uniform', 'static_weighted', 'confidence_weighted', 'max_margin']."
        )

    if self.teacher_weights is not None:
        if self.teacher_model_names_or_paths is not None and len(self.teacher_weights) != len(
            self.teacher_model_names_or_paths
        ):
            raise ValueError("teacher_weights must have the same length as teacher_model_names_or_paths.")
        if sum(self.teacher_weights) <= 0:
            raise ValueError("teacher_weights must have a strictly positive sum.")
```

Why it exists:

- rejects an explicitly empty teacher checkpoint list
- prevents unsupported aggregation modes
- ensures static weights are aligned with the teacher checkpoint list when checkpoint names are configured
- ensures the weight vector is meaningful

## `MultiTeacherGKDTrainer`

Main class:

```python
class MultiTeacherGKDTrainer(SFTTrainer):
```

This trainer follows the same overall style as `GKDTrainer`, but changes the teacher side from a single teacher to multiple teachers.

Conceptually, the trainer has four big jobs:

1. load and prepare multiple teachers
2. build a fused teacher distribution
3. train the student against that fused target
4. optionally generate from the fused teacher policy during `seq_kd`

## Function-by-function explanation

### `__init__(...)`

Start of the constructor:

```python
def __init__(
    self,
    model: PreTrainedModel | nn.Module | str | None = None,
    teacher_models: list[PreTrainedModel | nn.Module | str] | None = None,
    args: MultiTeacherGKDConfig | None = None,
    ...
):
```

This is the main setup method.

Important early logic:

```python
if teacher_models is None:
    teacher_models = args.teacher_model_names_or_paths
if teacher_models is None or len(teacher_models) == 0:
    raise ValueError("teacher_models must contain at least one teacher model.")
if args.teacher_weights is not None and len(args.teacher_weights) != len(teacher_models):
    raise ValueError("teacher_weights must have the same length as teacher_models.")
```

Meaning:

- teachers can be provided directly to the trainer
- or they can come from the config
- the trainer validates the resolved teacher list before initialization continues

Then the constructor sets up the collator path:

```python
args.remove_unused_columns = False
if data_collator is None:
    data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)
```

Why:

- the trainer needs conversational/prompt columns such as `prompts`
- so it cannot let the Trainer drop those columns

Then it preserves raw conversational fields:

```python
if args.dataset_kwargs is None:
    args.dataset_kwargs = {"skip_prepare_dataset": True}
else:
    args.dataset_kwargs["skip_prepare_dataset"] = True
```

Why:

- this matches the same pattern used in GKD
- it lets the ChatML collator access the raw dataset structure

Then it calls `super().__init__(...)`, which initializes the normal `SFTTrainer` machinery.

After that, the multi-teacher-specific state is created:

```python
self.teacher_aggregation = args.teacher_aggregation
self.teacher_models = self._prepare_teacher_models(teacher_models, args.teacher_model_init_kwargs)
self.teacher_weights = self._normalize_teacher_weights(args.teacher_weights)
```

Meaning:

- store the fusion strategy
- load and prepare all teacher models
- normalize their weights into a tensor that sums to 1

Then normal GKD-style runtime parameters are copied:

```python
self.lmbda = args.lmbda
self.beta = args.beta
self.temperature = args.temperature
self.seq_kd = args.seq_kd
```

These control:

- how often student rollouts happen
- how the JSD mixture is computed
- temperature scaling
- whether teacher-generated sequence KD is used

Finally, the generation config is built:

```python
generation_kwargs = {
    "max_new_tokens": args.max_new_tokens,
    "temperature": args.temperature,
    "do_sample": True,
    "top_k": 0,
    "use_cache": False if args.gradient_checkpointing else True,
    "pad_token_id": self.processing_class.pad_token_id,
}
self.generation_config = GenerationConfig(**generation_kwargs)
self.generation_kwargs = generation_kwargs
```

Why this matters:

- both student generation and fused-teacher generation depend on this config
- it keeps training-time generation behavior explicit and reproducible

### `_load_teacher_models(...)`

Code shape:

```python
def _load_teacher_models(self, teacher_models, teacher_model_init_kwargs):
```

This method resolves the raw teacher inputs into actual model objects.

Important validation:

```python
if teacher_model_init_kwargs is None:
    teacher_model_init_kwargs = {}
elif any(not isinstance(teacher_model, str) for teacher_model in teacher_models):
    raise ValueError(...)
```

Meaning:

- if loading kwargs are supplied, all teachers must be checkpoint strings
- if even one teacher is already instantiated, load-time kwargs are not allowed

Then dtype is normalized:

```python
teacher_model_init_kwargs["dtype"] = (
    teacher_model_init_kwargs["dtype"]
    if teacher_model_init_kwargs["dtype"] in ["auto", None]
    else getattr(torch, teacher_model_init_kwargs["dtype"])
)
```

Why:

- config values are often strings like `"bfloat16"`
- the actual loader needs the real torch dtype object

Then all string teachers are loaded:

```python
for teacher_model in teacher_models:
    if isinstance(teacher_model, str):
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)
    loaded_teacher_models.append(teacher_model)
```

This method does not prepare the models for Accelerate yet. It only turns the raw teacher list into actual model objects.

### `_prepare_teacher_models(...)`

Code shape:

```python
def _prepare_teacher_models(self, teacher_models, teacher_model_init_kwargs):
```

This method takes the models returned by `_load_teacher_models(...)` and moves them into the right evaluation setup.

Core logic:

```python
for teacher_model in loaded_teacher_models:
    if self.is_deepspeed_enabled:
        teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
    else:
        teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
    prepared_teacher_models.append(teacher_model)
```

Meaning:

- if DeepSpeed is active, prepare each teacher through DeepSpeed
- otherwise, use Accelerate in evaluation mode
- all prepared teachers are stored in `self.teacher_models`

### `_normalize_teacher_weights(...)`

Code:

```python
def _normalize_teacher_weights(self, teacher_weights: list[float] | None) -> torch.Tensor:
    if self.teacher_aggregation != "static_weighted" or teacher_weights is None:
        normalized_teacher_weights = torch.ones(len(self.teacher_models), dtype=torch.float32)
    else:
        normalized_teacher_weights = torch.tensor(teacher_weights, dtype=torch.float32)
    return normalized_teacher_weights / normalized_teacher_weights.sum()
```

What it does:

- if uniform or adaptive fusion is selected, initialize every teacher with the same base weight
- if static weighted fusion is selected, use the provided list
- normalize the vector so the weights sum to 1

This is important because the fusion later assumes a proper convex combination of teacher distributions.

### `generalized_jsd_loss_from_log_probs(...)`

Code shape:

```python
@staticmethod
def generalized_jsd_loss_from_log_probs(student_log_probs, teacher_log_probs, labels=None, beta=0.5):
```

Why this exists:

- in single-teacher GKD, the loss can start from one teacher logit tensor
- in multi-teacher GKD, teachers are fused first in probability space
- so by the time the loss is called, the trainer already has fused teacher log-probabilities

This function therefore takes `student_log_probs` and `teacher_log_probs` directly instead of raw teacher logits.

Core mixture logic:

```python
mixture_log_probs = torch.logaddexp(
    student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)
)
```

Meaning:

- form the generalized JSD mixture between student and teacher
- everything stays in log-probability space for numerical stability

Then KL terms are computed:

```python
kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
jsd = beta * kl_teacher + (1 - beta) * kl_student
```

When `labels` are provided, the implementation computes the same masked JSD over valid supervised positions in chunks.
This keeps full-vocabulary KL tensors bounded during large multi-teacher runs while preserving the same objective. Padding
tokens are masked using `labels != -100`, and the loss is reduced by the number of valid supervised tokens.

### `_aggregate_teacher_log_probs(...)`

Code shape:

```python
def _aggregate_teacher_log_probs(self, input_ids, attention_mask, prompt_lengths, shifted_labels=None):
```

This method is the core of the multi-teacher loss path.

It does three things:

1. run every teacher on the same sequence
2. extract teacher distributions on the generated region only
3. fuse those teacher distributions with static or adaptive token weights without stacking all teachers at once

Per-teacher forward pass:

```python
for teacher_idx, teacher_model in enumerate(self.teacher_models):
    teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
```

Then it slices the teacher logits to the completion region:

```python
shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
teacher_log_probs = F.log_softmax(shifted_teacher_logits, dim=-1)
```

Why `prompt_lengths - 1 : -1`:

- this matches causal LM shifting
- the model predicts the next token from the token before it

Finally, fusion happens here:

```python
teacher_weight = teacher_weights[teacher_idx].to(dtype=teacher_log_probs.dtype).log()
teacher_log_probs.add_(teacher_weight)
if aggregated_teacher_log_probs is None:
    aggregated_teacher_log_probs = teacher_log_probs
else:
    aggregated_teacher_log_probs = torch.logaddexp(aggregated_teacher_log_probs, teacher_log_probs)
return teacher_metrics, aggregated_teacher_log_probs
```

Meaning:

- add each teacher's log weight to its log-probabilities
- merge each weighted teacher distribution into the running fused distribution with `torch.logaddexp`
- this gives the fused teacher distribution in log-probability space
- return per-teacher scalar diagnostics and the fused log-probs for the loss

For `teacher_aggregation="confidence_weighted"`, the method first collects only selected-token log-probabilities for
each teacher. These small tensors are used to compute per-token teacher weights with:

```python
teacher_scores = 1 / (-selected_teacher_log_probs + 1e-8)
log_teacher_token_weights = F.log_softmax(teacher_scores, dim=0)
```

It then makes a second streaming pass over the teachers and fuses each full-vocabulary distribution with that teacher's
per-token dynamic weights. This keeps the implementation faithful to the thesis Eq. 14 while preserving the memory-safe
streaming aggregation design.

For `teacher_aggregation="max_margin"`, the method also first collects only selected-token teacher log-probabilities.
It additionally receives the student's selected-token log-probabilities from `compute_loss(...)` and computes:

```python
teacher_scores = (selected_teacher_log_probs.exp() - student_selected_log_probs.exp().unsqueeze(0)).abs()
selected_teacher_indices = teacher_scores.argmax(dim=0)
```

Those indices define a per-token winner-takes-all routing mask. The second streaming pass then applies a log weight of
`0` to the selected teacher and `-inf` to the others before fusing with `torch.logaddexp`. This implements
`pi_k = 1` for the largest-margin teacher and `0` for all other teachers while preserving the memory-safe aggregation
design.

This streaming aggregation is mathematically equivalent to stacking teachers and applying `logsumexp` over the teacher
axis, but it avoids materializing a `[num_teachers, batch_size, sequence_length, vocab_size]` tensor. This matters for
large-vocabulary multi-teacher runs where the stacked teacher tensor can dominate memory usage.

For `uniform` and `static_weighted`, this is the exact Phase 1 multi-teacher idea: static fusion in probability space.
For `confidence_weighted` and `max_margin`, this becomes adaptive fusion while keeping the same fused-teacher JSD
objective.

### `compute_loss(...)`

Code shape:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
```

This is the training loss for the student.

It first computes student outputs:

```python
student_outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
```

Then it slices to the generated region:

```python
prompt_lengths = inputs["prompts"].shape[1]
shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
shifted_student_log_probs = F.log_softmax(shifted_student_logits, dim=-1)
```

Then it gets the fused teacher target:

```python
shifted_labels = inputs["labels"][:, prompt_lengths:]
teacher_metrics, aggregated_teacher_log_probs = self._aggregate_teacher_log_probs(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    prompt_lengths=prompt_lengths,
    shifted_labels=shifted_labels,
)
self._log_distribution_metrics(teacher_metrics, aggregated_teacher_log_probs, shifted_labels)
```

Then it computes the final distillation loss:

```python
loss = self.generalized_jsd_loss_from_log_probs(
    student_log_probs=shifted_student_log_probs,
    teacher_log_probs=aggregated_teacher_log_probs,
    labels=shifted_labels,
    beta=self.beta,
)
```

So the complete logic is:

- student predicts distribution
- each teacher predicts distribution
- teachers are fused
- student is trained against the fused teacher target

### `generate_on_policy_outputs(...)`

Code shape:

```python
@staticmethod
def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
```

This is copied from GKD.

It is used for the student rollout branch and keeps the normal TRL/GKD on-policy path intact.

Main generation call:

```python
generated_outputs = model.generate(
    input_ids=inputs["prompts"],
    attention_mask=inputs.get("prompt_attention_mask", None),
    generation_config=generation_config,
    return_dict_in_generate=True,
)
```

Then it constructs:

- `generated_tokens`
- `new_attention_mask`
- `new_labels`

Padding tokens are turned into ignored labels (`-100`) for loss computation.

### `_generate_from_fused_teachers(...)`

Code shape:

```python
def _generate_from_fused_teachers(self, inputs):
```

This is the main multi-teacher generation path for `seq_kd`.

Unlike GKD, this does not generate from a single teacher model. Instead, it autoregressively generates from the fused teacher policy.

Start state:

```python
generated_tokens = inputs["prompts"]
new_attention_mask = inputs.get("prompt_attention_mask", None)
if new_attention_mask is None:
    new_attention_mask = torch.ones_like(generated_tokens)
```

This means generation starts from the prompt only.

Then generation settings are extracted from `self.generation_config`:

```python
max_new_tokens = self.generation_config.max_new_tokens
temperature = self.generation_config.temperature
do_sample = self.generation_config.do_sample
pad_token_id = self.generation_config.pad_token_id
eos_token_id = self.generation_config.eos_token_id
```

The method then loops token by token:

```python
for _ in range(max_new_tokens):
```

Inside each decoding step, every teacher scores the current prefix:

```python
for teacher_model in self.teacher_models:
    teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids=generated_tokens, attention_mask=new_attention_mask)

    teacher_logits = teacher_outputs.logits[:, -1, :] / temperature
    teacher_log_probs.append(F.log_softmax(teacher_logits, dim=-1))
```

Important detail:

- only the last-step distribution is used here
- this is decoding, so each step only needs the next-token distribution

Then the teachers are fused:

```python
fused_teacher_log_probs = torch.logsumexp(
    teacher_log_probs + teacher_weights.log().view(-1, 1, 1),
    dim=0,
)
```

Then the next token is chosen:

```python
if do_sample:
    next_tokens = torch.multinomial(fused_teacher_log_probs.exp(), num_samples=1).squeeze(1)
else:
    next_tokens = fused_teacher_log_probs.argmax(dim=-1)
```

So the generated sequence is sampled directly from the fused teacher distribution.

Then EOS and padding logic is handled:

```python
if eos_token_id is not None:
    next_finished = torch.isin(next_tokens, torch.tensor(eos_token_id, device=next_tokens.device))
...
if pad_token_id is not None:
    next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
```

Finally, the token is appended to the running sequence and the attention mask is extended.

This function is important because it makes Phase 1 genuinely multi-teacher even in the `seq_kd` branch.

### `training_step(...)`

Code shape:

```python
def training_step(self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None) -> torch.Tensor:
```

This controls where the rollout comes from before the loss is computed.

First branch:

```python
if self.seq_kd:
    new_input_ids, new_attention_mask, new_labels = self._generate_from_fused_teachers(inputs)
    inputs["input_ids"] = new_input_ids
    inputs["attention_mask"] = new_attention_mask
    inputs["labels"] = new_labels
```

Meaning:

- if sequence KD is enabled, training data comes from the fused teacher policy
- this is not a single-teacher shortcut

Second branch:

```python
if random.random() <= self.lmbda:
    with unwrap_model_for_generation(...):
        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(...)
```

Meaning:

- with probability `lmbda`, the student generates on-policy rollouts
- that keeps the original GKD on-policy idea alive

Final step:

```python
loss = super().training_step(model, inputs, num_items_in_batch)
return loss
```

This hands control back to the normal Trainer flow, which will eventually call `compute_loss(...)`.

## Logged diagnostics

The trainer logs distribution diagnostics through the standard `Trainer` logging path. This means the values appear in
`trainer.state.log_history` and are also sent to configured integrations such as Weights & Biases when `report_to` is
set accordingly.

Per-teacher metrics are logged with keys of the form `teachers/{idx}_{teacher_name}/...`. The index follows the order
of `teacher_model_names_or_paths`, while `teacher_name` is a sanitized version of the teacher model path or class name.
For example, the first teacher in `teacher_model_names_or_paths=["Qwen/Qwen2.5-Math-1.5B", ...]` would be logged under
`teachers/0_Qwen_Qwen2.5-Math-1.5B/...`.

- `teachers/{idx}_{teacher_name}/selected_logprob`: the mean log-probability assigned by that teacher to the actual generated or completion tokens.
- `teachers/{idx}_{teacher_name}/entropy`: the mean entropy of that teacher over the supervised tokens. Higher values indicate a flatter, less certain distribution.
- `teachers/{idx}_{teacher_name}/confidence`: the mean maximum token probability of that teacher over the supervised tokens. Higher values indicate sharper predictions.
- `teachers/{idx}_{teacher_name}/weight`: the normalized aggregation weight assigned to that teacher.

Fused-teacher metrics are logged with keys of the form `fused/...`:

- `fused/selected_logprob`: the mean log-probability assigned by the fused teacher distribution to the supervised tokens.
- `fused/entropy`: the mean entropy of the fused teacher distribution.
- `fused/confidence`: the mean maximum token probability of the fused teacher distribution.

These diagnostics are useful before introducing adaptive teacher routing. They show whether teachers agree, whether the
fused target is sharp or noisy, and whether the static weights are producing a meaningful supervision signal.

## Aggregation ablations

The current trainer supports two static aggregation modes and two adaptive aggregation modes:

- `teacher_aggregation="uniform"`: all teachers receive equal weight. This is the Phase 1 baseline and corresponds to a naive ensemble target.
- `teacher_aggregation="static_weighted"`: teachers receive fixed user-provided weights through `teacher_weights`.
- `teacher_aggregation="confidence_weighted"`: teachers receive token-level dynamic weights using the selected-token inverse-NLL confidence formula from the thesis document.
- `teacher_aggregation="max_margin"`: each token is routed to the teacher with the largest selected-token probability gap from the student.

These settings should be treated as ablations. `uniform` answers whether simple multi-teacher fusion helps at all,
`static_weighted` tests whether prior knowledge about teacher quality or domain relevance improves the fused target,
`confidence_weighted` tests whether token-level expert confidence improves over static fusion, and `max_margin` tests
whether routing toward the teacher with the largest teacher-student gap provides a stronger corrective signal.

## End-to-end flow

The complete Phase 1 training logic is:

1. build trainer/config
2. load multiple teachers
3. normalize teacher weights
4. choose rollout source:
   - fused teacher if `seq_kd=True`
   - student on-policy rollout with probability `lmbda`
   - otherwise use the collated input
5. run the student on the final sequence
6. run all teachers on the same sequence
7. fuse teacher distributions in probability space
8. compute generalized JSD between student and fused teacher

## Important current limitations

This document describes the current code, which still has some intentional limitations:

- `confidence_weighted` and `max_margin` adaptive routing are implemented; `min_ce`, `avg_ce`, and `domain_routed` are still pending
- no heterogeneous tokenizer support yet
- fused-teacher generation currently recomputes each teacher on the full prefix each decoding step
- `_paper` metadata in the trainer is still incomplete

## Suggested reading path

If you want to understand the code quickly, read it in this order:

1. `trl/experimental/multi_teacher_gkd/multi_teacher_gkd_config.py`
2. `trl/experimental/multi_teacher_gkd/__init__.py`
3. `MultiTeacherGKDTrainer.__init__`
4. `_load_teacher_models`
5. `_prepare_teacher_models`
6. `_aggregate_teacher_log_probs`
7. `compute_loss`
8. `_generate_from_fused_teachers`
9. `training_step`

That order mirrors the actual runtime flow of the trainer.
