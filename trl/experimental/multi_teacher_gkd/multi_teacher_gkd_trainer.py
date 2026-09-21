# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import textwrap
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ...data_utils import is_conversational
from ...models import prepare_deepspeed
from ...models.utils import unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import disable_dropout_in_model
from ..utils import DataCollatorForChatML, empty_cache
from .multi_teacher_gkd_config import MultiTeacherGKDConfig


if is_peft_available():
    from peft import PeftConfig


logger = logging.getLogger(__name__)


class MultiTeacherGKDTrainer(SFTTrainer):
    """Trainer for multi-teacher Generalized Knowledge Distillation (GKD) of language models.

    For details on GKD, see the paper: [On-Policy Distillation of Language Models: Learning from Self-Generated
    Mistakes](https://huggingface.co/papers/2306.13649).

    Args:
        model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Model to be trained, or the string identifier of the model to be instantiated from a pretrained model.
        teacher_models (`list` of [`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Teacher models for knowledge distillation, or the string identifiers of the models to be instantiated from
            pretrained models. If `None`, `args.teacher_model_names_or_paths` is used. At least one teacher must be
            provided through this argument or through `args.teacher_model_names_or_paths`.
        args ([`experimental.multi_teacher_gkd.MultiTeacherGKDConfig`], *optional*):
            Training arguments.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Data collator to batch samples from the dataset. It defaults to a
            [`experimental.utils.DataCollatorForChatML`] using the `processing_class`.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
           Class to process the data.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics at evaluation. Must take in an [`~transformers.EvalPrediction`] and return a
            dictionary string to float.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess the logits before computing the metrics. Must take in the `logits` and `labels` and
            return the logits to be used for metrics computation.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the `model` will be
            wrapped with the specified PEFT adapter.
        formatting_func (`Callable`, *optional*):
            Function to format the dataset. Must take in an example and return an example.
    """

    _tag_names = ["trl", "gkd", "multi-teacher-gkd"]
    _name = "MultiTeacherGKD"
    _paper = {
        "title": "",
        "id": "",
        # docstyle-ignore
        "citation": textwrap.dedent(""""""),
    }

    @staticmethod
    def _prepare_prompt_completion_dataset(
        dataset: Dataset | IterableDataset | None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None,
        args: MultiTeacherGKDConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset | None:
        if dataset is None:
            return None

        first_example = next(iter(dataset))
        if "input_ids" in first_example or "prompt" not in first_example or "completion" not in first_example:
            return dataset

        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Tokenizing {dataset_name} prompt-completion dataset"
            map_kwargs["num_proc"] = args.dataset_num_proc

        def tokenize_prompt_completion(example, processing_class, max_length):
            if is_conversational(example):
                prompt_ids = processing_class.apply_chat_template(
                    example["prompt"],
                    add_generation_prompt=True,
                    return_dict=False,
                    **example.get("chat_template_kwargs", {}),
                )
                prompt_completion_ids = processing_class.apply_chat_template(
                    example["prompt"] + example["completion"],
                    add_generation_prompt=False,
                    return_dict=False,
                    **example.get("chat_template_kwargs", {}),
                )
                if prompt_completion_ids[: len(prompt_ids)] != prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )
                completion_ids = prompt_completion_ids[len(prompt_ids) :]
            else:
                completion = example["completion"]
                if processing_class.eos_token is not None and not completion.endswith(processing_class.eos_token):
                    completion = completion + processing_class.eos_token
                prompt_ids = processing_class(text=example["prompt"], add_special_tokens=False).input_ids
                completion_ids = processing_class(text=completion, add_special_tokens=False).input_ids

            if max_length is not None and len(prompt_ids) + len(completion_ids) > max_length:
                if completion_ids and max_length > 1:
                    max_prompt_tokens = min(len(prompt_ids), max_length - 1)
                    prompt_ids = prompt_ids[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
                    completion_ids = completion_ids[: max_length - len(prompt_ids)]
                else:
                    prompt_ids = prompt_ids[-max_length:]
                    completion_ids = []

            input_ids = prompt_ids + completion_ids
            tokenized = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "prompts": prompt_ids,
            }
            if "domain" in example:
                tokenized["domain"] = example["domain"]
            return tokenized

        with PartialState().main_process_first():
            return dataset.map(
                tokenize_prompt_completion,
                fn_kwargs={"processing_class": processing_class, "max_length": args.max_length},
                **map_kwargs,
            )

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_models: list[PreTrainedModel | nn.Module | str] | None = None,
        args: MultiTeacherGKDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable | None = None,
    ):
        if teacher_models is None:
            teacher_models = args.teacher_model_names_or_paths
        if teacher_models is None or len(teacher_models) == 0:
            raise ValueError("teacher_models must contain at least one teacher model.")
        if args.teacher_weights is not None and len(args.teacher_weights) != len(teacher_models):
            raise ValueError("teacher_weights must have the same length as teacher_models.")
        self.teacher_metric_names = self._get_teacher_metric_names(teacher_models)

        # Ensure Trainer does not drop non-signature columns used by the collator (e.g., "prompts")
        args.remove_unused_columns = False
        # Respect a user-provided data_collator; otherwise, provide a ChatML collator.
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Ensure SFTTrainer does not pre-process the dataset when using this collator, so that raw conversational
        # fields remain available. Raw prompt-completion datasets are prepared below before SFTTrainer initializes.
        if args.dataset_kwargs is None:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
        else:
            args.dataset_kwargs["skip_prepare_dataset"] = True

        train_dataset = self._prepare_prompt_completion_dataset(train_dataset, processing_class, args, "train")
        if isinstance(eval_dataset, dict):
            eval_dataset = {
                key: self._prepare_prompt_completion_dataset(dataset, processing_class, args, key)
                for key, dataset in eval_dataset.items()
            }
        else:
            eval_dataset = self._prepare_prompt_completion_dataset(eval_dataset, processing_class, args, "eval")

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        self.teacher_aggregation = args.teacher_aggregation
        self._active_domain_teacher_idx = None
        self.teacher_models = self._prepare_teacher_models(teacher_models, args.teacher_model_init_kwargs)
        self.teacher_weights = self._normalize_teacher_weights(args.teacher_weights)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": True,
            "top_k": 0,
            "use_cache": True,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
        # Keep training-specific generation kwargs to overwrite model's original generation config
        self.generation_kwargs = generation_kwargs
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

    @staticmethod
    def _get_teacher_metric_names(teacher_models: list[PreTrainedModel | nn.Module | str]) -> list[str]:
        teacher_metric_names = []
        for teacher_idx, teacher_model in enumerate(teacher_models):
            if isinstance(teacher_model, str):
                teacher_name = teacher_model
            elif getattr(teacher_model, "name_or_path", None):
                teacher_name = teacher_model.name_or_path
            else:
                teacher_name = teacher_model.__class__.__name__
            teacher_name = "".join(char if char.isalnum() or char in ["-", "_", "."] else "_" for char in teacher_name)
            teacher_metric_names.append(f"{teacher_idx}_{teacher_name}")
        return teacher_metric_names

    def _load_teacher_models(
        self,
        teacher_models: list[PreTrainedModel | nn.Module | str],
        teacher_model_init_kwargs: dict[str, Any] | str | None,
    ) -> list[PreTrainedModel | nn.Module]:
        if teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif any(not isinstance(teacher_model, str) for teacher_model in teacher_models):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the MultiTeacherGKDConfig, but one of your teacher models "
                "is already instantiated."
            )
        else:
            teacher_model_init_kwargs = dict(teacher_model_init_kwargs)
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        loaded_teacher_models = []
        for teacher_model in teacher_models:
            if isinstance(teacher_model, str):
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)
            loaded_teacher_models.append(teacher_model)
        return loaded_teacher_models

    def _prepare_teacher_models(
        self,
        teacher_models: list[PreTrainedModel | nn.Module | str],
        teacher_model_init_kwargs: dict[str, Any] | str | None,
    ) -> list[PreTrainedModel | nn.Module]:
        loaded_teacher_models = self._load_teacher_models(teacher_models, teacher_model_init_kwargs)

        if self.teacher_aggregation == "domain_routed":
            for teacher_model in loaded_teacher_models:
                teacher_model.eval().requires_grad_(False).to("cpu")
            return loaded_teacher_models

        prepared_teacher_models = []
        for teacher_model in loaded_teacher_models:
            if self.is_deepspeed_enabled:
                teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            prepared_teacher_models.append(teacher_model)
        return prepared_teacher_models

    def _get_domain_teacher_indices(self, domains, batch_size, device):
        if domains is None:
            raise ValueError("domains must be provided when teacher_aggregation='domain_routed'.")
        if len(self.teacher_models) < 3:
            raise ValueError(
                "domain_routed requires at least 3 teacher models for instruct/general, math, and code domains."
            )
        if len(domains) != batch_size:
            raise ValueError("domains must have the same length as the input batch.")

        domain_to_teacher_idx = {"instruct": 0, "general": 0, "math": 1, "code": 2}
        for domain in domains:
            if domain not in domain_to_teacher_idx:
                raise ValueError(
                    f"Unknown domain '{domain}' for teacher_aggregation='domain_routed'. Expected one of "
                    f"{list(domain_to_teacher_idx)}."
                )
        return torch.tensor([domain_to_teacher_idx[domain] for domain in domains], device=device)

    def _activate_domain_teacher(self, teacher_idx, device):
        if self._active_domain_teacher_idx != teacher_idx:
            if self._active_domain_teacher_idx is not None:
                self.teacher_models[self._active_domain_teacher_idx].to("cpu")
            self.teacher_models[teacher_idx].to(device)
            self._active_domain_teacher_idx = teacher_idx
        return self.teacher_models[teacher_idx]

    def _normalize_teacher_weights(self, teacher_weights: list[float] | None) -> torch.Tensor:
        if self.teacher_aggregation != "static_weighted" or teacher_weights is None:
            normalized_teacher_weights = torch.ones(len(self.teacher_models), dtype=torch.float32)
        else:
            normalized_teacher_weights = torch.tensor(teacher_weights, dtype=torch.float32)
        return normalized_teacher_weights / normalized_teacher_weights.sum()

    @staticmethod
    def generalized_jsd_loss_from_log_probs(student_log_probs, teacher_log_probs, labels=None, beta=0.5):
        if labels is not None:
            mask = labels != -100
            num_tokens = mask.sum()
            if num_tokens == 0:
                return student_log_probs.sum() * 0.0

            valid_positions = mask.reshape(-1).nonzero().flatten()
            seq_length = mask.size(1)

            loss = student_log_probs.new_tensor(0.0)
            beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            chunk_size = 64
            for start in range(0, valid_positions.size(0), chunk_size):
                end = start + chunk_size
                chunk_positions = valid_positions[start:end]
                batch_positions = chunk_positions // seq_length
                token_positions = chunk_positions % seq_length
                student_log_probs_chunk = student_log_probs[batch_positions, token_positions]
                teacher_log_probs_chunk = teacher_log_probs[batch_positions, token_positions]

                if beta == 0:
                    loss = loss + F.kl_div(
                        student_log_probs_chunk, teacher_log_probs_chunk, reduction="sum", log_target=True
                    )
                elif beta == 1:
                    loss = loss + F.kl_div(
                        teacher_log_probs_chunk, student_log_probs_chunk, reduction="sum", log_target=True
                    )
                else:
                    # Keep full-vocab tensors bounded for long completions.
                    mixture_log_probs = torch.logaddexp(
                        student_log_probs_chunk + torch.log1p(-beta_tensor),
                        teacher_log_probs_chunk + torch.log(beta_tensor),
                    )
                    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs_chunk, reduction="sum", log_target=True)
                    kl_student = F.kl_div(mixture_log_probs, student_log_probs_chunk, reduction="sum", log_target=True)
                    loss = loss + beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student

            return loss / num_tokens

        # Compute log probabilities for student and probabilities for teacher
        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logaddexp(
                student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)
            )
            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)

    def _aggregate_teacher_log_probs(
        self,
        input_ids,
        attention_mask,
        prompt_lengths,
        shifted_labels=None,
        shifted_student_log_probs=None,
        domains=None,
    ):
        scored_adaptive_aggregations = ["confidence_weighted", "max_margin", "min_ce", "avg_ce"]
        label_required_aggregations = scored_adaptive_aggregations + ["domain_routed"]
        if self.teacher_aggregation in label_required_aggregations and shifted_labels is None:
            raise ValueError(
                f"shifted_labels must be provided when teacher_aggregation='{self.teacher_aggregation}'."
            )
        if self.teacher_aggregation == "max_margin" and shifted_student_log_probs is None:
            raise ValueError("shifted_student_log_probs must be provided when teacher_aggregation='max_margin'.")
        selected_teacher_indices = None
        if self.teacher_aggregation == "domain_routed":
            selected_teacher_indices = self._get_domain_teacher_indices(
                domains, input_ids.size(0), input_ids.device
            )
        teacher_metrics = []
        aggregated_teacher_log_probs = None
        teacher_weights = self.teacher_weights.to(device=input_ids.device)
        selected_teacher_log_probs = []

        valid_mask = None
        safe_labels = None
        if shifted_labels is not None:
            valid_mask = shifted_labels != -100
            safe_labels = shifted_labels.masked_fill(~valid_mask, 0)

        def masked_mean(values, mask=valid_mask):
            return values[mask].mean() if mask is not None and mask.any() else values.new_tensor(0.0)

        domain_teacher_token_weights = None
        if self.teacher_aggregation == "domain_routed":
            domain_teacher_token_weights = F.one_hot(
                selected_teacher_indices, num_classes=len(self.teacher_models)
            ).permute(1, 0).to(dtype=torch.float32)
            domain_teacher_token_weights = domain_teacher_token_weights.unsqueeze(-1).expand(
                -1, -1, shifted_labels.size(1)
            )

        if self.teacher_aggregation == "domain_routed":
            for teacher_idx, teacher_model in enumerate(self.teacher_models):
                rows = (selected_teacher_indices == teacher_idx).nonzero(as_tuple=True)[0]
                teacher_weight = masked_mean(domain_teacher_token_weights[teacher_idx]).item()
                if rows.numel() == 0:
                    teacher_metrics.append(
                        {"selected_logprob": 0.0, "entropy": 0.0, "confidence": 0.0, "weight": teacher_weight}
                    )
                    continue

                teacher_model = self._activate_domain_teacher(teacher_idx, input_ids.device)
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids[rows], attention_mask=attention_mask[rows]
                    )

                shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
                teacher_log_probs = F.log_softmax(shifted_teacher_logits, dim=-1)
                del teacher_outputs, shifted_teacher_logits

                teacher_probs = teacher_log_probs.exp()
                selected_log_probs = teacher_log_probs.gather(
                    -1, safe_labels[rows].unsqueeze(-1)
                ).squeeze(-1)
                entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
                confidence = teacher_probs.max(dim=-1).values
                routed_valid_mask = valid_mask[rows]

                teacher_metrics.append(
                    {
                        "selected_logprob": masked_mean(selected_log_probs, routed_valid_mask).item(),
                        "entropy": masked_mean(entropy, routed_valid_mask).item(),
                        "confidence": masked_mean(confidence, routed_valid_mask).item(),
                        "weight": teacher_weight,
                    }
                )
                if aggregated_teacher_log_probs is None:
                    aggregated_teacher_log_probs = teacher_log_probs.new_empty(
                        input_ids.size(0), teacher_log_probs.size(1), teacher_log_probs.size(2)
                    )
                aggregated_teacher_log_probs[rows] = teacher_log_probs
                del teacher_probs, selected_log_probs, entropy, confidence, teacher_log_probs

            return teacher_metrics, aggregated_teacher_log_probs

        for teacher_idx, teacher_model in enumerate(self.teacher_models):
            # compute teacher output in eval mode
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)

            # slice the logits for the generated tokens using the inputs["prompts"] lengths
            shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
            teacher_log_probs = F.log_softmax(shifted_teacher_logits, dim=-1)
            del teacher_outputs, shifted_teacher_logits

            if shifted_labels is not None:
                teacher_probs = teacher_log_probs.exp()
                selected_log_probs = teacher_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
                confidence = teacher_probs.max(dim=-1).values
                teacher_metrics.append(
                    {
                        "selected_logprob": masked_mean(selected_log_probs).item(),
                        "entropy": masked_mean(entropy).item(),
                        "confidence": masked_mean(confidence).item(),
                    }
                )
                if self.teacher_aggregation in ["static_weighted", "uniform"]:
                    teacher_metrics[-1]["weight"] = teacher_weights[teacher_idx].item()
                elif self.teacher_aggregation in scored_adaptive_aggregations:
                    selected_teacher_log_probs.append(selected_log_probs)
                elif self.teacher_aggregation == "domain_routed":
                    teacher_metrics[-1]["weight"] = masked_mean(domain_teacher_token_weights[teacher_idx]).item()
                del teacher_probs, selected_log_probs, entropy, confidence

            if self.teacher_aggregation not in scored_adaptive_aggregations:
                # Aggregate the teacher distributions in probability space.
                teacher_weight = teacher_weights[teacher_idx].to(dtype=teacher_log_probs.dtype).log()
                teacher_log_probs.add_(teacher_weight)
                if aggregated_teacher_log_probs is None:
                    aggregated_teacher_log_probs = teacher_log_probs
                else:
                    aggregated_teacher_log_probs = torch.logaddexp(aggregated_teacher_log_probs, teacher_log_probs)
            del teacher_log_probs

        if self.teacher_aggregation in scored_adaptive_aggregations:
            selected_teacher_log_probs = torch.stack(selected_teacher_log_probs, dim=0)
            if self.teacher_aggregation == "confidence_weighted":
                # Eq. (14) in the document: C_k = 1 / (-log P_Tk(y_t | x) + eps), then softmax over teachers.
                teacher_scores = 1 / (-selected_teacher_log_probs + 1e-8)
                log_teacher_token_weights = F.log_softmax(teacher_scores, dim=0)
                del teacher_scores
            elif self.teacher_aggregation == "max_margin":
                # Max-margin routing: select the teacher with the largest selected-token probability gap to the student.
                student_selected_log_probs = shifted_student_log_probs.gather(
                    -1, safe_labels.unsqueeze(-1)
                ).squeeze(-1)
                teacher_scores = (
                    selected_teacher_log_probs.exp() - student_selected_log_probs.exp().unsqueeze(0)
                ).abs()
                selected_teacher_indices = teacher_scores.argmax(dim=0)
                teacher_token_weights = F.one_hot(
                    selected_teacher_indices, num_classes=len(self.teacher_models)
                ).permute(2, 0, 1)
                log_teacher_token_weights = teacher_token_weights.to(dtype=selected_teacher_log_probs.dtype).log()
                del student_selected_log_probs, teacher_scores, selected_teacher_indices
            elif self.teacher_aggregation in ["min_ce", "avg_ce"]:
                valid_mask_f = valid_mask.to(dtype=selected_teacher_log_probs.dtype)
                valid_counts = valid_mask_f.sum(dim=-1).clamp(min=1)
                teacher_scores = -(selected_teacher_log_probs * valid_mask_f.unsqueeze(0)).sum(
                    dim=-1
                ) / valid_counts.unsqueeze(0)
                if self.teacher_aggregation == "min_ce":
                    selected_teacher_indices = teacher_scores.argmin(dim=0)
                    teacher_token_weights = F.one_hot(
                        selected_teacher_indices, num_classes=len(self.teacher_models)
                    ).permute(1, 0)
                    log_teacher_token_weights = teacher_token_weights.to(dtype=selected_teacher_log_probs.dtype).log()
                    del selected_teacher_indices
                else:
                    teacher_rewards = 1 / torch.exp(teacher_scores)
                    log_teacher_token_weights = F.log_softmax(teacher_rewards, dim=0)
                    del teacher_rewards
                log_teacher_token_weights = log_teacher_token_weights.unsqueeze(-1).expand_as(selected_teacher_log_probs)
                del valid_mask_f, valid_counts, teacher_scores
            teacher_token_weights = log_teacher_token_weights.exp()
            for teacher_idx, teacher_metrics_i in enumerate(teacher_metrics):
                teacher_metrics_i["weight"] = masked_mean(teacher_token_weights[teacher_idx]).item()
            del selected_teacher_log_probs, teacher_token_weights

            for teacher_idx, teacher_model in enumerate(self.teacher_models):
                # compute teacher output in eval mode
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)

                # slice the logits for the generated tokens using the inputs["prompts"] lengths
                shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
                teacher_log_probs = F.log_softmax(shifted_teacher_logits, dim=-1)
                del teacher_outputs, shifted_teacher_logits

                teacher_log_probs.add_(
                    log_teacher_token_weights[teacher_idx].unsqueeze(-1).to(teacher_log_probs.dtype)
                )
                if aggregated_teacher_log_probs is None:
                    aggregated_teacher_log_probs = teacher_log_probs
                else:
                    aggregated_teacher_log_probs = torch.logaddexp(aggregated_teacher_log_probs, teacher_log_probs)
                del teacher_log_probs
        return teacher_metrics, aggregated_teacher_log_probs

    def _log_distribution_metrics(self, teacher_metrics, fused_teacher_log_probs, shifted_labels):
        mode = "train" if self.model.training else "eval"
        valid_mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~valid_mask, 0)

        def masked_mean(values):
            return values[valid_mask].mean() if valid_mask.any() else values.new_tensor(0.0)

        for teacher_idx, teacher_metrics_i in enumerate(teacher_metrics):
            teacher_metric_name = self.teacher_metric_names[teacher_idx]
            if self.teacher_aggregation != "domain_routed" or teacher_metrics_i["weight"] > 0:
                self._metrics[mode][f"teachers/{teacher_metric_name}/selected_logprob"].append(
                    teacher_metrics_i["selected_logprob"]
                )
                self._metrics[mode][f"teachers/{teacher_metric_name}/entropy"].append(teacher_metrics_i["entropy"])
                self._metrics[mode][f"teachers/{teacher_metric_name}/confidence"].append(
                    teacher_metrics_i["confidence"]
                )
            self._metrics[mode][f"teachers/{teacher_metric_name}/weight"].append(teacher_metrics_i["weight"])

        fused_teacher_probs = fused_teacher_log_probs.exp()
        fused_selected_log_probs = fused_teacher_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        fused_entropy = -(fused_teacher_probs * fused_teacher_log_probs).sum(dim=-1)
        fused_confidence = fused_teacher_probs.max(dim=-1).values

        self._metrics[mode]["fused/selected_logprob"].append(masked_mean(fused_selected_log_probs).item())
        self._metrics[mode]["fused/entropy"].append(masked_mean(fused_entropy).item())
        self._metrics[mode]["fused/confidence"].append(masked_mean(fused_confidence).item())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :] / self.temperature
        shifted_student_log_probs = F.log_softmax(shifted_student_logits, dim=-1)
        del shifted_student_logits
        if not return_outputs:
            del student_outputs
        shifted_labels = inputs["labels"][:, prompt_lengths:]
        mode = "train" if self.model.training else "eval"
        num_target_tokens = (shifted_labels != -100).sum().detach().reshape(1)
        gathered_num_target_tokens = self.accelerator.gather(num_target_tokens)
        self._metrics[mode]["target_tokens"].append(gathered_num_target_tokens.float().mean().item())
        self._metrics[mode]["empty_target_batches"].append((gathered_num_target_tokens == 0).float().mean().item())
        teacher_metrics, aggregated_teacher_log_probs = self._aggregate_teacher_log_probs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            prompt_lengths=prompt_lengths,
            shifted_labels=shifted_labels,
            shifted_student_log_probs=shifted_student_log_probs,
            domains=inputs.get("domain"),
        )

        self._log_distribution_metrics(teacher_metrics, aggregated_teacher_log_probs, shifted_labels)

        # compute loss
        loss = self.generalized_jsd_loss_from_log_probs(
            student_log_probs=shifted_student_log_probs,
            teacher_log_probs=aggregated_teacher_log_probs,
            labels=shifted_labels,
            beta=self.beta,
        )

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def _generate_from_fused_teachers(self, inputs):
        # Generate output with respect to the prompt-only
        generated_tokens = inputs["prompts"]
        new_attention_mask = inputs.get("prompt_attention_mask", None)
        if new_attention_mask is None:
            new_attention_mask = torch.ones_like(generated_tokens)

        max_new_tokens = self.generation_config.max_new_tokens
        temperature = self.generation_config.temperature
        do_sample = self.generation_config.do_sample
        pad_token_id = self.generation_config.pad_token_id
        eos_token_id = self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        finished = torch.zeros(generated_tokens.size(0), dtype=torch.bool, device=generated_tokens.device)
        if self.teacher_aggregation == "domain_routed":
            selected_teacher_indices = self._get_domain_teacher_indices(
                inputs.get("domain"), generated_tokens.size(0), generated_tokens.device
            )

        for _ in range(max_new_tokens):
            if self.teacher_aggregation == "domain_routed":
                fused_teacher_log_probs = None
                for teacher_idx in selected_teacher_indices.unique().tolist():
                    rows = (selected_teacher_indices == teacher_idx).nonzero(as_tuple=True)[0]
                    teacher_model = self._activate_domain_teacher(teacher_idx, generated_tokens.device)
                    teacher_model.eval()
                    with torch.no_grad():
                        teacher_outputs = teacher_model(
                            input_ids=generated_tokens[rows], attention_mask=new_attention_mask[rows]
                        )

                    teacher_logits = teacher_outputs.logits[:, -1, :] / temperature
                    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                    if fused_teacher_log_probs is None:
                        fused_teacher_log_probs = teacher_log_probs.new_empty(
                            generated_tokens.size(0), teacher_log_probs.size(-1)
                        )
                    fused_teacher_log_probs[rows] = teacher_log_probs
                    del teacher_outputs, teacher_logits, teacher_log_probs
            else:
                teacher_log_probs = []
                for teacher_model in self.teacher_models:
                    # compute teacher output in eval mode
                    teacher_model.eval()
                    with torch.no_grad():
                        teacher_outputs = teacher_model(input_ids=generated_tokens, attention_mask=new_attention_mask)

                    teacher_logits = teacher_outputs.logits[:, -1, :] / temperature
                    teacher_log_probs.append(F.log_softmax(teacher_logits, dim=-1))

                # Aggregate the teacher distributions in probability space.
                teacher_log_probs = torch.stack(teacher_log_probs, dim=0)
                teacher_weights = self.teacher_weights.to(
                    device=teacher_log_probs.device, dtype=teacher_log_probs.dtype
                )
                fused_teacher_log_probs = torch.logsumexp(
                    teacher_log_probs + teacher_weights.log().view(-1, 1, 1),
                    dim=0,
                )

            if do_sample:
                next_tokens = torch.multinomial(fused_teacher_log_probs.exp(), num_samples=1).squeeze(1)
            else:
                next_tokens = fused_teacher_log_probs.argmax(dim=-1)

            if eos_token_id is not None:
                next_finished = torch.isin(next_tokens, torch.tensor(eos_token_id, device=next_tokens.device))
            else:
                next_finished = torch.zeros_like(finished)

            if pad_token_id is not None:
                next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)

            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(1)], dim=1)
            next_attention_mask = torch.where(finished, torch.zeros_like(next_tokens), torch.ones_like(next_tokens))
            new_attention_mask = torch.cat([new_attention_mask, next_attention_mask.unsqueeze(1)], dim=1)

            finished = finished | next_finished
            if finished.all():
                break

        # Calculate new attention mask
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[generated_tokens == pad_token_id] = -100

        return generated_tokens, new_attention_mask, new_labels

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.
        """
        if self.seq_kd:
            new_input_ids, new_attention_mask, new_labels = self._generate_from_fused_teachers(inputs)
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        if random.random() <= self.lmbda:
            with (
                unwrap_model_for_generation(
                    model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
