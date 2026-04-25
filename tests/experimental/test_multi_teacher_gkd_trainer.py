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

import pytest
import torch
import torch.nn.functional as F
from datasets import load_dataset
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.experimental.multi_teacher_gkd import MultiTeacherGKDConfig, MultiTeacherGKDTrainer

from ..testing_utils import TrlTestCase


class TestMultiTeacherGKDTrainerGenerateOnPolicy(TrlTestCase):
    @classmethod
    def setup_class(cls):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32").to(cls.device)
        cls.generation_config = GenerationConfig(
            max_new_tokens=20,
            num_return_sequences=1,
            pad_token_id=cls.tokenizer.pad_token_id,
            eos_token_id=cls.tokenizer.eos_token_id,
        )

    def test_generate_on_policy_outputs_deterministic(self):
        prompts = ["Hello, how are you?", "What's the weather like today?"]
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)

        inputs = {
            "prompts": tokenized_prompts["input_ids"].to(self.device),
            "prompt_attention_mask": tokenized_prompts["attention_mask"].to(self.device),
        }

        # Set temperature to 0 for deterministic output
        deterministic_generation_config = GenerationConfig(
            max_new_tokens=30,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
        )

        outputs = MultiTeacherGKDTrainer.generate_on_policy_outputs(
            self.model, inputs, deterministic_generation_config, self.tokenizer.pad_token_id
        )

        new_input_ids, new_attention_mask, new_labels = outputs

        # Decode the generated outputs
        generated_texts = self.tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)

        # Check if the generated texts start with the original prompts
        for prompt, generated_text in zip(prompts, generated_texts, strict=True):
            assert generated_text.startswith(prompt), (
                f"Generated text '{generated_text}' does not start with prompt '{prompt}'"
            )

        # Run the generation twice and check if the outputs are identical
        outputs2 = MultiTeacherGKDTrainer.generate_on_policy_outputs(
            self.model, inputs, deterministic_generation_config, self.tokenizer.pad_token_id
        )

        new_input_ids2, new_attention_mask2, new_labels2 = outputs2

        # Check if the two generations are identical
        assert torch.all(new_input_ids.eq(new_input_ids2)), "Deterministic generations are not identical"
        assert torch.all(new_attention_mask.eq(new_attention_mask2)), (
            "Attention masks for deterministic generations are not identical"
        )
        assert torch.all(new_labels.eq(new_labels2)), "Labels for deterministic generations are not identical"

    def test_generate_on_policy_outputs(self):
        prompts = ["Hello, how are you?", "What's the weather like today?"]
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)

        inputs = {
            "prompts": tokenized_prompts["input_ids"].to(self.device),
            "attention_mask": tokenized_prompts["attention_mask"].to(self.device),
        }

        outputs = MultiTeacherGKDTrainer.generate_on_policy_outputs(
            self.model, inputs, self.generation_config, self.tokenizer.pad_token_id
        )

        # Check that outputs is a tuple of three tensors
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3

        new_input_ids, new_attention_mask, new_labels = outputs

        # Check shapes
        batch_size = len(prompts)
        assert new_input_ids.shape[0] == batch_size
        assert new_attention_mask.shape[0] == batch_size
        assert new_labels.shape[0] == batch_size

        # Check types
        assert isinstance(new_input_ids, torch.Tensor)
        assert isinstance(new_attention_mask, torch.Tensor)
        assert isinstance(new_labels, torch.Tensor)

        # Check that new_input_ids and new_attention_mask have the same shape
        assert new_input_ids.shape == new_attention_mask.shape
        assert new_labels.shape == new_attention_mask.shape


class TestMultiTeacherGeneralizedJSDLoss(TrlTestCase):
    def setup_method(self):
        self.batch_size = 2
        self.seq_length = 3
        self.vocab_size = 5
        self.student_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        self.teacher_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        self.student_log_probs = F.log_softmax(self.student_logits, dim=-1)
        self.teacher_log_probs = F.log_softmax(self.teacher_logits, dim=-1)

    def test_generalized_jsd_loss_edge_cases(self):
        # Setup
        student_log_probs = torch.log(torch.tensor([[0.1, 0.9]])).unsqueeze(0)
        teacher_log_probs = torch.log(torch.tensor([[0.9, 0.1]])).unsqueeze(0)

        # Case 1: beta = 1 (should be equivalent to KL(student || teacher))
        loss_beta_1 = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            student_log_probs, teacher_log_probs, beta=1
        )
        expected_loss_beta_1 = F.kl_div(teacher_log_probs, student_log_probs, reduction="batchmean", log_target=True)
        assert round(abs(loss_beta_1.item() - expected_loss_beta_1.item()), 5) == 0

        # Case 2: beta = 0 (should be equivalent to KL(teacher || student))
        loss_beta_0 = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            student_log_probs, teacher_log_probs, beta=0
        )
        expected_loss_beta_0 = F.kl_div(student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True)
        assert round(abs(loss_beta_0.item() - expected_loss_beta_0.item()), 5) == 0

    def test_output_shape(self):
        loss = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.student_log_probs, self.teacher_log_probs
        )
        assert torch.is_tensor(loss)
        assert loss.shape == torch.Size([])

    def test_beta_values(self):
        loss_beta_0 = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.student_log_probs, self.teacher_log_probs, beta=0
        )
        loss_beta_1 = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.student_log_probs, self.teacher_log_probs, beta=1
        )
        assert loss_beta_0 != loss_beta_1

    def test_symmetry(self):
        student_teacher = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.student_log_probs, self.teacher_log_probs, beta=0.1
        )
        teacher_student = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.teacher_log_probs, self.student_log_probs, beta=0.1
        )
        assert student_teacher != teacher_student

        student_teacher = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.student_log_probs, self.teacher_log_probs, beta=0.5
        )
        teacher_student = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.teacher_log_probs, self.student_log_probs, beta=0.5
        )
        assert student_teacher == teacher_student

    def test_masking(self):
        labels = torch.tensor([[0, -100, 1], [1, 0, -100]])
        loss = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(
            self.student_log_probs,
            self.teacher_log_probs,
            labels=labels,
        )
        assert torch.is_tensor(loss)
        assert loss.shape == torch.Size([])

    def test_zero_loss_for_identical_inputs(self):
        identical_log_probs = F.log_softmax(torch.randn(self.batch_size, self.seq_length, self.vocab_size), dim=-1)
        loss = MultiTeacherGKDTrainer.generalized_jsd_loss_from_log_probs(identical_log_probs, identical_log_probs)
        assert round(abs(loss.item() - 0), 6) == 0


class TestMultiTeacherGKDConfig(TrlTestCase):
    def test_teacher_weight_validation(self):
        model_ids = [
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        ]

        with pytest.raises(ValueError, match="teacher_weights must have the same length"):
            MultiTeacherGKDConfig(
                output_dir=self.tmp_dir,
                bf16=False,
                teacher_model_names_or_paths=model_ids,
                teacher_weights=[1.0],
            )

        with pytest.raises(ValueError, match="teacher_weights must have a strictly positive sum"):
            MultiTeacherGKDConfig(
                output_dir=self.tmp_dir,
                bf16=False,
                teacher_model_names_or_paths=model_ids,
                teacher_weights=[0.0, 0.0],
            )

        with pytest.raises(ValueError, match="teacher_aggregation must be one of"):
            MultiTeacherGKDConfig(
                output_dir=self.tmp_dir,
                bf16=False,
                teacher_model_names_or_paths=model_ids,
                teacher_aggregation="unsupported",
            )


class TestMultiTeacherGKDTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_inputs(self):
        prompts = self.tokenizer(["Hello"], return_tensors="pt", padding=True)
        completions = self.tokenizer([" world"], return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = torch.cat([prompts["input_ids"], completions["input_ids"]], dim=1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, : prompts["input_ids"].shape[1]] = -100
        return {
            "prompts": prompts["input_ids"],
            "prompt_attention_mask": prompts["attention_mask"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _dummy_train_dataset(self):
        return load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

    def test_multi_teacher_gkd_trainer(self):
        training_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            # Using the same tiny model twice to exercise the multi-teacher code path cheaply.
            teacher_model_names_or_paths=[self.model_id, self.model_id],
            dataloader_drop_last=True,
            eval_strategy="steps",
            max_steps=2,
            eval_steps=1,
            save_steps=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

        trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert any("eval_loss" in log for log in trainer.state.log_history)

        # Check the params have changed
        assert any(
            not torch.allclose(param, trainer.model.get_parameter(n)) for n, param in previous_trainable_params.items()
        )

    def test_single_teacher_parity_with_gkd(self):
        gkd_args = GKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            report_to="none",
        )
        multi_teacher_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id],
            report_to="none",
        )
        inputs = self._prepare_inputs()

        gkd_trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=gkd_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )
        multi_teacher_trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=multi_teacher_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )

        gkd_trainer.model.eval()
        multi_teacher_trainer.model.eval()

        with torch.no_grad():
            gkd_loss = gkd_trainer.compute_loss(gkd_trainer.model, inputs)
            multi_teacher_loss = multi_teacher_trainer.compute_loss(multi_teacher_trainer.model, inputs)

        torch.testing.assert_close(multi_teacher_loss, gkd_loss, atol=1e-4, rtol=1e-4)

    def test_identical_teachers_equal_single_teacher(self):
        single_teacher_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id],
            report_to="none",
        )
        identical_teacher_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id, self.model_id],
            report_to="none",
        )
        inputs = self._prepare_inputs()

        single_teacher_trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=single_teacher_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )
        identical_teacher_trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=identical_teacher_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )

        single_teacher_trainer.model.eval()
        identical_teacher_trainer.model.eval()

        with torch.no_grad():
            single_teacher_loss = single_teacher_trainer.compute_loss(single_teacher_trainer.model, inputs)
            identical_teacher_loss = identical_teacher_trainer.compute_loss(identical_teacher_trainer.model, inputs)

        torch.testing.assert_close(identical_teacher_loss, single_teacher_loss)

    def test_fused_teacher_seq_kd_generation(self):
        training_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id, self.model_id],
            max_new_tokens=3,
            seq_kd=True,
            report_to="none",
        )
        trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=training_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )
        inputs = self._prepare_inputs()

        new_input_ids, new_attention_mask, new_labels = trainer._generate_from_fused_teachers(inputs)

        assert new_input_ids.shape[0] == inputs["prompts"].shape[0]
        assert new_input_ids.shape[1] > inputs["prompts"].shape[1]
        assert new_input_ids.shape == new_attention_mask.shape
        assert new_labels.shape == new_attention_mask.shape
        assert torch.all(new_input_ids[:, : inputs["prompts"].shape[1]].eq(inputs["prompts"]))

    def test_generate_from_fused_teachers_identical_teachers_matches_single_teacher(self):
        single_teacher_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id],
            max_new_tokens=3,
            report_to="none",
        )
        identical_teacher_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id, self.model_id],
            max_new_tokens=3,
            report_to="none",
        )
        single_teacher_trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=single_teacher_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )
        identical_teacher_trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=identical_teacher_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )
        single_teacher_trainer.generation_config.do_sample = False
        identical_teacher_trainer.generation_config.do_sample = False
        inputs = self._prepare_inputs()

        single_teacher_outputs = single_teacher_trainer._generate_from_fused_teachers(inputs)
        identical_teacher_outputs = identical_teacher_trainer._generate_from_fused_teachers(inputs)

        for single_teacher_output, identical_teacher_output in zip(
            single_teacher_outputs, identical_teacher_outputs, strict=True
        ):
            assert torch.all(single_teacher_output.eq(identical_teacher_output))

    def test_aggregate_teacher_log_probs_uniform(self):
        single_teacher_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id],
            report_to="none",
        )
        identical_teacher_args = MultiTeacherGKDConfig(
            output_dir=self.tmp_dir,
            bf16=False,
            teacher_model_names_or_paths=[self.model_id, self.model_id],
            report_to="none",
        )
        inputs = self._prepare_inputs()
        prompt_lengths = inputs["prompts"].shape[1]

        single_teacher_trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=single_teacher_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )
        identical_teacher_trainer = MultiTeacherGKDTrainer(
            model=self.model_id,
            args=identical_teacher_args,
            train_dataset=self._dummy_train_dataset(),
            processing_class=self.tokenizer,
        )

        single_teacher_log_probs = single_teacher_trainer._aggregate_teacher_log_probs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            prompt_lengths=prompt_lengths,
        )
        identical_teacher_log_probs = identical_teacher_trainer._aggregate_teacher_log_probs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            prompt_lengths=prompt_lengths,
        )

        torch.testing.assert_close(identical_teacher_log_probs, single_teacher_log_probs)

    def test_aggregate_teacher_log_probs_static_weighted(self):
        class DummyTeacherModel:
            def __init__(self, logits):
                self.logits = logits

            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.logits)

        teacher_1_probs = torch.tensor([0.8, 0.1, 0.1])
        teacher_2_probs = torch.tensor([0.1, 0.8, 0.1])
        teacher_1_logits = torch.log(teacher_1_probs).view(1, 1, 3).repeat(1, 2, 1)
        teacher_2_logits = torch.log(teacher_2_probs).view(1, 1, 3).repeat(1, 2, 1)
        teacher_weights = torch.tensor([0.25, 0.75])

        trainer = MultiTeacherGKDTrainer.__new__(MultiTeacherGKDTrainer)
        trainer.teacher_models = [DummyTeacherModel(teacher_1_logits), DummyTeacherModel(teacher_2_logits)]
        trainer.teacher_weights = teacher_weights
        trainer.temperature = 1.0

        aggregated_teacher_log_probs = trainer._aggregate_teacher_log_probs(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            prompt_lengths=1,
        )
        expected_teacher_probs = teacher_weights[0] * teacher_1_probs + teacher_weights[1] * teacher_2_probs
        expected_teacher_log_probs = torch.log(expected_teacher_probs).view(1, 1, 3)

        torch.testing.assert_close(aggregated_teacher_log_probs, expected_teacher_log_probs)
