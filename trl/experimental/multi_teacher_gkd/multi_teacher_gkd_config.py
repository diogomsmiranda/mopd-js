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

from dataclasses import dataclass, field

from ..gkd.gkd_config import GKDConfig


@dataclass
class MultiTeacherGKDConfig(GKDConfig):
    """
    Configuration class for [`experimental.multi_teacher_gkd.MultiTeacherGKDTrainer`].

    This class includes only the parameters that are specific to multi-teacher GKD training. For a full list of
    training arguments, please refer to the [`~transformers.TrainingArguments`] and [`experimental.gkd.GKDConfig`]
    documentation.

    Args:
        teacher_model_names_or_paths (`list[str]`, *optional*):
            Model names or paths of the teacher models. Optional when teacher models are passed directly to
            [`experimental.multi_teacher_gkd.MultiTeacherGKDTrainer`].
        teacher_weights (`list[float]`, *optional*):
            Static weights used to aggregate the teacher distributions. If `None`, uniform weights are used.
        teacher_aggregation (`str`, *optional*, defaults to `"uniform"`):
            Strategy used to aggregate teacher distributions. Supported values are `"uniform"`, `"static_weighted"`,
            and `"confidence_weighted"`.
    """

    teacher_model_names_or_paths: list[str] | None = field(
        default=None,
        metadata={"help": "Model names or paths of the teacher models."},
    )
    teacher_weights: list[float] | None = field(
        default=None,
        metadata={"help": "Static weights used to aggregate the teacher distributions."},
    )
    teacher_aggregation: str = field(
        default="uniform",
        metadata={
            "help": "Strategy used to aggregate teacher distributions. Supported values are 'uniform', "
            "'static_weighted', and 'confidence_weighted'."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.teacher_model_names_or_paths is not None and len(self.teacher_model_names_or_paths) == 0:
            raise ValueError("teacher_model_names_or_paths must contain at least one teacher model when provided.")

        if self.teacher_aggregation not in ["uniform", "static_weighted", "confidence_weighted"]:
            raise ValueError("teacher_aggregation must be one of ['uniform', 'static_weighted', 'confidence_weighted'].")

        if self.teacher_weights is not None:
            if self.teacher_model_names_or_paths is not None and len(self.teacher_weights) != len(
                self.teacher_model_names_or_paths
            ):
                raise ValueError("teacher_weights must have the same length as teacher_model_names_or_paths.")
            if sum(self.teacher_weights) <= 0:
                raise ValueError("teacher_weights must have a strictly positive sum.")
