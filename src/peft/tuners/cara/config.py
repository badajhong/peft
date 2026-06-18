# Copyright 2025-present the HuggingFace Inc. team.
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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class CaraConfig(PeftConfig):
    """Configuration class for CaRA (Cayley Rotational Adaptation)."""
    r: int = field(default=8, metadata={"help": "CaRA rank dimension"})
    noise_alpha: float = field(default=0.01, metadata={"help": "Transparent noise scaling factor"})
    noise_step_interval: int = field(default=5, metadata={"help": "Inject noise every N forward steps"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression to replace with CaRA."}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of extra modules to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    
    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CARA