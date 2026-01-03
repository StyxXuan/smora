# Copyright 2023-present the HuggingFace Inc. team.
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

import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_SRMOLE_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)
from peft.utils.integrations import gather_params_ctx

# from .gptq import SRMoLEQuantLinear  # Removed GPTQ support
from .layer import SRMoLELayer, SRMoLELinear


class SRMoLEModel(LoraModel):
    """
    Creates SRMoLE (Sparse Rank Mixture of Low-rank Experts) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`SRMoLEConfig`]): The configuration of the SRMoLE model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The SRMoLE model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM >>> from peft import LoraConfig, SRMoLEModel, SRMoLEConfig
        >>> config = SRMoLEConfig(
                peft_type="SRMOLE", task_type="SEQ_2_SEQ_LM", init_r=12, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = SRMoLEModel(model, config, "default")

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SRMoLEConfig`]): The configuration of the SRMoLE model.
    """

    # Note: don't redefine prefix here, it should be inherited from LoraModel

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key, 
    ):  
        # print(current_key)
        # import re
        # layer_idx = None
        # match = re.search(r'layers\.(\d+)\.', current_key)
        # if match:
        #     layer_idx = int(match.group(1))
            
        layer_idx = current_key
        
        kwargs = {
            "r": lora_config.r,
            "activate_r": lora_config.activate_r,
            "epsilon_greedy": lora_config.epsilon_greedy,
            "rank_partition": lora_config.rank_partition,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "layer_idx": layer_idx,
        }
        if kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]:
            raise ValueError(
                "SRMoLE does not support 8-bit or 4-bit quantization. "
                "Please use standard Linear layers without quantization."
            )

        # Quantization support removed for SRMoLE

        # If it is not an SRMoLELayer, create a new module, else update it with new adapters
        if not isinstance(target, SRMoLELayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.activate_r,
                lora_config.epsilon_greedy,
                lora_config.rank_partition,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        # Only support standard Linear layers, no quantization support
        if loaded_in_8bit or loaded_in_4bit:
            raise ValueError(
                "SRMoLE does not support 8-bit or 4-bit quantization. "
                "Please use standard Linear layers."
            )

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_module = SRMoLELinear(target, adapter_name, **kwargs)

        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_SRMOLE_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_SRMOLE_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)