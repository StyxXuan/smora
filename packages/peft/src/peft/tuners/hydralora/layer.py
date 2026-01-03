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
from typing import Any, List, Optional, Union, Tuple
import torch.nn.functional as F
import math
import packaging
import torch
import transformers
from torch import nn
from torch.autograd import Variable  

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
import time

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config


class HydraLoRALayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "expert_num", "routing_strategy")

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.expert_num = {}
        self.routing_strategy = {}
        self.lora_router = nn.ParameterDict({})
        
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()


    def update_layer(self, adapter_name, r, expert_num, routing_strategy, lora_alpha, lora_dropout, init_lora_weights):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.routing_strategy[adapter_name] = routing_strategy
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.expert_num[adapter_name] = expert_num

        # Actual trainable parameters
        # Right singular vectors

        self.lora_A[adapter_name] = nn.Parameter(
            torch.empty(
                self.in_features,  # 输入维度
                r,                 # 低秩维度
            )
        )

        self.lora_B[adapter_name] = nn.Parameter(
            torch.empty(
                self.expert_num[adapter_name],
                r,
                self.out_features,
            )
        )

        # The current rank
        self.lora_router[adapter_name] = nn.Linear(self.in_features, expert_num, bias=False)
        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            d, r = self.lora_A[adapter_name].shape
            param = torch.empty((r, d))
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            self.lora_A[adapter_name].data = param.T.contiguous()

            # 初始化 lora_B
            nn.init.zeros_(self.lora_B[adapter_name])

class HydraLoRALinear(nn.Module, HydraLoRALayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        expert_num: int = 0,
        routing_strategy: str = "soft",
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        HydraLoRALayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, expert_num, routing_strategy, lora_alpha, lora_dropout, init_lora_weights)
        self.routing_strategy = routing_strategy

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            B, S, in_feat = x.shape
            N = B * S

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                
                lora_A = self.lora_A[active_adapter]            # [in_features, r]
                
                lora_B = self.lora_B[active_adapter]            # [expert_num, r, out_features]
                dtype = lora_A.dtype
                lora_router = self.lora_router[active_adapter]  # Linear(in_features -> expert_num)
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                expert_num = self.expert_num[active_adapter]
                r = self.r[active_adapter]

                # 1. 对输入做 drop out
                #    x_dropped: [B, S, in_features]
                x_dropped = dropout(x)
                
                x_dropped = x_dropped.to(dtype)
                # 2. 路由器打分: [B, S, expert_num]
                logits = lora_router(x_dropped)
                
                # 3. 将 lora_A 复制 expert_num 次，使其形状与 lora_B 匹配
                lora_A_expanded = lora_A.unsqueeze(0).expand(expert_num, -1, -1)  # [expert_num, in_features, r]
                
                gating_probs = F.softmax(logits, dim=-1).to(dtype)  # [B, S, expert_num]
                mid = torch.einsum("bsd,edr->bser", x_dropped, lora_A_expanded)
                mid = torch.einsum("bser,erd->bsed", mid, lora_B)
                res = torch.einsum("bsed,bse->bsd", mid, gating_probs)
                # 加到 result
                result = result + res * scaling

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hydralora." + rep