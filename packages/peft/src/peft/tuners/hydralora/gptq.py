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
import torch
from torch import nn
import torch.nn.functional as F
from .layer import HydraLoRALayer


class HydraLoRAQuantLinear(torch.nn.Module, HydraLoRALayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        expert_num: int = 0,
        routing_strategy: str = "top-1",
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
        self.update_layer(adapter_name, r, expert_num, lora_alpha, lora_dropout, init_lora_weights)
        self.routing_strategy = routing_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # note: no check for self.merged because merging is not supported (yet)
        result = self.base_layer(x)
        B, S, in_feat = x.shape
        N = B * S
        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()

            lora_A = self.lora_A[active_adapter]            # [expert_num, in_features, r]
            lora_B = self.lora_B[active_adapter]            # [expert_num, r, out_features]
            lora_router = self.lora_router[active_adapter]  # Linear(in_features -> expert_num)
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            expert_num = self.expert_num[active_adapter]
            r = self.r[active_adapter]

            # 1. 对输入做 drop out
            #    x_dropped: [B, S, in_features]
            x_dropped = dropout(x)

            # 2. 路由器打分: [B, S, expert_num]
            logits = lora_router(x_dropped)
            
            # 3. 将 lora_A 复制 expert_num 次，使其形状与 lora_B 匹配
            lora_A_expanded = lora_A.unsqueeze(0).expand(expert_num, -1, -1)  # [expert_num, in_features, r]
            
            gating_probs = F.softmax(logits, dim=-1).to(torch.bfloat16)  # [B, S, expert_num]
            mid = torch.einsum("bsd,edr->bser", x_dropped, lora_A_expanded)
            mid = torch.einsum("bser,erd->bsed", mid, lora_B)
            lora_outpout = torch.einsum("bsed,bse->bsd", mid, gating_probs)
              

            if requires_conversion:
                lora_outpout = lora_outpout.to(expected_dtype)

            result = result + lora_outpout * scaling
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hydralora." + rep
