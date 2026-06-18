import math
from typing import Any
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

class CaraLayer(BaseTunerLayer):
    """Base class for CaRA layers."""
    adapter_layer_names: tuple[str, ...] = ("cara_A", "cara_B")
    other_param_names: tuple[str, ...] = ("r", "noise_alpha", "noise_step_interval")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.noise_alpha = {}
        self.noise_step_interval = {}
        
        self.cara_A = nn.ParameterDict({})
        self.cara_B = nn.ParameterDict({})
        self.merged_adapters = []
        
        self.step_counter = 0 
        
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, r, noise_alpha, noise_step_interval, init_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        self.r[adapter_name] = r
        self.noise_alpha[adapter_name] = noise_alpha
        self.noise_step_interval[adapter_name] = noise_step_interval
        
        target_device = self.get_base_layer().weight.device
        target_dtype = self.get_base_layer().weight.dtype
        
        self.cara_A[adapter_name] = nn.Parameter(
            torch.randn(self.in_features, r, device=target_device, dtype=target_dtype) * 0.01
        )
        self.cara_B[adapter_name] = nn.Parameter(
            torch.randn(self.in_features, r, device=target_device, dtype=target_dtype) * 0.01
        )
        
        self.get_base_layer().weight.requires_grad = False
        self.set_adapter(self.active_adapters)


class CaraLinear(nn.Module, CaraLayer):
    """CaRA layer for nn.Linear."""
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 8,
        noise_alpha: float = 0.01,
        noise_step_interval: int = 5,
        **kwargs,
    ) -> None:
        super().__init__()
        CaraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, noise_alpha, noise_step_interval, init_weights=True)
        self.set_adapter(self.active_adapters)

    def get_orthogonal_matrix(self, adapter_name):
        A = self.cara_A[adapter_name]
        B = self.cara_B[adapter_name]
        r = self.r[adapter_name]
        
        U = torch.cat([A, -B], dim=1)
        V = torch.cat([B, A], dim=1)
        I_2r = torch.eye(2 * r, device=A.device, dtype=A.dtype)
        
        inner_matrix = I_2r + torch.matmul(V.transpose(0, 1), U)
        Z = torch.linalg.inv(inner_matrix.to(torch.float32)).to(A.dtype)
        
        return U, Z, V

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        
        if self.disable_adapters or not self.active_adapters:
            return self.base_layer(x, *args, **kwargs)
            
        active_adapter = self.active_adapters[0]
        U, Z, V = self.get_orthogonal_matrix(active_adapter)

        # Cast orthogonal matrices to input dtype (e.g. bfloat16 under mixed precision)
        U = U.to(x.dtype)
        Z = Z.to(x.dtype)
        V = V.to(x.dtype)
        
        # 기하학적 회전 적용
        x_U_Z = torch.matmul(torch.matmul(x, U), Z)
        x_rot = x - 2 * torch.matmul(x_U_Z, V.transpose(0, 1))
        
        noise_alpha = self.noise_alpha[active_adapter]
        interval = self.noise_step_interval[active_adapter]
        
        if self.training:
            self.step_counter += 1
            if (self.step_counter % interval == 0) and (noise_alpha > 0.0):
                R_rand = torch.randn(self.in_features, self.in_features, device=x.device, dtype=x.dtype)
                R_skew = (R_rand - R_rand.transpose(0, 1)) / math.sqrt(2 * self.in_features) # 반대칭 만들기!
                x_rot = x_rot + (noise_alpha * torch.matmul(x_rot, R_skew))

        out = self.base_layer(x_rot, *args, **kwargs)
        return out.to(previous_dtype)
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        rep += f"(r={self.r}, noise_alpha={self.noise_alpha}, noise_step_interval={self.noise_step_interval})"
        return rep  