"""
layers.py : 세 tower 가 공통으로 쓰는 작은 부품.
원본: bak/wan/modules/model.py (sinusoidal_embedding_1d, WanRMSNorm, WanLayerNorm),
      models/action_expert.py (build_mlp)
"""

import re

import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """timestep [N] -> [N, dim]  (cos | sin, float64 로 계산)"""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class WanLayerNorm(nn.LayerNorm):
    """float32 로 정규화한 뒤 입력 dtype 으로 되돌린다. 기본은 affine 없음."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


def build_mlp(projector_type: str, in_features: int, out_features: int) -> nn.Module:
    """'linear' 또는 'mlp{N}x_silu' (Linear → [SiLU → Linear] × (N-1))"""
    if projector_type == 'linear':
        return nn.Linear(in_features, out_features)
    m = re.match(r'^mlp(\d+)x_silu$', projector_type)
    if m is None:
        raise ValueError(f'Unknown projector type: {projector_type}')
    modules = [nn.Linear(in_features, out_features)]
    for _ in range(1, int(m.group(1))):
        modules += [nn.SiLU(), nn.Linear(out_features, out_features)]
    return nn.Sequential(*modules)


def split_modulation(block_modulation: torch.Tensor, adaln_params: torch.Tensor):
    """
    블록 고유 파라미터 [1, 6, D] + timestep AdaLN 파라미터 [B, L, 6, D]  (float32 로 계산)
    -> 6개 [B, L, D] : (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)
    """
    mod = block_modulation.float().unsqueeze(0) + adaln_params.float()
    return tuple(m.squeeze(2) for m in mod.chunk(6, dim=2))
