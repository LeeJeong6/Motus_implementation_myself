"""
action_tower.py는 Action Expert Tower 를 정의하는 스크립트이다.
"""

"""
원본: models/action_expert.py (ActionExpertBlock, ActionDecoder, ActionExpert), models/motus.py:427 ActionModule

입력  : action tokens [B, 21, 1024]   (Action_encoder.py 출력: state 1 | action 16 | register 4)
        action timestep t [B]         (0~1000, video 와 독립적으로 샘플링)
출력  : action velocity [B, 16, 14]   (decoder 출력 [B, 21, 14] 에서 state/register 를 잘라낸 [:, 1:-4])

설정 : dim=1024, ffn_dim=4096, num_layers=5 (video tower 와 1:1 대응), from scratch
"""

"""
레이어 하나(ActionExpertBlock)에서 action tower 가 맡는 부분
  - modulation : 블록별 파라미터 [1, 6, 1024] + timestep AdaLN 파라미터 -> 6개로 chunk
  - qkv        : norm1 → AdaLN → wan_action_qkv [3, 24, 1024, 128] 로 WAN head space 사영
                 → RMSNorm(q, k) (3072 차원 기준) → [B, 21, 24, 128]
  - attn_out   : wan_action_o Linear(3072 → 1024) → residual × gate
  - ffn        : norm2 → AdaLN → Linear(1024→4096)-GELU-Linear(4096→1024) → residual × gate
  - (cross-attention 없음: T5 를 직접 보지 않는다)
"""

"""
블록 바깥 부품
  - time_embedding : sinusoidal(256) → Linear-SiLU-Linear → time_projection → [B, 21, 6, 1024]  (float32)
  - decoder        : norm + 2-way modulation → Linear(1024 → 14), 출력층 zero-init
"""

from typing import Tuple

import torch
import torch.nn as nn

from towers.layers import WanLayerNorm, WanRMSNorm, build_mlp, sinusoidal_embedding_1d, split_modulation


class ActionBlock(nn.Module):
    """원본: models/action_expert.py:218 ActionExpertBlock"""

    def __init__(self, dim: int, ffn_dim: int, wan_num_heads: int, wan_head_dim: int, eps: float):
        super().__init__()
        wan_dim = wan_num_heads * wan_head_dim
        self.norm1 = WanLayerNorm(dim, eps=eps)
        self.norm2 = WanLayerNorm(dim, eps=eps)
        # action(1024) -> WAN head space (24 × 128) 사영. [K=3(q,k,v), N, D_in, D_head]
        self.wan_action_qkv = nn.Parameter(torch.randn(3, wan_num_heads, dim, wan_head_dim) / (dim * wan_head_dim) ** 0.5)
        self.wan_action_o = nn.Linear(wan_dim, dim, bias=False)
        self.wan_action_norm_q = WanRMSNorm(wan_dim, eps=eps)
        self.wan_action_norm_k = WanRMSNorm(wan_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)


class ActionDecoder(nn.Module):
    """원본: models/action_expert.py:264 ActionDecoder (WAN Head 와 같은 2-way modulation)"""

    def __init__(self, dim: int, action_dim: int, eps: float):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps=eps)
        self.action_head = build_mlp('mlp1x_silu', dim, action_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.float().unsqueeze(0) + e.float().unsqueeze(2)).chunk(2, dim=2)
        return self.action_head(self.norm(x).float() * (1 + scale.squeeze(2)) + shift.squeeze(2))


class ActionTower(nn.Module):
    def __init__(self, dim: int = 1024, ffn_dim: int = 4096, num_layers: int = 5, action_dim: int = 14,
                 wan_num_heads: int = 24, wan_head_dim: int = 128, eps: float = 1e-6, freq_dim: int = 256):
        super().__init__()
        self.dim, self.freq_dim = dim, freq_dim
        self.wan_num_heads, self.wan_head_dim = wan_num_heads, wan_head_dim

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([ActionBlock(dim, ffn_dim, wan_num_heads, wan_head_dim, eps)
                                     for _ in range(num_layers)])
        self.decoder = ActionDecoder(dim, action_dim, eps)
        self.initialize_weights()

    def initialize_weights(self):
        """원본: models/action_expert.py:362"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.decoder.action_head[-1].weight)   # 출력층 zero-init
        nn.init.zeros_(self.decoder.action_head[-1].bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    # ---- timestep (float32) ----
    def time_embedding_fn(self, t: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """t [B] -> e [B, L, dim], e0 [B, L, 6, dim]  (원본: models/motus.py:439)"""
        with torch.autocast(device_type=t.device.type, enabled=False):
            emb = sinusoidal_embedding_1d(self.freq_dim, t).float()
            e = self.time_embedding(emb.to(self.time_embedding[0].weight.dtype)).float()
            e0 = self.time_projection(e.to(self.time_projection[1].weight.dtype)).float()
        B = t.shape[0]
        return e.unsqueeze(1).expand(B, seq_len, -1), e0.view(B, 1, 6, self.dim).expand(B, seq_len, 6, self.dim)

    def modulation(self, adaln_params: torch.Tensor, i: int):
        return split_modulation(self.blocks[i].modulation, adaln_params)

    # ---- joint self-attention 전/후 ----
    def qkv(self, x: torch.Tensor, i: int, mod):
        """원본: models/motus.py:232-245"""
        blk = self.blocks[i]
        B, L, _ = x.shape
        h = blk.norm1(x).float() * (1 + mod[1]) + mod[0]
        qkv = torch.einsum("BTD,KNDE->KBTNE", h.to(blk.wan_action_qkv.dtype), blk.wan_action_qkv)  # [3, B, L, N, E]
        n, d = self.wan_num_heads, self.wan_head_dim
        q = blk.wan_action_norm_q(qkv[0].flatten(-2)).view(B, L, n, d)
        k = blk.wan_action_norm_k(qkv[1].flatten(-2)).view(B, L, n, d)
        return q, k, qkv[2]

    def attn_out(self, x: torch.Tensor, o: torch.Tensor, i: int, mod) -> torch.Tensor:
        y = self.blocks[i].wan_action_o(o.flatten(2).to(self.blocks[i].wan_action_o.weight.dtype))
        return x + y.float() * mod[2]

    # ---- FFN ----
    def ffn(self, x: torch.Tensor, i: int, mod) -> torch.Tensor:
        blk = self.blocks[i]
        h = blk.norm2(x).float() * (1 + mod[4]) + mod[3]
        return x + blk.ffn(h).float() * mod[5]

    # ---- 출력 ----
    def head_out(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, e)                                 # [B, 21, 14]


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L_a = 2, 21
    tower = ActionTower(num_layers=5)
    print(f"ActionTower 파라미터: {sum(p.numel() for p in tower.parameters()) / 1e6:.1f}M (블록 5개, from scratch)")
    x = torch.randn(B, L_a, 1024)
    t = torch.tensor([300.0, 900.0])
    e, e0 = tower.time_embedding_fn(t, L_a)
    mod = tower.modulation(e0, 0)
    q, k, v = tower.qkv(x, 0, mod)
    print(f"qkv {tuple(q.shape)}  (WAN head space)")
    x = tower.attn_out(x, v, 0, mod)
    x = tower.ffn(x, 0, mod)
    full = tower.head_out(x, e)
    print(f"decoder {tuple(full.shape)} -> action velocity {tuple(full[:, 1:-4].shape)}")
