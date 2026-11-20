"""
video_tower.py는 Video Generation Tower (WAN2.2-5B DiT 블록) 를 정의하는 스크립트이다.
"""

"""
원본: bak/wan/modules/model.py (WanAttentionBlock, WanModel), models/motus.py:98 VideoModule

입력  : video tokens [B, 360, 3072]   (Video_encoder.py 출력)
        T5 context   [B, 512, 3072]   (Text_encoder.py 출력)
        video timestep t [B]          (0~1000)
출력  : video velocity [B, 48, 3, 24, 20]   (head + unpatchify, loss 는 0번 latent 제외)

설정 : dim=3072, num_heads=24, head_dim=128, ffn_dim=14336, num_layers=5 (검증용, 원본 30)
       pretrained WAN 은 불러오지 않고 랜덤 초기화 (shape 확인용)
"""

"""
레이어 하나(WanAttentionBlock)에서 video tower 가 맡는 부분
  - modulation : 블록별 파라미터 [1, 6, 3072] + timestep AdaLN 파라미터 -> 6개로 chunk
  - qkv        : norm1 → AdaLN(shift0, scale1) → q, k, v Linear → RMSNorm(q, k) → [B, L, 24, 128]
  - attn_out   : o Linear → residual × gate2
  - cross_attn : norm3 → cross-attention (Q=video, K/V=T5 context) → residual (AdaLN/gate 없음)
  - ffn        : norm2 → AdaLN(shift3, scale4) → Linear-GELU-Linear → residual × gate5
"""

"""
블록 바깥(모델 단위) 부품
  - time_embedding : sinusoidal(256) → Linear-SiLU-Linear → time_projection → [B, L, 6, 3072]
  - head           : WAN Head (norm + 2-way modulation + Linear(3072 → 48·1·2·2))
  - unpatchify     : [B, 360, 192] -> [B, 48, 3, 24, 20]
  - text_embedding 은 Text_encoder.py 에 둔다 (토큰화/조건 준비 단계로 분리)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from towers.layers import WanLayerNorm, WanRMSNorm, sinusoidal_embedding_1d, split_modulation


# ---------------------------------------------------------------------------
# 블록 부품  (파라미터 이름은 원본 WAN checkpoint 키와 같게 맞춤: blocks.i.self_attn.q ...)
# ---------------------------------------------------------------------------
class AttnProj(nn.Module):
    """WanSelfAttention / WanCrossAttention 의 가중치 (q, k, v, o + QK RMSNorm)"""

    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)


class VideoBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = AttnProj(dim, num_heads, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)  # cross_attn_norm=True
        self.cross_attn = AttnProj(dim, num_heads, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.float().unsqueeze(0) + e.float().unsqueeze(2)).chunk(2, dim=2)
        return self.head(self.norm(x).float() * (1 + scale.squeeze(2)) + shift.squeeze(2))


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------
class VideoTower(nn.Module):
    def __init__(self, dim: int = 3072, ffn_dim: int = 14336, num_heads: int = 24, num_layers: int = 5,
                 freq_dim: int = 256, out_dim: int = 48, patch_size: Tuple[int, int, int] = (1, 2, 2),
                 eps: float = 1e-6):
        super().__init__()
        self.dim, self.num_heads, self.freq_dim = dim, num_heads, freq_dim
        self.out_dim, self.patch_size = out_dim, patch_size

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([VideoBlock(dim, ffn_dim, num_heads, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        self.init_weights()

    def init_weights(self):
        """원본: WanModel.init_weights (bak/wan/modules/model.py:645)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)

    # ---- timestep ----
    def time_embedding_fn(self, t: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        t [B] (0~1000) -> head 용 e [B, L, dim], AdaLN 용 e0 [B, L, 6, dim]  (float32)
        원본은 t 를 [B, L] 로 펼친 뒤 계산하지만 같은 t 이므로 [B] 로 계산 후 펼친다.
        """
        with torch.autocast(device_type=t.device.type, enabled=False):
            emb = sinusoidal_embedding_1d(self.freq_dim, t).float()                         # [B, 256]
            e = self.time_embedding(emb.to(self.time_embedding[0].weight.dtype)).float()    # [B, dim]
            e0 = self.time_projection(e.to(self.time_projection[1].weight.dtype)).float()   # [B, 6·dim]
        B = t.shape[0]
        return e.unsqueeze(1).expand(B, seq_len, -1), e0.view(B, 1, 6, self.dim).expand(B, seq_len, 6, self.dim)

    def modulation(self, adaln_params: torch.Tensor, i: int):
        return split_modulation(self.blocks[i].modulation, adaln_params)

    # ---- joint self-attention 전/후 ----
    def qkv(self, x: torch.Tensor, i: int, mod) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blk = self.blocks[i]
        h = blk.norm1(x).float() * (1 + mod[1]) + mod[0]
        a = blk.self_attn
        return a.split_heads(a.norm_q(a.q(h))), a.split_heads(a.norm_k(a.k(h))), a.split_heads(a.v(h))

    def attn_out(self, x: torch.Tensor, o: torch.Tensor, i: int, mod) -> torch.Tensor:
        y = self.blocks[i].self_attn.o(o.flatten(2))
        return x + y.float() * mod[2]

    # ---- T5 cross-attention (video 만) ----
    def cross_attn(self, x: torch.Tensor, context: torch.Tensor, i: int) -> torch.Tensor:
        blk = self.blocks[i]
        a = blk.cross_attn
        h = blk.norm3(x)
        q = a.split_heads(a.norm_q(a.q(h))).transpose(1, 2)            # [B, N, L_v, D]
        k = a.split_heads(a.norm_k(a.k(context))).transpose(1, 2)      # [B, N, 512, D]
        v = a.split_heads(a.v(context)).transpose(1, 2)
        dtype = torch.bfloat16 if q.is_cuda else torch.float32
        y = F.scaled_dot_product_attention(q.to(dtype), k.to(dtype), v.to(dtype)).transpose(1, 2).flatten(2)
        return x + a.o(y).float()

    # ---- FFN ----
    def ffn(self, x: torch.Tensor, i: int, mod) -> torch.Tensor:
        blk = self.blocks[i]
        h = blk.norm2(x).float() * (1 + mod[4]) + mod[3]
        return x + blk.ffn(h).float() * mod[5]

    # ---- 출력 ----
    def head_out(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.head(x, e)                                          # [B, L_v, 48·1·2·2]

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """[B, L_v, 192] -> [B, 48, F, H', W']  (원본: WanModel.unpatchify)"""
        B = x.shape[0]
        f, h, w = grid_sizes[0].tolist()
        pf, ph, pw = self.patch_size
        u = x.float().view(B, f, h, w, pf, ph, pw, self.out_dim)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        return u.reshape(B, self.out_dim, f * pf, h * ph, w * pw)


# ---------------------------------------------------------------------------
# Dummy input 으로 확인
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, L_v = 2, 360
    grid = torch.tensor([[3, 12, 10]] * B)

    tower = VideoTower(num_layers=5).to(device)
    tower.blocks.to(torch.bfloat16)
    print(f"VideoTower 파라미터: {sum(p.numel() for p in tower.parameters()) / 1e6:.1f}M (블록 5개)")

    x = torch.randn(B, L_v, 3072, device=device)
    ctx = torch.randn(B, 512, 3072, device=device)
    t = torch.tensor([300.0, 900.0], device=device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        e, e0 = tower.time_embedding_fn(t, L_v)
        mod = tower.modulation(e0, 0)
        q, k, v = tower.qkv(x, 0, mod)
        print(f"time emb e {tuple(e.shape)}  e0 {tuple(e0.shape)}   qkv {tuple(q.shape)}")
        x = tower.attn_out(x, v, 0, mod)          # (attention 결과 대신 v 를 넣어 shape 만 확인)
        x = tower.cross_attn(x, ctx, 0)
        x = tower.ffn(x, 0, mod)
        out = tower.unpatchify(tower.head_out(x, e), grid)
    print(f"after block {tuple(x.shape)}   velocity {tuple(out.shape)}  (head zero-init -> {out.abs().max().item():.1f})")
