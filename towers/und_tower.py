"""
und_tower.py는 Understanding Expert Tower 를 정의하는 스크립트이다.
"""

"""
원본: models/und_expert.py (UndExpertBlock, UndExpert), models/motus.py:283 UndModule

입력  : VLM last hidden state [B, L_u, 2048]   (VLM.py 출력)
출력  : 없음 (최종 und 토큰은 버린다). 매 레이어 joint attention 의 K/V 로 video/action 에 정보를 준다.

설정 : dim=512, ffn_dim=2048, num_layers=5 (video tower 와 1:1 대응), from scratch
"""

"""
진입 전 1회
  - vlm_adapter : mlp3x_silu  Linear(2048→512) → SiLU → Linear(512→512) → SiLU → Linear(512→512)
                  [B, L_u, 2048] -> [B, L_u, 512]

레이어 하나(UndExpertBlock)에서 und tower 가 맡는 부분
  - qkv      : norm1 (일반 LayerNorm, AdaLN 없음) → wan_und_qkv [3, 24, 512, 128] → RMSNorm(q, k) → [B, L_u, 24, 128]
  - attn_out : wan_und_o Linear(3072 → 512) → residual (gate 없음)
  - ffn      : norm2 → Linear(512→2048)-GELU-Linear(2048→512) → residual
  - (timestep 조건 / cross-attention / head 없음)
"""

import torch
import torch.nn as nn

from towers.layers import WanLayerNorm, WanRMSNorm, build_mlp


class UndBlock(nn.Module):
    """원본: models/und_expert.py:74 UndExpertBlock"""

    def __init__(self, dim: int, ffn_dim: int, wan_num_heads: int, wan_head_dim: int, eps: float):
        super().__init__()
        wan_dim = wan_num_heads * wan_head_dim
        self.norm1 = WanLayerNorm(dim, eps=eps)
        self.norm2 = WanLayerNorm(dim, eps=eps)
        self.wan_und_qkv = nn.Parameter(torch.randn(3, wan_num_heads, dim, wan_head_dim) / (dim * wan_head_dim) ** 0.5)
        self.wan_und_o = nn.Linear(wan_dim, dim, bias=False)
        self.wan_und_norm_q = WanRMSNorm(wan_dim, eps=eps)
        self.wan_und_norm_k = WanRMSNorm(wan_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))


class UndTower(nn.Module):
    def __init__(self, dim: int = 512, ffn_dim: int = 2048, num_layers: int = 5, vlm_input_dim: int = 2048,
                 projector_type: str = "mlp3x_silu", wan_num_heads: int = 24, wan_head_dim: int = 128,
                 eps: float = 1e-5):
        super().__init__()
        self.wan_num_heads, self.wan_head_dim = wan_num_heads, wan_head_dim
        self.vlm_adapter = build_mlp(projector_type, vlm_input_dim, dim)
        self.blocks = nn.ModuleList([UndBlock(dim, ffn_dim, wan_num_heads, wan_head_dim, eps)
                                     for _ in range(num_layers)])
        # 원본 UndExpert 는 별도 init 이 없다 (PyTorch 기본 초기화)

    def adapt(self, vlm_hidden: torch.Tensor) -> torch.Tensor:
        """[B, L_u, 2048] -> [B, L_u, 512]  (원본: motus.py:336)"""
        return self.vlm_adapter(vlm_hidden.to(self.vlm_adapter[0].weight.dtype))

    def qkv(self, x: torch.Tensor, i: int):
        """원본: motus.py:247-256"""
        blk = self.blocks[i]
        B, L, _ = x.shape
        h = blk.norm1(x)
        qkv = torch.einsum("BTD,KNDE->KBTNE", h.to(blk.wan_und_qkv.dtype), blk.wan_und_qkv)
        n, d = self.wan_num_heads, self.wan_head_dim
        q = blk.wan_und_norm_q(qkv[0].flatten(-2)).view(B, L, n, d)
        k = blk.wan_und_norm_k(qkv[1].flatten(-2)).view(B, L, n, d)
        return q, k, qkv[2]

    def attn_out(self, x: torch.Tensor, o: torch.Tensor, i: int) -> torch.Tensor:
        return x + self.blocks[i].wan_und_o(o.flatten(2).to(self.blocks[i].wan_und_o.weight.dtype)).float()

    def ffn(self, x: torch.Tensor, i: int) -> torch.Tensor:
        blk = self.blocks[i]
        return x + blk.ffn(blk.norm2(x)).float()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L_u = 2, 141
    tower = UndTower(num_layers=5)
    print(f"UndTower 파라미터: {sum(p.numel() for p in tower.parameters()) / 1e6:.1f}M (adapter + 블록 5개, from scratch)")
    x = tower.adapt(torch.randn(B, L_u, 2048))
    q, k, v = tower.qkv(x, 0)
    print(f"adapt {tuple(x.shape)}   qkv {tuple(q.shape)}")
    x = tower.ffn(tower.attn_out(x, v, 0), 0)
    print(f"after block {tuple(x.shape)}")
