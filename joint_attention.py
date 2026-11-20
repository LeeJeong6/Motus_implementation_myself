"""
joint_attention.py는 세 tower(video / action / understanding)의 토큰이 레이어마다 함께 수행하는
trimodal joint self-attention (MoT) 을 정의하는 스크립트이다.
"""

"""
원본은 이 연산이 WAN self-attention 안에 들어가 있다 (bak/wan/modules/model.py:198-270).
여기서는 tower 와 분리된 독립 함수로 만든다. (가중치 없음, 순수 연산)

입력  : 각 tower 가 WAN head space(24 heads × 128 = 3072)로 사영한 Q, K, V
          video  q,k,v [B, L_v, 24, 128]   (L_v = 360)
          action q,k,v [B, L_a, 24, 128]   (L_a = 21)
          und    q,k,v [B, L_u, 24, 128]   (L_u = VLM 시퀀스 길이)
출력  : attention 결과를 다시 3조각으로 나눈 것 (head shape 유지, 출력 사영은 각 tower 가 담당)
          o_v [B, L_v, 24, 128], o_a [B, L_a, 24, 128], o_u [B, L_u, 24, 128]
"""

"""
처리 순서
  1. 3D RoPE (f, h, w) 를 video 의 q, k 에만 적용  (action / und 에는 위치 회전 없음)
  2. [video | action | und] 를 시퀀스 축으로 concat
  3. full (non-causal) attention 한 번  (원본: flash_attention, 여기서는 F.scaled_dot_product_attention)
  4. 길이 (L_v, L_a, L_u) 로 split 해서 반환
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 3D RoPE  (원본: bak/wan/modules/model.py:29 rope_params, :41 rope_apply, :521 WanModel.freqs)
# ---------------------------------------------------------------------------
def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """[max_seq_len, dim/2] 복소수 회전 계수 e^{i·pos·ω}"""
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len),
                        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def build_rope_freqs(head_dim: int = 128, max_seq_len: int = 1024) -> torch.Tensor:
    """
    head_dim 128 = 복소수 64쌍을 (f, h, w) 축에 나눠 배정: f=22, h=21, w=21
    -> [1024, 64] complex128
    """
    d = head_dim
    return torch.cat([
        rope_params(max_seq_len, d - 4 * (d // 6)),  # f 축: 44 -> 22쌍
        rope_params(max_seq_len, 2 * (d // 6)),      # h 축: 42 -> 21쌍
        rope_params(max_seq_len, 2 * (d // 6)),      # w 축: 42 -> 21쌍
    ], dim=1)


def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    x [B, L, N, D] (L = f·h·w 인 video 토큰) 에 3D RoPE 적용 -> float32 [B, L, N, D]
    배치 안의 모든 샘플은 같은 grid (f, h, w) 라고 가정한다 (해상도 고정 학습).
    """
    B, L, N, D = x.shape
    C = D // 2
    f, h, w = grid_sizes[0].tolist()
    assert (grid_sizes == grid_sizes[0]).all(), "batch 내 grid 크기가 모두 같아야 함"
    assert f * h * w == L, f"video 토큰 수 {L} != f·h·w {f * h * w}"

    fpart, hpart, wpart = freqs.to(x.device).split([C - 2 * (C // 3), C // 3, C // 3], dim=1)
    freq_grid = torch.cat([
        fpart[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        hpart[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        wpart[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(L, 1, C)                                                  # [L, 1, C]

    x_c = torch.view_as_complex(x.to(torch.float64).reshape(B, L, N, C, 2))     # [B, L, N, C]
    return torch.view_as_real(x_c * freq_grid).reshape(B, L, N, D).float()


# ---------------------------------------------------------------------------
# 2. Joint attention
# ---------------------------------------------------------------------------
QKV = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def joint_attention(video_qkv: QKV, other_qkvs: Sequence[QKV], grid_sizes: torch.Tensor, freqs: torch.Tensor,
                    attn_dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
    """
    video_qkv  : (q, k, v) [B, L_v, N, D]  -> RoPE 적용
    other_qkvs : [(q, k, v) of action [B, L_a, N, D], (q, k, v) of und [B, L_u, N, D], ...]  -> RoPE 없음
    return     : [o_v, o_a, o_u, ...]  각 [B, L_*, N, D]
    """
    q_v, k_v, v_v = video_qkv
    q_v, k_v = rope_apply(q_v, grid_sizes, freqs), rope_apply(k_v, grid_sizes, freqs)

    if attn_dtype is None:  # 원본 flash_attention 처럼 cuda 에서는 bf16 으로 계산
        attn_dtype = torch.bfloat16 if q_v.is_cuda else torch.float32

    qs = [q_v] + [q for q, _, _ in other_qkvs]
    ks = [k_v] + [k for _, k, _ in other_qkvs]
    vs = [v_v] + [v for _, _, v in other_qkvs]
    lengths = [q.shape[1] for q in qs]

    # [B, L_total, N, D] -> SDPA 형식 [B, N, L_total, D]
    q = torch.cat([t.to(attn_dtype) for t in qs], dim=1).transpose(1, 2)
    k = torch.cat([t.to(attn_dtype) for t in ks], dim=1).transpose(1, 2)
    v = torch.cat([t.to(attn_dtype) for t in vs], dim=1).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v)            # full attention, scale = 1/sqrt(D)
    out = out.transpose(1, 2)                                # [B, L_total, N, D]
    return list(out.split(lengths, dim=1))


# ---------------------------------------------------------------------------
# 3. Dummy input 으로 확인
# ---------------------------------------------------------------------------
def _reference_rope(x, grid_sizes, freqs):
    """원본 rope_apply_original 과 같은 샘플별 루프 구현 (검증용)"""
    n, c = x.size(2), x.size(3) // 2
    fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        L = f * h * w
        x_i = torch.view_as_complex(x[i, :L].to(torch.float64).reshape(L, n, -1, 2))
        f_i = torch.cat([fr[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                         fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                         fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(L, 1, -1)
        out.append(torch.view_as_real(x_i * f_i).flatten(2))
    return torch.stack(out).float()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 2, 24, 128
    grid = torch.tensor([[3, 12, 10]] * B)
    L_v, L_a, L_u = 360, 21, 141
    freqs = build_rope_freqs(D)

    mk = lambda L: tuple(torch.randn(B, L, N, D) for _ in range(3))
    vq, aq, uq = mk(L_v), mk(L_a), mk(L_u)

    # RoPE: 원본 루프 구현과 일치 + 회전이라 벡터 크기 보존
    r = rope_apply(vq[0], grid, freqs)
    print(f"RoPE == 원본 루프 구현     : {torch.allclose(r, _reference_rope(vq[0], grid, freqs), atol=1e-5)}")
    print(f"RoPE 후 norm 보존          : {torch.allclose(r.norm(dim=-1), vq[0].norm(dim=-1), atol=1e-4)}")

    o_v, o_a, o_u = joint_attention(vq, [aq, uq], grid, freqs, attn_dtype=torch.float32)
    print(f"\no_v {tuple(o_v.shape)}  o_a {tuple(o_a.shape)}  o_u {tuple(o_u.shape)}")

    # 하나의 attention 인지 확인: action 쪽 V 만 바꿔도 video 출력이 바뀌어야 한다
    aq2 = (aq[0], aq[1], aq[2] + 1.0)
    o_v2, _, _ = joint_attention(vq, [aq2, uq], grid, freqs, attn_dtype=torch.float32)
    print(f"action V 변경 -> video 출력 변화 : {(o_v2 - o_v).abs().max().item():.4f}  (> 0 이면 joint)")
