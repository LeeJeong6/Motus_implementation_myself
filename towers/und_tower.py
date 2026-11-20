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

# TODO
# - build_mlp('mlp3x_silu', 2048, 512)  : vlm_adapter
# - class UndBlock(nn.Module)  : norm1/2, wan_und_qkv, wan_und_o, norm_q/k, ffn
# - class UndTower(nn.Module)  : vlm_adapter + blocks × num_layers
#     adapt / qkv / attn_out / ffn
# - __main__ : dummy VLM hidden state 로 shape 확인
