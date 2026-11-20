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
  - (선택) text_embedding : Text_encoder.py 와 어디에 둘지 결정
"""

# TODO
# - class VideoBlock(nn.Module)  : 위 레이어 부품 (self-attn QKVO, cross-attn, FFN, norm, modulation)
# - class VideoTower(nn.Module)  : blocks × num_layers + time_embedding + head
#     time_embedding / modulation / qkv / attn_out / cross_attn / ffn / head / unpatchify
# - __main__ : dummy video tokens 로 레이어 부품별 shape 확인
