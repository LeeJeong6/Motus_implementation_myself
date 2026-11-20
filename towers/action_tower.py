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

# TODO
# - class ActionBlock(nn.Module)  : norm1/2, wan_action_qkv, wan_action_o, norm_q/k, ffn, modulation
# - class ActionDecoder(nn.Module)
# - class ActionTower(nn.Module)  : blocks × num_layers + time_embedding + decoder
#     time_embedding / modulation / qkv / attn_out / ffn / head
# - __main__ : dummy action tokens 로 shape 확인
