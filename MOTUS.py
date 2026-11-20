"""
Motus: A Unified Latent ActionWorld Model 를 직접 구현한다.

"""

"""
MOTUS.py는 encoder 들의 출력 토큰을 받아 세 tower 를 레이어마다 맞물려 돌리는 조립(orchestrator) 스크립트이다.
원본: models/motus.py (Motus.training_step :768, inference_step :911)

검증 단계 설정 : num_layers = 5, pretrained backbone 은 불러오지 않음 (랜덤 초기화), dummy data 로 shape 만 확인
"""

"""
구성 요소
  [토큰화 / 조건]                                  [tower]                      [연산]
  VLM.py            -> und 입력   [B, L_u, 2048]    towers/und_tower.py          joint_attention.py
  Video_encoder.py  -> video 토큰 [B, 360, 3072]    towers/video_tower.py
  Action_encoder.py -> action 토큰 [B, 21, 1024]    towers/action_tower.py
  Text_encoder.py   -> T5 context [B, 512, 3072]
"""

"""
training_step 흐름
  1. 토큰화
       video  : Video_encoder.forward_train  -> video tokens, video target, t_video, grid_sizes
       action : Action_encoder.forward_train -> action tokens, action target, t_action
       und    : VLM forward -> und_tower.adapt -> und tokens [B, L_u, 512]
       text   : Text_encoder -> T5 context
  2. time embedding
       video_tower.time_embedding(t_video), action_tower.time_embedding(t_action)
  3. 레이어 루프 (i = 0..num_layers-1)
       ① modulation : video / action AdaLN 파라미터
       ② qkv        : 세 tower 가 각자 q, k, v 를 WAN head space 로 사영
       ③ joint_attention([video, action, und])  ->  o_v, o_a, o_u
       ④ attn_out   : 세 tower 가 각자 출력 사영 + residual
       ⑤ video_tower.cross_attn(video, T5 context)   (video 만)
       ⑥ ffn        : 세 tower 가 각자 FFN
  4. head
       video_tower.head -> video velocity [B, 48, 3, 24, 20]
       action_tower.head -> action velocity [B, 16, 14]
  5. loss
       video_loss  = MSE(velocity(0번 latent 제외), target)
       action_loss = MSE(velocity, target)
       total = w_v · video_loss + w_a · action_loss

inference_step 흐름
  - 미래 video latent / action 을 순수 노이즈에서 시작
  - t: 1 -> 0 으로 Euler 적분, 매 스텝 위의 1~4 반복 (0번 latent 는 매 스텝 teacher forcing)
  - 마지막에 VAE decode -> 예측 프레임, action latent -> 예측 action chunk
"""

# TODO
# - class MotusConfig    : num_layers=5, 각 tower 차원, loss weight 등
# - class Motus(nn.Module)
#     __init__        : encoder / tower 생성 (pretrained 미사용)
#     forward_layers  : 레이어 루프 (training / inference 공용)
#     training_step   : 위 1~5
#     inference_step  : denoising 루프
# - __main__ : dummy data 로 training_step / inference_step shape 확인
