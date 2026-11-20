"""
towers/ : MoT 의 세 expert tower.

각 tower 는 "입력을 받아 30개 레이어를 끝까지 도는 모델"이 아니라
레이어 단위 부품을 내주는 파라미터 묶음이다. 레이어 루프는 MOTUS.py 가 돈다.

공통 인터페이스 (und 는 time embedding / AdaLN / head 없음)
  time_embedding(t)            -> (head_emb, adaln_params)
  modulation(adaln_params, i)  -> 6개 AdaLN 파라미터 (shift/scale/gate × attn, ffn)
  qkv(x, i, mod)               -> q, k, v   [B, L, 24, 128]  (WAN head space)
  attn_out(x, o, i, mod)       -> x         출력 사영 + residual (+gate)
  ffn(x, i, mod)               -> x
  head(x, head_emb)            -> velocity
"""
