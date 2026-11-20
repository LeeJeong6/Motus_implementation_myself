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

# TODO
# - rope_params(max_len, dim)            : RoPE 주파수 테이블 (원본: WanModel.freqs)
# - rope_apply(x, grid_sizes, freqs)      : video 토큰에 3D RoPE 적용 (head_dim 128 을 f/h/w 로 분할)
# - joint_attention(qkv_list, grid_sizes, freqs) -> (o_v, o_a, o_u)
# - __main__ : dummy q/k/v 로 shape 확인
