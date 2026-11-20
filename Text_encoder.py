"""
Text_encoder.py는 Video Generation Tower의 cross-attention에 들어갈 T5 text context를 만드는 스크립트이다.
(video 토큰이 아니라 "조건"이며, 레이어 루프 전에 한 번만 계산한다.)
"""

"""
원본 흐름
  지시문 문자열
    -> umT5-XXL encoder (frozen, 오프라인 전처리)          [L_text, 4096]
    -> 512 길이로 0-padding                                [B, 512, 4096]
    -> WAN text_embedding: Linear(4096→3072) → GELU → Linear(3072→3072)  (trainable)
    -> T5 context                                         [B, 512, 3072]

원본 위치
  - T5 인코딩      : bak/wan/modules/t5.py (T5EncoderModel), inference/real_world/Motus/encode_t5_instruction.py
  - 패딩           : data/dataset.py:278 _process_language_embeddings_batch
  - text_embedding : bak/wan/modules/model.py:501, models/motus.py:130 preprocess_t5_embeddings
"""

"""
구현 범위 (검증 단계)
  - pretrained umT5 는 불러오지 않는다. dummy T5 출력 [L_text, 4096] 을 랜덤으로 만든다.
  - text_embedding 은 원본 WAN 과 같은 구조로 랜덤 초기화한다.
  - 원본처럼 text_embedding 가중치를 video_tower 안에 둘지, 여기 둘지 구현 시 결정한다.
"""

"""
이게 왜 있나 했는데, 논문에는 안나와있지만 video generation model의 attention은 joint attention과 cross-attention이 있었음,,
이때 joint attention은 [video | action |und] 이고 cross-attention은 video token과 T5 tokens끼리 cross-attention하는 것이다.
즉 지시문이 pre-trained VLM으로만 represented되는 게 아니라 cross-attention으로 video-generation model 내부에서도 그 정보를 계속 참조하는 것이다.
"""

# TODO
# - pad_t5_embeddings(list of [L_i, 4096], text_len=512) -> [B, 512, 4096]
# - class TextEmbedding(nn.Module): Linear(4096, 3072) → GELU(tanh) → Linear(3072, 3072)
# - make_dummy_t5(B) : 길이가 다른 랜덤 T5 출력 생성
# - __main__ : dummy 입력으로 [B, 512, 3072] 확인
