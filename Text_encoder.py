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

"""
결정: text_embedding 은 여기(Text_encoder.py)에 둔다.
      원본은 WAN 모델 안(wan_model.text_embedding)에 있지만, 토큰화/조건 준비 단계를 tower 와 분리하는
      이 프로젝트의 구성에 맞춰 분리했다. 연산은 원본과 같다.
"""

from typing import List

import torch
import torch.nn as nn

T5_DIM = 4096
TEXT_LEN = 512


def pad_t5_embeddings(embeddings: List[torch.Tensor], text_len: int = TEXT_LEN) -> torch.Tensor:
    """[L_i, 4096] 리스트 -> 0-padding / 자르기로 [B, 512, 4096]  (원본: data/dataset.py:278)"""
    padded = []
    for emb in embeddings:
        if emb.shape[0] <= text_len:
            padded.append(torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])]))
        else:
            padded.append(emb[:text_len])
    return torch.stack(padded, dim=0)


class TextEmbedding(nn.Module):
    """T5 [B, 512, 4096] -> T5 context [B, 512, 3072]  (원본: WanModel.text_embedding)"""

    def __init__(self, text_dim: int = T5_DIM, dim: int = 3072):
        super().__init__()
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        for m in self.text_embedding.modules():  # 원본 WanModel.init_weights: normal(0.02), bias 0
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                nn.init.zeros_(m.bias)

    def forward(self, t5_embeddings) -> torch.Tensor:
        if isinstance(t5_embeddings, list):
            t5_embeddings = pad_t5_embeddings(t5_embeddings)
        w = self.text_embedding[0].weight
        return self.text_embedding(t5_embeddings.to(w.device, w.dtype))


def make_dummy_t5(B: int, min_len: int = 8, max_len: int = 30, seed: int = 0) -> List[torch.Tensor]:
    """umT5 출력 대신 쓰는 랜덤 [L_i, 4096] (지시문마다 길이가 다름)"""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(min_len, max_len + 1, (B,), generator=g).tolist()
    return [torch.randn(L, T5_DIM, generator=g) * 0.1 for L in lengths]


if __name__ == "__main__":
    B = 2
    dummy = make_dummy_t5(B)
    print(f"dummy T5 출력 길이: {[e.shape[0] for e in dummy]}  (각 [L, 4096])")
    padded = pad_t5_embeddings(dummy)
    print(f"padding 후     : {tuple(padded.shape)}")
    enc = TextEmbedding()
    ctx = enc(dummy)
    print(f"T5 context     : {tuple(ctx.shape)}   (video tower cross-attention 의 K/V)")
    print(f"text_embedding 파라미터: {sum(p.numel() for p in enc.parameters()) / 1e6:.1f}M (trainable)")
