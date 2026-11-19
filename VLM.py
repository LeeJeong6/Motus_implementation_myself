"""
VLM.py는 Motus 모델의 pre-trained VLM에 해당하는 모듈을 설계한 스크립트이다.
"""

"""
pretrained VLM은 원본 코드에서 Qwen3-VL-2B(frozen)으로 정의한다.
그리고 이 VLM의 출력은 Understanding Expert로 입력되며
이 스크립트는 Understanding expert 이전, 즉 pretrained VLM만을 정의한다.
"""

"""
입력 : [system/user template token | image patch tokens | instruction text tokens | "<|im_start|>assistant\n"]
출력 : [B, L_vlm, 2048]
"""

"""
원본 논문에서는 transformers의 Qwen3VLForConditionalGeneration 을 import하며
AutoProcessor, qwen_vl_utils.process_vision_info 로 전처리한다.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


VLM_CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"


def preprocess_vlm_messages(text_instruction: str, image_pil: Image.Image, processor) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": text_instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs



def collate_vlm_inputs(vlm_inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].shape[1] for x in vlm_inputs)
    input_ids, attention_mask = [], []
    for x in vlm_inputs:
        pad = max_len - x["input_ids"].shape[1]
        input_ids.append(torch.cat([x["input_ids"], x["input_ids"].new_zeros(1, pad)], dim=1))
        attention_mask.append(torch.cat([x["attention_mask"], x["attention_mask"].new_zeros(1, pad)], dim=1))
    return {
        "input_ids": torch.cat(input_ids, dim=0),                                 # [B, L_vlm]
        "attention_mask": torch.cat(attention_mask, dim=0),                       # [B, L_vlm]
        "pixel_values": torch.cat([x["pixel_values"] for x in vlm_inputs], 0),    # [sum N_patch, 1536]
        "image_grid_thw": torch.cat([x["image_grid_thw"] for x in vlm_inputs], 0),# [B, 3]
    }



class PretrainedVLM(nn.Module):
    def __init__(self, checkpoint: str = VLM_CHECKPOINT, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint, dtype=dtype, device_map=device, trust_remote_code=True
        )
        # Frozen
        self.vlm_model.eval()
        for p in self.vlm_model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.vlm_model.config.text_config.hidden_size  # 2B: 2048

    def _inputs_to_embeds(self, vlm_inputs: Dict[str, torch.Tensor]):
        """원본: models/motus.py:340 _process_vlm_inputs_to_tokens"""
        input_ids = vlm_inputs["input_ids"].to(self.device)
        attention_mask = vlm_inputs["attention_mask"].to(self.device)
        pixel_values = vlm_inputs["pixel_values"].to(self.device)
        image_grid_thw = vlm_inputs["image_grid_thw"].to(self.device)

        # (1) 텍스트 토큰 임베딩 lookup: [B, L, 2048]  (이미지 자리는 아직 <|image_pad|> 임베딩)
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids)

        # (2) vision encoder: 이미지 패치 -> 이미지 토큰 임베딩 (+ DeepStack 중간층 특징)
        image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)  # [sum N_img_tok, 2048]

        # (3) <|image_pad|> 자리에 이미지 임베딩을 끼워넣기
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]  # [B, L] 이미지 토큰 위치

        # (4) M-RoPE position ids: [3, B, L]  (t, h, w 3축)
        position_ids, _ = self.vlm_model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids

    @torch.no_grad()
    def forward(self, vlm_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = \
            self._inputs_to_embeds(vlm_inputs)

    
        vlm_output = self.vlm_model.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
     
        return vlm_output.hidden_states[-1]  # [B, L_vlm, 2048]


def describe_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, processor) -> str:
    """한 샘플의 토큰 구성을 요약: 이미지 토큰 / 텍스트(템플릿+지시문+assistant 프롬프트) / padding."""
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_total = input_ids.numel()
    n_valid = int(attention_mask.sum())
    n_img = int((input_ids == image_pad_id).sum())
    return f"total={n_total}  image={n_img}  text={n_valid - n_img}  pad={n_total - n_valid}"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(VLM_CHECKPOINT)
    vlm = PretrainedVLM(VLM_CHECKPOINT, device=device)

    # Dummy 현재 관측 이미지 (robotwin 설정: H=384, W=320) + 길이가 다른 지시문 2개 -> B=2
    rng = np.random.default_rng(0)
    dummy_images = [Image.fromarray(rng.integers(0, 256, (384, 320, 3), dtype=np.uint8)) for _ in range(2)]
    dummy_instructions = [
        "Pick up the bottle and place it on the plate.",
        "Grab the red block.",
    ]

    per_sample = [preprocess_vlm_messages(t, img, processor) for t, img in zip(dummy_instructions, dummy_images)]
    batch = collate_vlm_inputs(per_sample)

    print("=== VLM 입력 ===")
    for k, v in batch.items():
        print(f"{k:15s} {tuple(v.shape)}  {v.dtype}")
    for b in range(batch["input_ids"].shape[0]):
        print(f"[sample {b}] {describe_tokens(batch['input_ids'][b], batch['attention_mask'][b], processor)}")
    print("[sample 0 decoded]")
    print(processor.tokenizer.decode(per_sample[0]["input_ids"][0]).replace("<|image_pad|>", "")
          + f"   (<|image_pad|> x {int(per_sample[0]['image_grid_thw'].prod() // 4)} 생략)")

    und_input = vlm(batch)

    print("\n=== VLM 출력 (Understanding Expert 입력) ===")
    print(f"last hidden state: {tuple(und_input.shape)}  {und_input.dtype}   # [B, L_vlm, {vlm.hidden_size}]")
    print(f"requires_grad: {und_input.requires_grad}  (frozen)")
