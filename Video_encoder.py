"""
Video_encoder.py는 Motus 모델의 Video Encoder에 해당하는 모듈을 설계한 스크립트이다.
"""

"""
Frozen VAE가 픽셀을 latent로 압축하고 WAN의 patch embedding이 latent를 토큰으로 바꾸는 과정이다
"""

"""
먼저 First Frame은 [B,3,H,W] 이다.
미래 GT Video Frame은 [B,F,3,H,W]이다.

이 둘을 concat한 Full Video는 [B,3,1+F,H,W]이다. # H=384, W=320, F=8
Full Video는 Wan2.2 VAE encoder에 의해 [B,48,3,24,20]이 된다.
즉 공간은 16배 압축되는 것이고 시간은 4배(1+8 -> 1+2) 압축되며 channel은 16배 증가하게 된다.

[B,48,3,24,20]에서 0번 latent frame을 제외하고 나머지에 flow-matching noise를 섞는다.(shape변화 X)
그 뒤 patch embedding(Conv3d(48->3072, (1,2,2))를 통과하여 최종적으로 [B,360,3072]가 된다.
"""

"""
Wan 2.2 VAE는 frozen이며 patch embedding은 pretrained WAN모듈을 사용하며 trainable이다.
"""

"""
이 모듈에서는 토큰 하나하나가 시공간 패치이다.
공간 정보는 32x32를 압축하고 있고 시간 정보는 4개 프레임을 압축하고 있음.

0번 latent frame은 현재 관측으로 항상 깨끗한 값으로 쓰이고 loss에서도 제외함.

1번 이후 latent frame은 noisy한 미래 장면으로 모델은 이 부분의 velocity를 예측하게 됨.
"""

"""
inference때는 미래 프레임이 아예 없으므로 미래 프레임은 전부 노이즈로 들어가게 되고 학습때만 F만큼 미래 프레임이 GT로 들어가게 되는 것이다.
"""

"""
학습 때 noise schedular 및 VAE 등의 구현은 실제 원본 코드에서 복사해서 modules/vae2_2.py 등에 구현했다.
noise 값에 따라 vae output, latent 등을 시각화해보기 위해서 visualize_vae_noise.py를 만들었고 이 결과는 outputs폴더 내에서 확인 가능하다.
"""

import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from modules.vae2_2 import Wan2_2_VAE      # Wan2.2 VAE 코드를 프로젝트 안에 복사 (modules/)
from Action_encoder import FlowMatchSigmas  # action 과 같은 scheduler 설정

WAN_CHECKPOINT = "Wan-AI/Wan2.2-TI2V-5B"


NUM_VIDEO_FRAMES = 8
VIDEO_HEIGHT = 384
VIDEO_WIDTH = 320


def resolve_wan_file(filename: str, checkpoint: str = WAN_CHECKPOINT) -> str:
    """로컬 폴더면 그 안의 파일, 아니면 HF Hub 에서 해당 파일만 받아온다 (캐시 재사용)."""
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, filename)
    return hf_hub_download(checkpoint, filename)


# ---------------------------------------------------------------------------
# 1. Patch embedding (pretrained WAN 가중치, trainable)
#    원본: bak/wan/modules/model.py:499  nn.Conv3d(in_dim=48, dim=3072, kernel=stride=(1,2,2))
#    원본은 WanModel 전체를 로드하지만, 여기서는 토큰화에 필요한 patch_embedding 만 로드한다.
# ---------------------------------------------------------------------------
def load_patch_embedding(checkpoint: str = WAN_CHECKPOINT) -> nn.Conv3d:
    config = json.load(open(resolve_wan_file("config.json", checkpoint)))
    patch_embedding = nn.Conv3d(config["in_dim"], config["dim"], kernel_size=(1, 2, 2), stride=(1, 2, 2))

    index = json.load(open(resolve_wan_file("diffusion_pytorch_model.safetensors.index.json", checkpoint)))
    shard = resolve_wan_file(index["weight_map"]["patch_embedding.weight"], checkpoint)
    with safe_open(shard, framework="pt") as f:
        state = {k: f.get_tensor(f"patch_embedding.{k}") for k in ("weight", "bias")}
    patch_embedding.load_state_dict(state)
    return patch_embedding


# ---------------------------------------------------------------------------
# 2. Video Encoder
#    원본: models/motus.py:796-822 (학습), :939-949, :971 (추론)
#          models/wan_model.py:67 encode_video, models/motus.py:108 VideoModule.prepare_input
# ---------------------------------------------------------------------------
class VideoInput(nn.Module):
    def __init__(self, checkpoint: str = WAN_CHECKPOINT, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 num_video_frames: int = NUM_VIDEO_FRAMES, load_pretrained: bool = True):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_video_frames = num_video_frames

        if load_pretrained:
            # Frozen VAE: nn.Module 이 아닌 일반 클래스라 self.parameters() 에 포함되지 않는다 (= 학습/저장 대상 아님)
            self.vae = Wan2_2_VAE(vae_pth=resolve_wan_file("Wan2.2_VAE.pth", checkpoint), device=self.device)
            # Trainable patch embedding (pretrained WAN 가중치)
            self.patch_embedding = load_patch_embedding(checkpoint).to(self.device, dtype)
        else:
            # 검증용: VAE 없이 latent 를 직접 입력받고, patch embedding 은 랜덤 초기화
            self.vae = None
            self.patch_embedding = nn.Conv3d(48, 3072, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            nn.init.xavier_uniform_(self.patch_embedding.weight.view(3072, -1))  # 원본 WanModel.init_weights
            self.patch_embedding.to(self.device, dtype)

        self.scheduler = FlowMatchSigmas()  # 원본: models/motus.py:650 (video 용 별도 인스턴스, 같은 설정)

    # ---- VAE ----
    @torch.no_grad()
    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] in [-1,1] -> [B,48,1+(T-1)/4,H/16,W/16]  (원본: models/wan_model.py:67)"""
        vae_param = next(self.vae.model.parameters())
        return self.vae.encode(video_pixels.to(vae_param.device, vae_param.dtype))

    # ---- latent -> tokens ----
    def prepare_input(self, video_latent: torch.Tensor) -> torch.Tensor:
        """[B,48,T',H',W'] -> Conv3d -> [B,3072,T',H'/2,W'/2] -> [B, L_v, 3072]  (원본: models/motus.py:108)"""
        patched = self.patch_embedding(video_latent.to(self.dtype))
        return patched.flatten(2).transpose(1, 2)

    @staticmethod
    def grid_sizes(video_latent: torch.Tensor) -> torch.Tensor:
        """3D RoPE 에 쓰일 (F, H, W) 토큰 격자 크기  (원본: models/motus.py:622-630)"""
        B, _, T, H, W = video_latent.shape
        return torch.tensor([T, H // 2, W // 2], dtype=torch.long).unsqueeze(0).expand(B, -1)

    # ---- 학습 ----
    def forward_train(self, first_frame: torch.Tensor, video_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        first_frame  [B,3,H,W]   in [0,1]  현재 관측
        video_frames [B,F,3,H,W] in [0,1]  미래 GT
        """
        B = first_frame.shape[0]
        first_frame = first_frame.to(self.device, self.dtype)
        video_frames = video_frames.to(self.device, self.dtype)

        # (1) [0,1] -> [-1,1], 시간축 concat: [B,3,1+F,H,W]
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_norm = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        full_video = torch.cat([first_frame_norm, video_norm], dim=2)

        # (2) Frozen VAE
        clean_latent = self.encode_video(full_video).to(self.dtype)                # [B,48,1+F/4,H/16,W/16]
        condition_latent = self.encode_video(first_frame_norm).to(self.dtype)      # [B,48,1,H/16,W/16]

        # (3) ~ (4)
        return self.forward_train_from_latent(clean_latent, condition_latent)

    def forward_train_from_latent(self, clean_latent: torch.Tensor, condition_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        VAE 이후 단계만: clean latent [B,48,T',H',W'] + condition latent [B,48,1,H',W']
        -> 노이즈 섞기 + teacher forcing + patch embedding
        """
        B = clean_latent.shape[0]
        clean_latent = clean_latent.to(self.device, self.dtype)
        condition_latent = condition_latent.to(self.device, self.dtype)

        # (3) Flow-matching noise (샘플마다 σ 1개)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (B,))
        t_embed = self.scheduler.timesteps[timestep_id].to(self.device, self.dtype)                 # [B] -> AdaLN
        sigma = self.scheduler.sigmas[timestep_id].to(self.device, self.dtype).view(B, 1, 1, 1, 1)
        noise = torch.randn_like(clean_latent)
        noisy_latent = clean_latent * (1 - sigma) + noise * sigma
        noisy_latent[:, :, 0:1] = condition_latent          # teacher forcing: 0번 latent 는 깨끗한 현재 관측
        target = noise - clean_latent                        # velocity
        target[:, :, 0:1] = 0                                # 0번 latent 는 loss 에서 제외

        # (4) patch embedding -> tokens
        video_tokens = self.prepare_input(noisy_latent)      # [B, L_v, 3072]

        return {
            "video_tokens": video_tokens,
            "noisy_latent": noisy_latent,
            "clean_latent": clean_latent,
            "condition_latent": condition_latent,
            "target": target,
            "sigma": sigma,
            "t_embed": t_embed,
            "grid_sizes": self.grid_sizes(noisy_latent),
        }

    # ---- 추론 ----
    def forward_inference(self, first_frame: torch.Tensor, video_latent: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """원본: models/motus.py:939-949 (초기화), :971 (매 스텝 토큰화)"""
        first_frame = first_frame.to(self.device, self.dtype)
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        condition_latent = self.encode_video(first_frame_norm).to(self.dtype)      # [B,48,1,H',W']
        return self.forward_inference_from_latent(condition_latent, video_latent)

    def forward_inference_from_latent(self, condition_latent: torch.Tensor,
                                      video_latent: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """VAE 이후 단계만: 미래 latent 노이즈 초기화(또는 현재 스텝 latent) + teacher forcing + patch embedding"""
        condition_latent = condition_latent.to(self.device, self.dtype)
        if video_latent is not None:
            video_latent = video_latent.to(self.device, self.dtype).clone()
        if video_latent is None:  # t=1: 미래 latent 는 순수 노이즈
            B, C, _, H, W = condition_latent.shape
            T = 1 + self.num_video_frames // 4
            video_latent = torch.randn(B, C, T, H, W, device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_latent           # 0번 latent 는 항상 현재 관측

        return {
            "video_tokens": self.prepare_input(video_latent),
            "video_latent": video_latent,
            "condition_latent": condition_latent,
            "grid_sizes": self.grid_sizes(video_latent),
        }


# ---------------------------------------------------------------------------
# 3. Dummy input 으로 forward 확인
# ---------------------------------------------------------------------------
def make_dummy_video(B: int, F: int, H: int, W: int) -> torch.Tensor:
    """오른쪽으로 움직이는 사각형 + 배경 그라디언트, [B, 1+F, 3, H, W] in [0,1]"""
    ys = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
    xs = torch.linspace(0, 1, W).view(1, W).expand(H, W)
    frames = []
    for b in range(B):
        clip = []
        for f in range(1 + F):
            img = torch.stack([xs, ys, 0.5 * torch.ones_like(xs)])     # 배경
            x0 = 20 + f * 15 + b * 40
            img[:, 150:230, x0:x0 + 80] = torch.tensor([1.0, 0.2, 0.2]).view(3, 1, 1)  # 빨간 사각형
            clip.append(img)
        frames.append(torch.stack(clip))
    return torch.stack(frames)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, F, H, W = 2, NUM_VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH

    enc = VideoInput(device=device)
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_vae = sum(p.numel() for p in enc.vae.model.parameters())
    print(f"patch_embedding (trainable): {n_trainable / 1e6:.2f}M   VAE (frozen, 모듈 밖): {n_vae / 1e6:.1f}M")

    clip = make_dummy_video(B, F, H, W)
    first_frame, video_frames = clip[:, 0], clip[:, 1:]

    print("\n=== Video encoder 입력 ===")
    print(f"first_frame    {tuple(first_frame.shape)}   # 현재 관측")
    print(f"video_frames   {tuple(video_frames.shape)}   # 미래 GT")

    out = enc.forward_train(first_frame, video_frames)
    print("\n=== 학습: VAE + 노이즈 ===")
    print(f"clean_latent   {tuple(out['clean_latent'].shape)}   # [B,48,1+F/4,H/16,W/16]")
    print(f"sigma          {[round(float(s), 3) for s in out['sigma'].flatten()]}   (샘플마다 1개)")
    print(f"t_embed        {[round(float(t), 1) for t in out['t_embed']]}   (AdaLN 에서 사용)")
    print(f"0번 latent == condition latent : {torch.equal(out['noisy_latent'][:, :, 0:1], out['condition_latent'])}")
    print(f"0번 target == 0                : {bool((out['target'][:, :, 0] == 0).all())}")
    print(f"causal VAE 확인: |clean[0] − condition| max = "
          f"{(out['clean_latent'][:, :, 0:1] - out['condition_latent']).abs().max().item():.4f}  (≈0 이면 0번 latent 는 첫 프레임만 봄)")

    print("\n=== Video encoder 출력 (메인 모듈 입력) ===")
    g = out["grid_sizes"][0].tolist()
    print(f"video_tokens   {tuple(out['video_tokens'].shape)}  {out['video_tokens'].dtype}   # [B, {g[0]}×{g[1]}×{g[2]}, 3072]")
    print(f"grid_sizes     {g}   (F, H, W) -> 3D RoPE")
    print(f"layout         tokens[0:{g[1] * g[2]}] = 0번 latent(현재, 깨끗함) | "
          f"tokens[{g[1] * g[2]}:{g[0] * g[1] * g[2]}] = 미래 latent (노이즈)")
    print(f"requires_grad  {out['video_tokens'].requires_grad}  (patch_embedding 학습 대상)")

    out_inf = enc.forward_inference(first_frame)
    print("\n=== 추론: 미래 latent 는 순수 노이즈 ===")
    print(f"video_latent   {tuple(out_inf['video_latent'].shape)}")
    print(f"video_tokens   {tuple(out_inf['video_tokens'].shape)}")
