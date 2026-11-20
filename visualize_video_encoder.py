"""
Video_encoder.py 의 VAE + flow-matching noise 를 눈으로 확인하는 스크립트.

  (1) 입력 비디오 [1+F 프레임]
  (2) VAE encode -> decode (σ=0, 재구성)
  (3) σ 별로 미래 latent 에만 노이즈를 섞고 (0번 latent 는 teacher forcing) decode
  (4) 48채널 latent 를 PCA 로 3채널(RGB)로 줄여서 latent 자체도 시각화

사용법:
  python visualize_vae_noise.py                       # 합성 장면
  python visualize_vae_noise.py --image my.jpg        # 이미지 1장 -> 패닝으로 가짜 비디오 생성
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F_
from PIL import Image

from Video_encoder import NUM_VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, VideoInput

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Serif CJK KR", "DejaVu Sans"]  # 한글 표시
plt.rcParams["axes.unicode_minus"] = False

SIGMAS = [0.25, 0.5, 0.75, 1.0]
OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# 입력 비디오 만들기  ->  [1+F, 3, H, W] in [0,1]
# ---------------------------------------------------------------------------
def make_scene_video(T: int, H: int, W: int) -> torch.Tensor:
    """나무 테이블 + 고정된 컵/접시 + 오른쪽으로 움직이며 내려오는 '그리퍼' 사각형."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    frames = []
    for t in range(T):
        img = torch.stack([0.55 + 0.1 * torch.sin(40 * ys), 0.38 + 0.05 * torch.sin(40 * ys), 0.22 * torch.ones_like(xs)])
        img[:, : H // 3] = torch.tensor([0.75, 0.8, 0.85]).view(3, 1, 1)                     # 벽
        plate = ((xs - 0.7) ** 2 / 0.02 + (ys - 0.7) ** 2 / 0.006) < 1
        img[:, plate] = torch.tensor([0.95, 0.95, 0.95]).view(3, 1)                           # 접시
        cup = ((xs - 0.3) ** 2 + (ys - 0.62) ** 2) < 0.004
        img[:, cup] = torch.tensor([0.1, 0.4, 0.9]).view(3, 1)                                # 파란 컵
        cx, cy = 0.2 + 0.06 * t, 0.25 + 0.03 * t                                             # 움직이는 물체
        grip = ((xs - cx).abs() < 0.08) & ((ys - cy).abs() < 0.05)
        img[:, grip] = torch.tensor([0.9, 0.15, 0.15]).view(3, 1)
        frames.append(img.clamp(0, 1))
    return torch.stack(frames)


def make_video_from_image(path: str, T: int, H: int, W: int) -> torch.Tensor:
    """이미지 1장을 조금 크게 resize 한 뒤 crop 창을 옆으로 밀어서 카메라 패닝 비디오를 만든다."""
    img = Image.open(path).convert("RGB")
    scale = 1.25
    img = img.resize((int(W * scale), int(H * scale)))
    arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    max_dx, y0 = arr.shape[2] - W, (arr.shape[1] - H) // 2
    return torch.stack([arr[:, y0:y0 + H, int(max_dx * t / (T - 1)):int(max_dx * t / (T - 1)) + W] for t in range(T)])


# ---------------------------------------------------------------------------
# VAE decode / latent PCA
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode(enc: VideoInput, latent: torch.Tensor) -> torch.Tensor:
    """[1,48,T',H',W'] -> [1+F, 3, H, W] in [0,1]"""
    vae_param = next(enc.vae.model.parameters())
    pixels = enc.vae.decode([latent[0].to(vae_param.device, vae_param.dtype)])[0]   # [3, T, H, W] in [-1,1]
    return ((pixels.float().cpu() + 1) / 2).clamp(0, 1).permute(1, 0, 2, 3)


def fit_pca(latent: torch.Tensor):
    """clean latent 기준으로 48채널 -> 3채널 PCA 기저를 구한다 (모든 σ 에 같은 기저 사용)."""
    x = latent[0].float().flatten(1).T.cpu()                    # [N, 48]
    mean = x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x - mean, q=3, center=False)
    proj = (x - mean) @ v
    lo, hi = proj.quantile(0.01, dim=0), proj.quantile(0.99, dim=0)
    return mean, v, lo, hi


def latent_to_rgb(latent: torch.Tensor, pca, out_hw) -> torch.Tensor:
    """[1,48,T',H',W'] -> [T', 3, H, W]  (시각화를 위해 nearest 업샘플)"""
    mean, v, lo, hi = pca
    _, C, T, h, w = latent.shape
    x = latent[0].float().flatten(1).T.cpu()
    rgb = (((x - mean) @ v - lo) / (hi - lo)).clamp(0, 1).T.reshape(3, T, h, w).permute(1, 0, 2, 3)
    return F_.interpolate(rgb, size=out_hw, mode="nearest")


def noisy_latent_at(clean: torch.Tensor, cond: torch.Tensor, sigma: float, noise: torch.Tensor) -> torch.Tensor:
    """Video_encoder.forward_train 과 같은 식: clean·(1−σ) + noise·σ, 0번 latent 는 teacher forcing"""
    noisy = clean * (1 - sigma) + noise * sigma
    noisy[:, :, 0:1] = cond
    return noisy


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="이미지 경로 (없으면 합성 장면)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    T, H, W = 1 + NUM_VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH
    video = make_video_from_image(args.image, T, H, W) if args.image else make_scene_video(T, H, W)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VideoInput(device=device)

    # Video_encoder 와 같은 전처리: [0,1] -> [-1,1], [1,3,T,H,W]
    full_video = (video * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    clean = enc.encode_video(full_video).float()                           # [1,48,3,24,20]
    cond = enc.encode_video(full_video[:, :, 0:1]).float()                 # [1,48,1,24,20]
    noise = torch.randn_like(clean)                                        # 모든 σ 에 같은 노이즈 사용
    print(f"video {tuple(full_video.shape)} -> latent {tuple(clean.shape)}")

    rows = [("input", video), ("VAE recon\n(σ=0)", decode(enc, clean))]
    latent_rows = [("latent σ=0", clean)]
    for s in SIGMAS:
        noisy = noisy_latent_at(clean, cond, s, noise)
        rows.append((f"decode\nσ={s}", decode(enc, noisy)))
        latent_rows.append((f"latent σ={s}", noisy))

    recon_psnr = -10 * torch.log10(F_.mse_loss(rows[1][1], video)).item()
    print(f"VAE 재구성 PSNR: {recon_psnr:.2f} dB")

    # 어떤 픽셀 프레임이 어떤 latent 에 속하는지: 0 | 1-4 | 5-8
    latent_idx = [0] + [1 + (f - 1) // 4 for f in range(1, T)]

    OUT_DIR.mkdir(exist_ok=True)

    # --- Figure 1: 픽셀 공간 ---
    fig, axes = plt.subplots(len(rows), T, figsize=(T * 1.6, len(rows) * 2.0))
    for r, (name, frames) in enumerate(rows):
        for c in range(T):
            ax = axes[r, c]
            ax.imshow(frames[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"f{c}" + (" (현재)" if c == 0 else "") + f"\nlatent {latent_idx[c]}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
                for s in ax.spines.values():
                    s.set_edgecolor("tab:green"); s.set_linewidth(2)
    fig.suptitle("Pixel space: 미래 latent 에만 노이즈 (초록 테두리 = teacher forcing 된 0번 latent)", fontsize=10)
    fig.tight_layout()
    p1 = OUT_DIR / "vae_noise_pixels.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # --- Figure 2: latent 공간 (PCA RGB) ---
    pca = fit_pca(clean)
    Tl = clean.shape[2]
    fig, axes = plt.subplots(len(latent_rows), Tl, figsize=(Tl * 2.0, len(latent_rows) * 2.2))
    for r, (name, lat) in enumerate(latent_rows):
        rgb = latent_to_rgb(lat, pca, (H, W))
        for c in range(Tl):
            ax = axes[r, c]
            ax.imshow(rgb[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(["latent 0 (f0)", "latent 1 (f1-4)", "latent 2 (f5-8)"][c], fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle("Latent space [48,3,24,20] → PCA 3채널", fontsize=10)
    fig.tight_layout()
    p2 = OUT_DIR / "vae_noise_latent.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    print(f"saved: {p1}\nsaved: {p2}")


if __name__ == "__main__":
    main()
