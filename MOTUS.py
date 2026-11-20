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

"""
dtype 정책 (11GB GPU 기준)
  - video tower 블록(가장 큼)만 bf16 으로 두고, 나머지 파라미터는 float32
  - forward 는 torch.autocast(bf16) 안에서 수행, timestep 임베딩 / AdaLN 계산은 float32 (원본과 같음)
VLM 은 frozen 이므로 VLM.py 의 출력(last hidden state [B, L_u, 2048])을 입력으로 받는다.
VAE 도 frozen 이므로 검증 단계에서는 VAE latent 를 직접 입력으로 받는다.
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Action_encoder import (ACTION_CHUNK_SIZE, ACTION_DIM, STATE_DIM, ActionInput,
                            calculate_sampling_indices, load_state_action)
from joint_attention import build_rope_freqs, joint_attention
from Text_encoder import TextEmbedding, make_dummy_t5, pad_t5_embeddings
from towers.action_tower import ActionTower
from towers.und_tower import UndTower
from towers.video_tower import VideoTower
from Video_encoder import NUM_VIDEO_FRAMES, VideoInput


@dataclass
class MotusConfig:
    device: str = "cuda"
    num_layers: int = 5                 # 검증용 (원본 30)
    # video tower (WAN2.2-5B)
    video_dim: int = 3072
    video_ffn_dim: int = 14336
    num_heads: int = 24                 # WAN head space: 24 × 128
    # action tower
    action_expert_dim: int = 1024
    action_ffn_mult: int = 4
    num_registers: int = 4
    # und tower
    und_dim: int = 512
    und_ffn_mult: int = 4
    vlm_dim: int = 2048
    # loss
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    # dtype
    video_blocks_dtype: torch.dtype = torch.bfloat16


class Motus(nn.Module):
    def __init__(self, cfg: MotusConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # ---- 토큰화 / 조건 (pretrained 미사용) ----
        self.video_input = VideoInput(device=cfg.device, dtype=torch.float32, load_pretrained=False)
        self.action_input = ActionInput(dim=cfg.action_expert_dim, num_registers=cfg.num_registers,
                                        device=cfg.device, dtype=torch.float32)
        self.text_encoder = TextEmbedding(dim=cfg.video_dim)

        # ---- 세 tower ----
        self.video_tower = VideoTower(dim=cfg.video_dim, ffn_dim=cfg.video_ffn_dim, num_heads=cfg.num_heads,
                                      num_layers=cfg.num_layers)
        self.action_tower = ActionTower(dim=cfg.action_expert_dim, ffn_dim=cfg.action_expert_dim * cfg.action_ffn_mult,
                                        num_layers=cfg.num_layers, action_dim=ACTION_DIM, wan_num_heads=cfg.num_heads)
        self.und_tower = UndTower(dim=cfg.und_dim, ffn_dim=cfg.und_dim * cfg.und_ffn_mult, num_layers=cfg.num_layers,
                                  vlm_input_dim=cfg.vlm_dim, wan_num_heads=cfg.num_heads)

        self.to(self.device)
        self.video_tower.blocks.to(cfg.video_blocks_dtype)
        # RoPE 계수 (복소수라 buffer 로 등록하면 .to(dtype) 에 망가지므로 일반 속성으로 둔다 - 원본과 같음)
        self.freqs = build_rope_freqs(cfg.video_dim // cfg.num_heads).to(self.device)

    # -----------------------------------------------------------------------
    # 레이어 루프 (training / inference 공용)
    # -----------------------------------------------------------------------
    def forward_layers(self, v, a, u, v_e0, a_e0, t5_context, grid_sizes):
        vt, at, ut = self.video_tower, self.action_tower, self.und_tower
        for i in range(self.cfg.num_layers):
            # ① AdaLN modulation (video / action)
            vm, am = vt.modulation(v_e0, i), at.modulation(a_e0, i)
            # ② 각 tower 가 WAN head space 로 q, k, v 사영
            # ③ trimodal joint attention  [video | action | und]
            o_v, o_a, o_u = joint_attention(vt.qkv(v, i, vm), [at.qkv(a, i, am), ut.qkv(u, i)], grid_sizes, self.freqs)
            # ④ 출력 사영 + residual
            v, a, u = vt.attn_out(v, o_v, i, vm), at.attn_out(a, o_a, i, am), ut.attn_out(u, o_u, i)
            # ⑤ T5 cross-attention (video 만)
            v = vt.cross_attn(v, t5_context, i)
            # ⑥ FFN
            v, a, u = vt.ffn(v, i, vm), at.ffn(a, i, am), ut.ffn(u, i)
        return v, a, u

    def predict_velocity(self, video_tokens, action_tokens, vlm_hidden, t5_context, t_video, t_action, grid_sizes):
        """토큰 + timestep -> (video velocity [B,48,T',H',W'], action velocity [B,16,14])"""
        u = self.und_tower.adapt(vlm_hidden.to(self.device))
        v_e, v_e0 = self.video_tower.time_embedding_fn(t_video.float(), video_tokens.shape[1])
        a_e, a_e0 = self.action_tower.time_embedding_fn(t_action.float(), action_tokens.shape[1])

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            v, a, _ = self.forward_layers(video_tokens, action_tokens, u, v_e0, a_e0, t5_context, grid_sizes)
            video_vel = self.video_tower.unpatchify(self.video_tower.head_out(v, v_e), grid_sizes)
            action_full = self.action_tower.head_out(a, a_e)                       # [B, 1+16+4, 14]
        action_vel = action_full[:, 1:1 + ACTION_CHUNK_SIZE].float()               # state / register 제거
        return video_vel.float(), action_vel

    # -----------------------------------------------------------------------
    # 학습
    # -----------------------------------------------------------------------
    def training_step(self, clean_latent, condition_latent, state, actions, t5, vlm_hidden) -> Dict[str, torch.Tensor]:
        # 1. 토큰화 (노이즈 섞기 포함)
        vid = self.video_input.forward_train_from_latent(clean_latent, condition_latent)
        act = self.action_input.forward_train(state, actions)
        t5_context = self.text_encoder(t5)

        # 2~4. time embedding + 레이어 루프 + head
        video_vel, action_vel = self.predict_velocity(vid["video_tokens"], act["action_tokens"], vlm_hidden,
                                                      t5_context, vid["t_embed"], act["t_embed"], vid["grid_sizes"])
        # 5. loss
        video_pred = video_vel.clone()
        video_pred[:, :, 0:1] = 0                                                   # 0번 latent 제외
        video_loss = F.mse_loss(video_pred, vid["target"].float())
        action_loss = F.mse_loss(action_vel, act["target"].float())
        total = self.cfg.video_loss_weight * video_loss + self.cfg.action_loss_weight * action_loss
        return {"total_loss": total, "video_loss": video_loss, "action_loss": action_loss}

    # -----------------------------------------------------------------------
    # 추론 (원본: models/motus.py:911 inference_step, Euler)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def inference_step(self, condition_latent, state, t5, vlm_hidden, num_steps: int = 10,
                       generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0]
        t5_context = self.text_encoder(t5)
        condition_latent = condition_latent.to(self.device).float()
        _, C, _, H, W = condition_latent.shape
        T = 1 + NUM_VIDEO_FRAMES // 4
        video_latent = torch.randn(B, C, T, H, W, device=self.device, generator=generator)
        video_latent[:, :, 0:1] = condition_latent
        action_latent = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, device=self.device, generator=generator)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        for i in range(num_steps):
            t, dt = timesteps[i], timesteps[i + 1] - timesteps[i]
            vid = self.video_input.forward_inference_from_latent(condition_latent, video_latent)
            act = self.action_input.forward_inference(state, action_latent)
            t_scaled = (t * 1000).expand(B)
            video_vel, action_vel = self.predict_velocity(vid["video_tokens"], act["action_tokens"], vlm_hidden,
                                                          t5_context, t_scaled, t_scaled, vid["grid_sizes"])
            video_latent = video_latent + video_vel * dt
            action_latent = action_latent + action_vel * dt
            video_latent[:, :, 0:1] = condition_latent                                # teacher forcing
        return video_latent, action_latent


# ---------------------------------------------------------------------------
# Dummy data 로 확인
#   pretrained 가 없으므로 "처음부터 정답을 맞추는지"는 의미가 없다.
#   대신 고정된 dummy 배치에 overfit 시킨 뒤, 노이즈에서 denoising 해서 GT action 을 복원하는지 본다.
# ---------------------------------------------------------------------------
def make_dummy_batch(B: int = 2, L_u: int = 141, seed: int = 0) -> Dict[str, torch.Tensor]:
    random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    T = 200
    time_axis = torch.linspace(0, 2 * np.pi, T).unsqueeze(1)
    states, actions = [], []
    for b in range(B):  # 샘플마다 다른 관절 궤적
        qpos = torch.sin(time_axis * (b + 1) + torch.arange(STATE_DIM) * 0.3 + b)
        cond_idx, _, action_idx = calculate_sampling_indices(T)
        s, a = load_state_action(qpos, cond_idx, action_idx)
        states.append(s)
        actions.append(a)
    clean_latent = torch.randn(B, 48, 1 + NUM_VIDEO_FRAMES // 4, 24, 20, generator=g)
    return {
        "clean_latent": clean_latent,                         # VAE latent 대신
        "condition_latent": clean_latent[:, :, 0:1].clone(),  # causal VAE: 0번 latent = 현재 프레임 latent
        "state": torch.stack(states),                         # [B, 14]
        "actions": torch.stack(actions),                      # [B, 16, 14]
        "t5": pad_t5_embeddings(make_dummy_t5(B, seed=seed)), # umT5 출력 대신 [B, 512, 4096]
        "vlm_hidden": torch.randn(B, L_u, 2048, generator=g), # VLM last hidden state 대신
    }


def swap_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """조건(이미지 latent / state / 지시문 / VLM)을 샘플끼리 맞바꾼 배치"""
    return {k: v.flip(0) for k, v in batch.items()}


def action_mse(model: Motus, batch, seed: int = 123, num_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=model.device).manual_seed(seed)
    _, pred = model.inference_step(batch["condition_latent"], batch["state"], batch["t5"], batch["vlm_hidden"],
                                   num_steps=num_steps, generator=gen)
    gt = batch["actions"].to(model.device)
    return ((pred - gt) ** 2).mean(dim=(1, 2)), pred


def count(m: nn.Module, trainable_only: bool = False) -> float:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only) / 1e6


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train_video_blocks", action="store_true",
                        help="video tower 블록까지 학습 (11GB GPU 에서는 메모리 부족 가능)")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Motus(MotusConfig(device=device))
    if not args.train_video_blocks:
        model.video_tower.blocks.requires_grad_(False)

    print("=== 파라미터 (M) ===")
    for name in ["video_input", "action_input", "text_encoder", "video_tower", "action_tower", "und_tower"]:
        m = getattr(model, name)
        print(f"{name:14s} total {count(m):7.1f}   trainable {count(m, True):7.1f}")

    batch = make_dummy_batch()
    print("\n=== 입력 (dummy) ===")
    for k, v in batch.items():
        print(f"{k:17s} {tuple(v.shape)}")

    # ---- 1. shape 확인 ----
    vid = model.video_input.forward_train_from_latent(batch["clean_latent"], batch["condition_latent"])
    act = model.action_input.forward_train(batch["state"], batch["actions"])
    with torch.no_grad():
        v_vel, a_vel = model.predict_velocity(vid["video_tokens"], act["action_tokens"], batch["vlm_hidden"],
                                              model.text_encoder(batch["t5"]), vid["t_embed"], act["t_embed"],
                                              vid["grid_sizes"])
    print("\n=== 토큰 -> tower -> velocity ===")
    print(f"video tokens  {tuple(vid['video_tokens'].shape)} -> video velocity  {tuple(v_vel.shape)}")
    print(f"action tokens {tuple(act['action_tokens'].shape)} -> action velocity {tuple(a_vel.shape)}")
    print(f"und input     {tuple(batch['vlm_hidden'].shape)} -> adapter -> [B, L_u, 512] (출력 없음)")

    # ---- 2. 학습 전 추론 ----
    mse_before, _ = action_mse(model, batch)
    print(f"\n학습 전 action MSE (샘플별): {[round(x, 4) for x in mse_before.tolist()]}")

    # ---- 3. dummy 배치에 overfit ----
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    print(f"\n=== 학습 ({args.steps} steps, lr {args.lr}, 학습 파라미터 {sum(p.numel() for p in params) / 1e6:.1f}M) ===")
    history = []
    for step in range(1, args.steps + 1):
        out = model.training_step(**batch)
        opt.zero_grad(set_to_none=True)
        out["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history.append([out["video_loss"].item(), out["action_loss"].item()])
        if step == 1 or step % 50 == 0:
            h = np.array(history[-50:])
            print(f"step {step:4d}  video_loss {h[:, 0].mean():.4f}  action_loss {h[:, 1].mean():.4f}")
    if device == "cuda":
        print(f"최대 GPU 메모리: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ---- 4. 학습 후 추론: 노이즈에서 action 복원 ----
    mse_after, pred = action_mse(model, batch)
    gt = batch["actions"].to(device)
    mean_baseline = ((gt.mean(0, keepdim=True) - gt) ** 2).mean(dim=(1, 2))
    mse_swapped, _ = action_mse(model, swap_batch(batch))
    mse_swapped_vs_own = ((_ - gt.flip(0)) ** 2).mean(dim=(1, 2))  # 바꾼 조건의 GT 와 비교

    print("\n=== 추론 결과 (10 Euler steps, 순수 노이즈에서 시작) ===")
    print(f"학습 전 action MSE                  : {[round(x, 4) for x in mse_before.tolist()]}")
    print(f"학습 후 action MSE                  : {[round(x, 4) for x in mse_after.tolist()]}")
    print(f"참고) 두 GT 의 평균을 예측할 때 MSE  : {[round(x, 4) for x in mean_baseline.tolist()]}")
    print(f"조건을 맞바꾼 뒤 → 바뀐 조건의 GT 와  : {[round(x, 4) for x in mse_swapped_vs_own.tolist()]}  (조건을 따라가면 작음)")

    j = 0
    print(f"\n[sample 0, joint {j}] 시간축 action (GT vs 예측)")
    print("GT  :", " ".join(f"{x:+.2f}" for x in gt[0, :, j].tolist()))
    print("Pred:", " ".join(f"{x:+.2f}" for x in pred[0, :, j].tolist()))

    # ---- 5. 그림 저장 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    h = np.array(history)
    axes[0].plot(h[:, 0], alpha=.4, label="video_loss")
    axes[0].plot(h[:, 1], alpha=.4, label="action_loss")
    axes[0].set_yscale("log"); axes[0].set_xlabel("step"); axes[0].legend(); axes[0].set_title("학습 loss")
    for b in range(2):
        ax = axes[b + 1]
        for jj, c in zip([0, 4, 9], ["C0", "C1", "C2"]):
            ax.plot(gt[b, :, jj].cpu(), color=c, lw=2, label=f"GT joint {jj}")
            ax.plot(pred[b, :, jj].cpu(), color=c, ls="--", marker="o", ms=3, label=f"pred joint {jj}")
        ax.set_title(f"sample {b}: action chunk (16 steps)"); ax.set_xlabel("k (미래 step)")
        if b == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "motus_action_overfit.png", dpi=110)
    print(f"\nsaved: {out_dir / 'motus_action_overfit.png'}")
