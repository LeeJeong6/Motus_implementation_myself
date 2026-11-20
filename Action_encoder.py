"""
Action_encoder.py는 Motus 모델의 action encoder에 해당하는 모듈을 설계한 스크립트이다.
"""

"""
Action Expert는 원본에서 따로 불러오는 것은 아님.
초기화 하여 from-scratch로 같이 학습함.
"""

"""
입력 : state [B,14], actions [B,16,14], register [B,4,1024] # register는 fine-tuning에만 쓰이고 pretrain시 없음
출력 : action tokens [B,21,1024]
"""

"""
의미론적으로 해석하자면 state tokens은 proprioception이다. 즉 noisy하지 않은 condition token이라는 것
action token은 미래의 행동임. 즉 k-step 후 로봇이 있어야 할 관절 상태이고, 전체는 미래 관절 궤적이다.
따라서 학습중에는 노이즈가 섞인 상태로 들어가며 velocity를 예측하게끔 된다.
"""

"""
추가로 register token은 slot으로 출력에는 영향을 주지않는 창고 역할을 함
"""

"""
1D sin-cos embedding이 함께 들어가며 time-step dependent information이 별도로 추가되지는 않는다
"""

"""
정리하자면 입려은 [현재 state + 미래 chunk]
token마다(register token은 제외) MLP를 통과하여 출력은 [21,1024]이다.
여기서 condition은 현재 state이며 generation은 action chunk를 하며
positional encoding으로 1D sin-cos(덧셈)이 추가되며
from-scratch로 학습된다.
"""

import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# robotwin 설정 (configs/robotwin.yaml)
STATE_DIM = 14
ACTION_DIM = 14
NUM_VIDEO_FRAMES = 8
VIDEO_ACTION_FREQ_RATIO = 2
GLOBAL_DOWNSAMPLE_RATE = 3
ACTION_CHUNK_SIZE = NUM_VIDEO_FRAMES * VIDEO_ACTION_FREQ_RATIO  # 16  (원본: MotusConfig.__post_init__)
ACTION_EXPERT_DIM = 1024
NUM_REGISTERS = 4  # finetune 모드 (pretrain 모드는 0)


# ---------------------------------------------------------------------------
# 1. 데이터에서 state / action 뽑기
#    원본: data/robotwin2/robotwin_agilex_dataset.py:405 _calculate_sampling_indices
#          + :301 _load_robot_data
#    state  = qpos[t]                (현재, 조건 프레임과 같은 시점)
#    action = qpos[t + 3k], k=1..16  (미래 절대 관절 위치)
# ---------------------------------------------------------------------------
def calculate_sampling_indices(total_frames: int,
                               action_chunk_size: int = ACTION_CHUNK_SIZE,
                               global_downsample_rate: int = GLOBAL_DOWNSAMPLE_RATE,
                               num_video_frames: int = NUM_VIDEO_FRAMES,
                               video_action_freq_ratio: int = VIDEO_ACTION_FREQ_RATIO) -> Tuple[int, List[int], List[int]]:
    physical_chunk_size = action_chunk_size * global_downsample_rate
    max_condition_idx = total_frames - physical_chunk_size - 1
    condition_frame_idx = 0 if max_condition_idx < 0 else random.randint(0, max_condition_idx)

    action_indices = [min(condition_frame_idx + (i + 1) * global_downsample_rate, total_frames - 1)
                      for i in range(action_chunk_size)]
    video_indices = []
    for i in range(num_video_frames):
        step = (i + 1) * video_action_freq_ratio - 1
        video_indices.append(action_indices[step] if step < len(action_indices) else action_indices[-1])
    return condition_frame_idx, video_indices, action_indices


def load_state_action(qpos: torch.Tensor, condition_frame_idx: int, action_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    initial_state = qpos[min(condition_frame_idx, len(qpos) - 1)].float()  # [14]
    action_sequence = torch.stack([qpos[i] for i in action_indices]).float()  # [16, 14]
    return initial_state, action_sequence


# ---------------------------------------------------------------------------
# 2. Flow-Matching 노이즈 (학습 시)
#    원본: models/motus.py:662-672 FlowMatchScheduler(shift=5.0, sigma_min=0.0,
#          extra_one_step=True, num_train_timesteps=1000).set_timesteps(1000, training=True)
#          + models/motus.py:825-832
#    sigma 테이블만 필요하므로 bak/wan/utils/fm.py 의 계산을 그대로 옮겼다.
# ---------------------------------------------------------------------------
class FlowMatchSigmas:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, sigma_min: float = 0.0):
        self.num_train_timesteps = num_train_timesteps
        sigmas = torch.linspace(1.0, sigma_min, num_train_timesteps + 1)[:-1]  # extra_one_step=True
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)               # shift 적용 (노이즈 많은 쪽으로 치우침)
        self.timesteps = self.sigmas * num_train_timesteps                      # time embedding 용 (0~1000)


def add_action_noise(actions: torch.Tensor, scheduler: FlowMatchSigmas) -> Dict[str, torch.Tensor]:
    """noisy = clean·(1−σ) + noise·σ,  target(velocity) = noise − clean"""
    B = actions.shape[0]
    timestep_id = torch.randint(0, scheduler.num_train_timesteps, (B,))
    t_embed = scheduler.timesteps[timestep_id].to(actions.device, actions.dtype)           # [B]  -> 이후 AdaLN 에서 사용
    sigma = scheduler.sigmas[timestep_id].to(actions.device, actions.dtype).view(B, 1, 1)  # [B,1,1] 샘플마다 1개
    noise = torch.randn_like(actions)
    noisy_actions = actions * (1 - sigma) + noise * sigma
    return {"noisy_actions": noisy_actions, "target": noise - actions, "sigma": sigma, "t_embed": t_embed}


# ---------------------------------------------------------------------------
# 3. Action Encoder
#    원본: models/action_expert.py:89 StateActionEncoder (finetune 모드)
#          registers 는 원본에서 ActionExpert(:345) 가 들고 있다가 encoder 에 넘겨준다.
# ---------------------------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> torch.Tensor:
    """원본: models/action_expert.py:28  (고정, 학습되지 않음)"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return torch.from_numpy(np.concatenate([np.sin(out), np.cos(out)], axis=1)).float()  # [M, D]


def build_mlp(projector_type: str, in_features: int, out_features: int) -> nn.Module:
    """'mlp3x_silu' -> Linear(in,out) → SiLU → Linear(out,out) → SiLU → Linear(out,out)"""
    if projector_type == 'linear':
        return nn.Linear(in_features, out_features)
    m = re.match(r'^mlp(\d+)x_silu$', projector_type)
    if m is None:
        raise ValueError(f'Unknown projector type: {projector_type}')
    modules = [nn.Linear(in_features, out_features)]
    for _ in range(1, int(m.group(1))):
        modules += [nn.SiLU(), nn.Linear(out_features, out_features)]
    return nn.Sequential(*modules)


class StateActionEncoder(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, dim: int = ACTION_EXPERT_DIM,
                 action_chunk_size: int = ACTION_CHUNK_SIZE, num_registers: int = NUM_REGISTERS):
        super().__init__()
        """
        state encoder와 action encoder는 의미는 같지만 현재 vs 미래를 표현하는 차이이다.
        state는 항상 깨끗한 실제 관절값으로 입력 분포가 action과 다르다. 
        즉 state는 proprioception이고 actioon은 생성 대상인거임. 
        """
        self.state_encoder = build_mlp('mlp3x_silu', state_dim, dim)
        self.action_encoder = build_mlp('mlp3x_silu', action_dim, dim)

        max_seq_len = (action_chunk_size + 1) + 1 + num_registers
        self.register_buffer('pos_embedding', get_1d_sincos_pos_embed_from_grid(dim, np.arange(max_seq_len)).unsqueeze(0))

    def forward(self, state_tokens: torch.Tensor, action_tokens: torch.Tensor,
                registers: Optional[torch.Tensor] = None) -> torch.Tensor:
        state_encoded = self.state_encoder(state_tokens)     # [B, 1, dim]
        action_encoded = self.action_encoder(action_tokens)  # [B, 16, dim]
        encoded = torch.cat([state_encoded, action_encoded], dim=1)  # [B, 17, dim]
        if registers is not None:
            encoded = torch.cat([encoded, registers], dim=1)          # [B, 21, dim]
        return encoded + self.pos_embedding[:, :encoded.shape[1], :]


class ActionInput(nn.Module):
    """
    Action Expert 블록에 들어가기 직전까지:
      (학습) GT action 에 노이즈 섞기  /  (추론) 순수 노이즈에서 시작
      → [state | noisy action chunk | registers] 인코딩
    """

    def __init__(self, dim: int = ACTION_EXPERT_DIM, num_registers: int = NUM_REGISTERS,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.input_encoder = StateActionEncoder(dim=dim, num_registers=num_registers)
        # 원본: models/action_expert.py:345-350
        self.registers = nn.Parameter(torch.empty(1, num_registers, dim).normal_(std=0.02)) if num_registers > 0 else None
        self.scheduler = FlowMatchSigmas()
        self._initialize_weights()
        self.to(device=self.device, dtype=dtype)  # 원본: motus.py:614 action_expert.to(bf16)

    def _initialize_weights(self):
        """원본: models/action_expert.py:362 initialize_weights (Linear -> xavier_uniform, bias 0)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode(self, state: torch.Tensor, noisy_actions: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        registers = self.registers.expand(B, -1, -1) if self.registers is not None else None  # [B, 4, dim]
        state_tokens = state.unsqueeze(1).to(self.dtype)                                      # [B, 1, 14]
        return self.input_encoder(state_tokens, noisy_actions.to(self.dtype), registers)

    def forward_train(self, state: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """원본: models/motus.py:825-843"""
        state = state.to(self.device, self.dtype)
        actions = actions.to(self.device, self.dtype)
        noised = add_action_noise(actions, self.scheduler)
        noised["action_tokens"] = self._encode(state, noised["noisy_actions"])  # [B, 21, 1024]
        return noised

    def forward_inference(self, state: torch.Tensor, action_latent: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """원본: models/motus.py:950-951, 972-975  (denoising 루프의 한 스텝 입력)"""
        state = state.to(self.device, self.dtype)
        B = state.shape[0]
        if action_latent is None:  # t=1: 순수 노이즈에서 시작
            action_latent = torch.randn(B, ACTION_CHUNK_SIZE, ACTION_DIM, device=self.device, dtype=self.dtype)
        return {"action_latent": action_latent, "action_tokens": self._encode(state, action_latent)}


# ---------------------------------------------------------------------------
# 4. Dummy input 으로 forward 확인
# ---------------------------------------------------------------------------
def describe_layout(num_actions: int = ACTION_CHUNK_SIZE, num_registers: int = NUM_REGISTERS) -> str:
    return (f"[0]=state (조건, 깨끗함) | [1:{1 + num_actions}]=action chunk (노이즈) | "
            f"[{1 + num_actions}:{1 + num_actions + num_registers}]=registers (학습 가능한 slot)")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy 에피소드 2개: 관절 궤적 qpos [T, 14] (T=200 프레임, 부드러운 sin 궤적)
    B, T = 2, 200
    time_axis = torch.linspace(0, 2 * np.pi, T).unsqueeze(1)
    dummy_qpos = [torch.sin(time_axis * (b + 1) + torch.arange(STATE_DIM) * 0.3) for b in range(B)]

    states, actions = [], []
    for b in range(B):
        cond_idx, video_idx, action_idx = calculate_sampling_indices(T)
        s, a = load_state_action(dummy_qpos[b], cond_idx, action_idx)
        states.append(s)
        actions.append(a)
        print(f"[sample {b}] condition t={cond_idx}  action idx={action_idx[:3]}...{action_idx[-1]}  "
              f"video idx={video_idx}")
    state = torch.stack(states)      # [B, 14]
    actions = torch.stack(actions)   # [B, 16, 14]

    enc = ActionInput(device=device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"\nActionInput 파라미터: {n_params / 1e6:.2f}M  (from scratch)")

    print("\n=== Action encoder 입력 ===")
    print(f"state          {tuple(state.shape)}")
    print(f"actions (GT)   {tuple(actions.shape)}")
    print(f"registers      {tuple(enc.registers.expand(B, -1, -1).shape)}")

    out = enc.forward_train(state, actions)
    print("\n=== 학습: 노이즈 섞기 ===")
    print(f"sigma          {[round(float(s), 3) for s in out['sigma'].flatten()]}   (샘플마다 1개)")
    print(f"t_embed        {[round(float(t), 1) for t in out['t_embed']]}   (AdaLN 에서 사용, encoder 에는 안 들어감)")
    print(f"noisy_actions  {tuple(out['noisy_actions'].shape)}")
    print(f"target(v)      {tuple(out['target'].shape)}   = noise − clean")

    print("\n=== Action encoder 출력 (Action Expert 입력) ===")
    print(f"action_tokens  {tuple(out['action_tokens'].shape)}  {out['action_tokens'].dtype}   # [B, 1+16+4, 1024]")
    print(f"layout         {describe_layout()}")
    print(f"requires_grad  {out['action_tokens'].requires_grad}  (학습 대상)")

    out_inf = enc.forward_inference(state)
    print("\n=== 추론: 순수 노이즈에서 시작 ===")
    print(f"action_latent  {tuple(out_inf['action_latent'].shape)}")
    print(f"action_tokens  {tuple(out_inf['action_tokens'].shape)}")
