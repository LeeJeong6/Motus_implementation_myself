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


