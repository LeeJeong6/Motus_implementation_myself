Motus: A Unified Latent ActionWorld Model

를 내가 직접 구현해보자

![alt text](./images/image.png)

위 그림은 Motus의 main figure이다.

여기서 MOT 전의 Video encoder, action encoder, pretrained-VLM을 구현한 게
~~.py, ~~.py ~~.py 이다.

![alt text](./images/fig3.png)

video encoder의 Latent VAE는 위와 같은 의미를 가진다.

![alt text](./images/latent_actions_in_paper.png)

위 논문에서 설명하는 방법론을 보면

optical flow base representation이라는 말을 쓴다.

실제 robot action label이 없는 인터넷 영상이나 사람의 시연 영상에서도 로봇이 활용할 수 있는 action representation 을 학습하겠다는 것이다.

이 논문은 optical flow를 이용해 영상에서 움직임을 추출하고 이를 14차원의 latent action으로 압축하여 로봇 action space로 연결한다.

optical flow는 연속된 두 프레임 사이에서 각 픽셀이 얼마나, 어느 방향으로 이동했는지를 나타내는 표현이다.

이 정보를 활용하여 실제 로봇의 관절값을 몰라도 영상에서 움직임 패턴을 추출할 수 있다.
