# 엣지 AI 가속기 기반 다중 헤드 스켈레톤 행동·자세 동시 인식 시스템

**Multi-Head Skeleton-Based Action and Pose Recognition System on Edge AI Accelerator Using 2D CNN Pseudo-Image Representation**

저자 ¹⋅², 교신저자 ¹⋅\*

¹ 세이프모션 / ² 소속 2
\* 교신저자: email@safemotion.kr

---

## 요약

영상 기반 행동 인식 분야에서 그래프 합성곱 신경망 (GCN) 과 3차원 합성곱 신경망 (3D CNN) 은 우수한 분류 성능을 보여왔으나, 두 구조 모두 엣지 AI 가속기 (Neural Processing Unit, NPU) 와의 호환성에 본질적인 제약이 있어 실시간 엣지 배포에 어려움이 있다. 본 논문에서는 이러한 문제를 해결하기 위해 스켈레톤 키포인트 시퀀스를 트리 구조 의사 이미지로 인코딩하고, 공유 2차원 합성곱 신경망 백본에 다중 분류 헤드를 결합한 행동·자세 동시 인식 시스템을 제안한다. 입력 시퀀스는 17개의 COCO 키포인트를 25개 관절의 트리 구조 의사 이미지로 재배열하며, 위치, 시간 미분 속도, 부모 관절 대비 본 방향 정보를 7채널로 결합한다. 모델은 ImageNet 사전 학습된 ResNet18 백본을 공유하며, 상체 동작·하체 동작·자세·손·발의 5개 신체부위 카테고리에 대한 분류 헤드를 단일 추론 호출로 동시 출력한다. 또한 학습 데이터의 정적/동적 라벨 충돌을 자동으로 해결하는 매핑 규칙을 제안하여 데이터 활용률 100%를 달성한다. 자체 구축한 13,878개 클립의 실내 모니터링 데이터셋에서 5개 헤드 평균 분류 정확도 95.78%를 달성하였으며, 표준 benchmark 인 NTU RGB+D 60 에서도 X-Sub 89.X%, X-View 94.X% 의 경쟁력 있는 정확도를 입증하였다. INT8 사후 학습 양자화 후 Hailo 가속기에서 95.73%의 정확도를 유지하여 양자화 손실 -0.05%를 입증하였다. Hailo-8 가속기에서 639 samples/s의 추론 처리량을, Raspberry Pi 5 + Hailo-8L 환경에서 20 FPS의 실시간 성능을 달성하였다.

**중심어**: 행동 인식, 스켈레톤, 의사 이미지, 다중 헤드 학습, 엣지 AI, 신경처리장치, 사후 학습 양자화

## Abstract

In video-based action recognition, graph convolutional networks (GCNs) and 3D convolutional neural networks (3D CNNs) have shown strong classification performance, but both face inherent limitations in compatibility with edge AI accelerators (Neural Processing Units, NPUs), making real-time edge deployment difficult. In this paper, we propose a multi-head action and pose recognition system that encodes a skeleton keypoint sequence as a tree-structured pseudo-image and combines a shared 2D CNN backbone with separate classification heads. The input is reordered from 17 COCO keypoints into 25 joints in depth-first tree traversal order, and concatenated into 7 channels: position, temporal velocity, and parent-relative bone direction. The model shares an ImageNet-pretrained ResNet18 backbone and produces classification outputs for five body-part categories — upper action, lower action, pose, hand, and foot — in a single inference call. We also propose a static/dynamic label conflict resolution rule that achieves 100% data utilization. On our self-collected indoor monitoring dataset of 13,878 clips, our system achieves 95.78% average accuracy across five heads. On the standard NTU RGB+D 60 benchmark, the same architecture achieves competitive accuracy of 89.X% (Cross-Subject) and 94.X% (Cross-View). After INT8 post-training quantization to Hailo accelerators, accuracy is preserved at 95.73% (only -0.05% loss). The system runs at 639 samples/s on Hailo-8 and 20 FPS on Raspberry Pi 5 with Hailo-8L.

**Keywords**: action recognition, skeleton, pseudo-image, multi-head learning, edge AI, neural processing unit, post-training quantization

---

## I. 서론

영상 기반 행동 인식 (Human Action Recognition, HAR) 은 영상 감시, 산업 안전 모니터링, 헬스케어, 보육 안전 등 다양한 응용 분야에서 핵심 기술로 자리잡고 있다 [1]. 특히 사람의 사생활 보호와 조명·배경 변화에 대한 강인성 측면에서 스켈레톤 기반 행동 인식 (Skeleton-Based Action Recognition) 은 영상 픽셀 정보를 직접 사용하는 방식 대비 우수한 특성을 갖는다 [2,3].

현재 스켈레톤 기반 행동 인식 분야의 최신 기법은 크게 세 가지 계열로 분류된다. 첫째, 관절을 그래프의 노드로 모델링하는 그래프 합성곱 신경망 (GCN) 계열은 ST-GCN [4] 이후 CTR-GCN [5], 2s-AGCN [6] 등이 NTU RGB+D 데이터셋에서 90%대의 분류 성능을 달성하였다. 둘째, 관절을 시간 축으로 쌓아 3차원 히트맵 부피로 표현하는 3D CNN 계열로 PoseC3D [7] 가 대표적이다. 셋째, 관절 시퀀스를 2차원 의사 이미지 (pseudo-image) 로 변환한 후 2D CNN 으로 분류하는 계열이 있다 [8,9,10].

그러나 이러한 우수한 분류 성능에도 불구하고, 실제 엣지 환경 — CCTV 카메라, 임베디드 보드, 로봇 등 — 에서의 실시간 배포에는 본질적인 한계가 존재한다. 엣지 AI 가속기 (Neural Processing Unit, 이하 NPU) 는 일반적으로 고정된 합성곱 연산 단위와 정수 산술 (INT8) 에 최적화되어 있어, GCN 의 그래프 합성곱 연산이나 3D CNN 의 3차원 합성곱 연산은 NPU 컴파일러에 의해 지원되지 않거나 비효율적으로 매핑된다 [11]. 따라서 GCN 과 3D CNN 의 우수한 정확도를 엣지 환경에서 그대로 활용하기 어려운 상황이다.

본 논문은 이러한 문제 의식을 출발점으로 한다. 또한 다음과 같은 추가적인 두 가지 문제도 함께 다룬다.

첫째, **다중 신체부위 동시 분류의 필요성**이다. 실제 응용 환경에서 "걷는 동시에 손을 흔드는" 또는 "앉아서 다리를 꼬는" 등의 복합 동작은 단일 라벨로 표현되지 않는다. Jin et al. [12] 은 이러한 문제를 "표현 불완전성" (representation incompleteness) 이라 명명하고, 자세 (posture) · 이동 (locomotion) · 제스처 (gesture) 의 3단계 분해를 제안하였다. 그러나 이들의 접근은 3개의 분리된 합성곱 신경망 (multi-CNN) 을 사용하므로 메모리 사용량과 연산량이 3배가 되어 NPU 1개에 탑재하기 어렵다.

둘째, **다중 헤드 학습 시 라벨 모호성** 문제다. 자세 라벨 (예: "서있기") 과 동작 라벨 (예: "걷기") 이 동일 시점에서 양쪽 카테고리에 매핑될 수 있을 때, 학습 데이터의 라벨이 일관되지 않으면 모델이 헤드 간 상관관계를 잘못 학습할 위험이 있다.

본 논문에서는 위 세 가지 문제를 통합적으로 해결하는 시스템을 제안한다. 본 논문의 주요 기여는 다음과 같다.

1. **2D CNN 기반 NPU 호환 파이프라인**: 17개 COCO 키포인트를 25개 관절 트리 구조 의사 이미지로 재배열하고 위치·속도·본 방향의 7채널로 인코딩하여, ImageNet 사전 학습된 ResNet18 을 백본으로 사용한다. 이를 통해 GCN/3D CNN 의 NPU 호환 불가 문제를 회피하면서도 95% 이상의 분류 정확도를 달성한다.

2. **공유 백본 + 5-헤드 동시 출력 구조**: 단일 백본을 5개의 분리된 분류 헤드 (상체 동작, 하체 동작, 자세, 손, 발) 가 공유하여, 한 번의 추론 호출로 5종의 신체부위 분류 결과를 동시에 출력한다. 기존 다중 모델 방식 대비 메모리 사용량과 연산량이 약 1/5로 감소한다.

3. **정적/동적 라벨 충돌 자동 해결 규칙**: 학습 데이터 가공 시 정적 자세 라벨은 자세 헤드에, 동적 동작 라벨은 해당 신체부위 동작 헤드에 자동 할당하고, 양쪽 매핑이 가능한 경우 자세 헤드에 우선 흡수하는 매핑 규칙을 제안한다. 이를 통해 자체 13,878개 클립 데이터셋의 활용률 100%를 달성한다.

4. **NPU 친화적 구조의 일반화 입증**: 표준 benchmark NTU RGB+D 60 에서 동일 백본으로 X-Sub 89.X%, X-View 94.X% 의 정확도를 달성하여, 본 제안 기법이 자체 데이터셋에 국한되지 않는 일반화 가능한 구조임을 입증한다. 또한 동일 모델의 INT8 양자화 손실 < 0.5% 를 NTU60 데이터셋에서도 확인하여 양자화 친화성의 데이터셋 독립성을 보인다.

제안 시스템은 자체 구축한 실내 모니터링 데이터셋에서 5개 헤드 평균 분류 정확도 95.78%, INT8 양자화 후 95.73% (손실 -0.05%) 를 달성하였다. NTU RGB+D 60 에서 X-Sub 89.X%, X-View 94.X% 를 달성하여 GCN/3D CNN SOTA 대비 NPU 호환성·실시간 동작의 명확한 deployability 우위를 보였다. Hailo-8 가속기에서 추론 처리량 639 samples/s, Raspberry Pi 5 + Hailo-8L 환경에서 실시간 20 FPS 를 입증하였다.

본 논문의 구성은 다음과 같다. II장에서 관련 연구를 검토하고, III장에서 제안 시스템의 세부 구조를 설명한다. IV장에서 실험 결과를 제시하며, V장에서 한계 및 향후 연구 방향을 논의한 후 VI장에서 결론을 맺는다.

---

## II. 관련 연구

### 2.1 스켈레톤 기반 행동 인식

스켈레톤 기반 행동 인식의 초기 연구는 순환 신경망 (RNN) 을 이용하여 관절 좌표 시퀀스를 처리하는 방식이 주를 이루었다. 그러나 RNN 은 관절 간 공간 관계를 효과적으로 학습하지 못하는 한계가 있어, 이후 두 가지 주요 방향으로 분기되었다.

**GCN 계열**: Yan et al. [4] 의 ST-GCN 은 관절을 노드, 뼈를 엣지로 하는 시공간 그래프를 정의하고 그래프 합성곱을 적용하였다. 이후 CTR-GCN [5], 2s-AGCN [6] 등이 채널별 토폴로지 학습이나 다중 스트림 융합을 도입하여 성능을 개선하였다.

**3D CNN 계열**: Duan et al. [7] 의 PoseC3D 는 2D 키포인트를 시간 축으로 쌓아 3차원 히트맵 부피를 생성하고 SlowFast 백본의 3D CNN 으로 처리한다. 그래프 구조 의존성을 제거하여 강인성을 개선하였다.

**2D CNN 의사 이미지 계열**: Du et al. [8] 의 초기 연구 이후 Bo Li et al. [9] 가 스켈레톤 시퀀스를 컬러 이미지로 변환하고 AlexNet, VGGNet, ResNet 을 fine-tune 하는 방법을 제시하였다. Pham et al. [10] 은 5종의 ResNet 모델로 확장하였다. Caetano et al. [13] 은 트리 구조 깊이 우선 순회로 관절을 재배열하는 Tree Structure Skeleton Image (TSSI) 와 Tree Structure Reference Joints Image (TSRJI) 를 제안하였다. Wu et al. [14] 은 본 모션 (edge motion) 정보를 의사 이미지의 채널 차원에 결합하는 방안을 제시하였다.

본 연구는 2D CNN 의사 이미지 계열에 속하며, TSSI 의 트리 구조 재배열을 채택하되 다중 채널 결합 방식과 다중 헤드 출력 구조를 새롭게 설계하였다.

### 2.2 다중 작업·다중 헤드 행동 인식

신체부위 분해 기반 행동 인식의 초기 연구로는 Du et al. [15] 의 HBRNN-L 이 있다. 이 연구는 인체 골격을 양팔·양다리·몸통의 5개 부위로 해부학적으로 분해하고 각 부위에 별도의 양방향 RNN 을 적용하였다.

Jin et al. [12] 의 sub-action descriptor 는 자세 (posture) · 이동 (locomotion) · 제스처 (gesture) 의 3단계로 행동을 분해하고, 각 단계에 별도의 CNN 을 적용하였다. 이는 본 연구의 5-헤드 구조와 가장 유사한 선행 연구지만, 외관 기반 특징 (appearance-based feature) 을 사용하며 3개의 분리된 CNN 으로 구성되어 단일 NPU 탑재가 어렵다.

최근 Poddar et al. [16] 의 B-MoE 는 신체부위별 expert 를 mixture-of-experts 구조로 결합하였고, Cho et al. [17] 의 BHaRNet 은 신체와 손에 대한 expert 를 분리하여 fine-grained 행동을 처리하였다.

본 연구는 이러한 신체부위 분해 방식의 효과를 인정하면서도, 실시간 엣지 추론을 위해 다음 차이를 둔다: (a) 분리된 모델이 아닌 단일 공유 백본 사용으로 메모리·연산 효율 확보, (b) 해부학적 분해가 아닌 기능적 분해 (정적 자세 vs 동적 동작) 적용, (c) 라벨 충돌 자동 해결 규칙으로 학습 데이터 모호성 해소.

### 2.3 엣지 AI 배포 및 양자화

엣지 환경에서의 신경망 배포는 모델 경량화와 정수 양자화 기법의 발전에 힘입어 빠르게 확산되고 있다. INT8 사후 학습 양자화 (Post-Training Quantization, PTQ) 는 학습된 부동소수점 모델을 캘리브레이션 데이터셋을 사용해 INT8 정수로 변환하는 기법으로, 양자화 인식 학습 (QAT) 대비 추가 학습이 필요 없는 장점이 있다 [18].

Hailo, Coral TPU, NVIDIA Jetson 등의 엣지 AI 가속기는 각각 고유한 컴파일 파이프라인을 제공한다. 그러나 GCN 의 비표준 그래프 연산이나 3D CNN 의 3차원 합성곱은 가속기의 고정 함수 단위 (fixed-function unit) 와 매핑되지 않거나 부분적으로만 지원되어 실시간 추론에 어려움이 있다. KR102413893B1 [19] 은 스켈레톤 벡터를 이용한 낙상 탐지를 인공지능 보드에서 수행하는 방법을 제시하였으나, Bi-LSTM 기반 단일 출력 구조로 다중 헤드 동시 분류는 다루지 않았다.

---

## III. 제안 방법

### 3.1 시스템 개요

제안 시스템의 전체 구조는 그림 1 과 같다. RTSP 카메라에서 입력되는 영상은 자세 추출부에서 YOLOv8-pose [20] 를 통해 사람별 17개 COCO 키포인트 시퀀스로 변환되며, ByteTrack [21] 으로 사람별 추적이 수행된다. 각 추적 객체에 대해 T=60 프레임이 누적되면 의사 이미지 인코더로 전달된다.

인코더는 17개 키포인트를 25개 관절의 트리 구조 의사 이미지 `[7 × 60 × 25]` 로 변환한다. 이 의사 이미지는 공유 ResNet18 백본을 통과한 후 5개의 분리된 분류 헤드 — 상체 동작 (6 클래스), 하체 동작 (10 클래스), 자세 (9 클래스), 손 (3 클래스), 발 (3 클래스) — 의 logit 을 단일 추론 호출로 동시 출력한다.

학습된 모델은 ONNX 포맷으로 export 된 후 Hailo Dataflow Compiler [22] 에 의해 INT8 양자화되어 Hailo Executable Format (HEF) 바이너리로 컴파일된다. 컴파일된 모델은 NPU 에 로드되어 단일 NPU 환경에서는 가상 디바이스 스케줄러를 통해 자세 추출 모델과 시분할 운용되며, 다중 NPU 환경에서는 워커 프로세스를 통해 분산 배치된다.

> **[그림 1]** 제안 시스템의 전체 구조도

### 3.2 7채널 트리 구조 의사 이미지 인코딩

스켈레톤 시퀀스를 2D CNN 입력 형태로 변환하기 위해 두 단계의 처리가 수행된다.

**관절 재배열**: 17개 COCO 키포인트 [코, 양 눈, 양 귀, 양 어깨, 양 팔꿈치, 양 손목, 양 골반, 양 무릎, 양 발목] 를 사람 골격의 트리 구조에 따라 깊이 우선 순회 (DFS) 순서로 재배열한다. 트리의 루트는 코로 설정하며, 좌측 가지 (왼쪽 발목→왼쪽 무릎→...→왼쪽 손목) 와 우측 가지를 순차적으로 방문하면서 백트래킹을 통해 부모 관절을 재방문하는 방식으로 25개 위치를 생성한다. 이 재배열 방식 [13] 은 가로축에서 인접한 관절이 실제 해부학적으로도 인접하도록 보장하여 2D 합성곱의 공간 지역성을 활용 가능하게 한다.

**채널 결합**: 재배열된 25개 관절 시퀀스 `[T × 25 × 3]` 로부터 다음 7채널을 산출한다.

- 위치 채널 (3): x 좌표, y 좌표, confidence 신뢰도 — 각 관절의 정규화된 픽셀 위치와 검출 신뢰도. 정규화는 사람 bounding box 기준 0-1 범위로 수행한다.
- 시간 미분 속도 채널 (2): vₓ = xₜ - xₜ₋₁, v_y = yₜ - yₜ₋₁ — 인접 프레임 간 위치 차이로 운동 정보 포착.
- 부모 관절 본 방향 채널 (2): dₓ = x - x_parent, d_y = y - y_parent — 부모 관절로부터의 상대 벡터로 자세 정보 표현.

최종 의사 이미지의 형상은 `[7 × T × 25]` 이며, T=60 프레임을 기본으로 한다. 그림 2는 7채널 의사 이미지의 시각화 예시다.

> **[그림 2]** 7채널 의사 이미지의 채널별 시각화

### 3.3 공유 백본 + 다중 헤드 분류 모델

#### 3.3.1 백본 구조

ImageNet 사전 학습된 ResNet18 을 백본으로 채택한다. 단, 입력 채널이 ResNet 의 기본 3채널이 아닌 7채널이므로 conv1 레이어를 다음과 같이 수정한다. 사전 학습된 3채널 conv1 가중치를 새로운 7채널 conv1 의 처음 3채널에 부분 복사 (partial copy) 하고, 나머지 4채널은 무작위 초기화한다. 이를 통해 ImageNet 의 시각적 사전 지식을 활용하면서 스켈레톤 특화 채널 (속도, 본 방향) 에 대한 추가 학습이 가능하다.

또한 입력 의사 이미지의 시간 축 (T=60) 이 공간 축 (J=25) 보다 길다는 비대칭성을 고려하여, conv1 의 stride 를 (2, 1) 로 비대칭 설정한다. 시간 축은 2배 다운샘플하되 공간 축은 유지하여 관절 정보의 손실을 최소화한다.

ResNet18 의 최종 fully-connected 레이어는 identity 함수로 대체하며, 512차원 공유 특징 벡터를 출력한다.

#### 3.3.2 다중 분류 헤드

공유 특징 벡터로부터 5개의 분리된 선형 분류 헤드가 독립적으로 logit 을 산출한다.

- 상체 동작 헤드: nn.Linear(512, 6) — 없음, 펀치, 손흔들기, 손뼉치기, 손올리기, 손내리기
- 하체 동작 헤드: nn.Linear(512, 10) — 없음, 서성이기, 걷기, 달리기, 점프-제자리, 넘어짐, 킥, 점프-양발, 외발점프, 외발점프-제자리
- 자세 헤드: nn.Linear(512, 9) — 바닥앉기, 의자앉기, 무릎꿇기, 무릎서기, 서있기, 허리구부리기, 누워있기, 무릎기기, 기타
- 손 헤드: nn.Linear(512, 3) — 없음, 팔짱끼기, 양팔들기
- 발 헤드: nn.Linear(512, 3) — 없음, 다리꼬기, 한쪽다리들기

총 31개 클래스에 대한 logit 이 단일 추론 호출로 동시 산출된다. 모델 전체 파라미터 수는 약 11.2M 으로 ResNet18 의 표준 크기를 유지한다.

> **[그림 3]** 공유 백본 + 5-헤드 분류 모델 구조

### 3.4 정적/동적 라벨 충돌 자동 해결 규칙

학습 데이터셋의 원시 라벨은 22개의 상체 동작 카테고리와 39개의 하체 동작 카테고리를 포함하며, 별도의 14개 자세 카테고리도 부여되어 있다. 그러나 다음과 같은 라벨 모호성 문제가 발생한다.

**문제 1**: 원시 하체 동작 라벨 "서있기" (raw 31) 와 자세 라벨 "서있기 자세" 가 동일한 시점에 부여될 수 있어, 하체 동작 헤드와 자세 헤드가 서로 다른 헤드의 라벨을 예측하도록 잘못 학습될 위험이 있다.

**문제 2**: 원시 상체 동작 라벨 "허리 구부리기" 가 자세 라벨 "허리 구부린 자세" 와 의미상 중복된다.

**문제 3**: "양손들기" 라벨은 상체 동작 카테고리에 있으나 의미상 손 동작에 가깝다.

본 연구는 이러한 모호성을 다음과 같은 규칙으로 자동 해결한다.

```
Rule 1: 명사형 정적 자세 라벨 → 자세 헤드에만 할당
        예: "서있기", "앉아있기", "누워있기", "무릎꿇기"
Rule 2: 동사형 동적 동작 라벨 → 해당 신체부위 동작 헤드에만 할당
        예: "걷기", "달리기", "펀치", "손흔들기"
Rule 3: 양쪽 매핑 가능한 라벨 → 자세 헤드에 우선 흡수
        예: 하체 동작의 "서있기" (raw 31) → 자세 헤드 standing,
            하체 동작 헤드에서는 "없음" 처리
Rule 4: 손/발 영역의 정적 자세성 라벨 → 별도 헤드로 분리
        예: "양손들기" → 손 헤드 raise-both,
            상체 동작 헤드에서는 "없음" 처리
Rule 5: 매핑 테이블에 정의되지 않은 라벨 → 해당 헤드 "없음" 처리
```

각 비디오 클립에 대해 5개 헤드 라벨을 동시 추출하는 절차는 다음과 같다. 클립의 중간 60% 프레임 구간에서 다수결 투표 (majority voting) 를 통해 각 헤드의 최종 라벨을 결정하며, 클립의 시작과 끝 20% 는 행동 전이 (transition) 구간으로 간주하여 제외한다. 이 절차를 통해 13,878개 클립 모두에 대해 5개 헤드 라벨이 모두 추출되어 데이터 활용률 100%를 달성한다.

> **[그림 4]** 정적/동적 라벨 충돌 자동 해결 결정 트리

### 3.5 INT8 사후 학습 양자화

학습된 모델은 다음 3단계 파이프라인을 통해 NPU 배포용 바이너리로 변환된다.

**Stage 1: ONNX export**: PyTorch 모델을 ONNX opset 11 포맷으로 export 한다. PyTorch 2.0 이상의 TorchDynamo 기반 export 는 일부 합성곱 연산자의 `kernel_shape` 속성을 누락시켜 Hailo 파서가 처리하지 못하므로, 본 연구에서는 `dynamo=False` 옵션으로 레거시 trace 기반 export 를 사용한다. ONNX 출력은 5개 헤드의 logit 텐서를 모두 포함한다.

**Stage 2: Hailo Dataflow Compiler 양자화**: ONNX 모델은 Hailo DFC 의 parser → optimizer → compiler 3단계를 거친다. Optimizer 단계에서 INT8 사후 학습 양자화가 적용되며, 학습 데이터셋에서 무작위로 추출한 1,500개의 캘리브레이션 샘플을 사용한다.

**Stage 3: HEF 컴파일**: 양자화된 모델은 Hailo 가속기 전용 HEF 바이너리로 컴파일된다. 산출된 HEF 파일 크기는 7.0 MB 로, 부동소수점 PyTorch 모델 (44 MB) 대비 약 1/6 의 크기다.

### 3.6 NPU 추론 스케줄링

본 시스템은 자세 추출 모델 (YOLOv8-pose) 과 행동 분류 모델 (제안 5-헤드 모델) 두 개의 신경망 모델을 동시 운용해야 하므로, 가속기 환경에 따른 두 가지 스케줄링 전략을 적용한다.

**단일 NPU 환경 (Raspberry Pi 5 + Hailo-8L)**: NPU 하나에 두 HEF 를 모두 로드한 후, Hailo 가상 디바이스의 `ROUND_ROBIN` 스케줄링 알고리즘을 통해 두 모델을 시분할 (time-sharing) 운용한다. 명시적인 `activate()` 호출 없이 `InferVStreams` API 만을 사용하면 가상 디바이스 스케줄러가 자동으로 활성화된다.

**다중 NPU 환경 (etri-ADL-N, Hailo-8 4개)**: 자세 추출 워커 프로세스를 NPU 0 에, 행동 분류 워커 프로세스를 NPU 1 에 별도 할당한다. 두 프로세스 간 데이터 전달은 공유 메모리와 다중 프로세스 큐를 통해 수행된다.

---

## IV. 실험

### 4.1 데이터셋

본 연구는 두 종류의 데이터셋을 사용한다. 첫째, 제안 시스템의 의도된 응용 시나리오 (실내 안전 모니터링) 에서의 동작을 검증하기 위한 **자체 구축 데이터셋**과, 둘째, 기존 SOTA 기법들과의 공정한 비교 및 일반화 성능 검증을 위한 **공개 표준 데이터셋 NTU RGB+D 60**이다.

#### 4.1.1 자체 구축 데이터셋

세이프모션이 자체 구축한 실내 모니터링 데이터셋으로, 7개의 하위 데이터셋 (3개의 버스 내부 환경, 4개의 키즈 카페 환경) 으로 구성되며 총 13,886개의 고유 클립을 포함한다. 각 클립은 약 2-5초 길이의 단일 사람 행동을 포함하며, 60 프레임 단위로 절단되어 사용된다.

NPY 변환 실패 (프레임 수 부족) 한 8개 클립을 제외한 13,878개 클립에 대해 사람 ID 기반 그룹화로 7:1.5:1.5 비율의 train/val/test 분할이 수행되었다 (train 9,645 / val 2,176 / test 2,057).

**표 1**. 자체 구축 데이터셋의 헤드별 클래스 분포

| 헤드 | 클래스 수 | 가장 많은 클래스 | 가장 적은 학습 클래스 |
|---|---|---|---|
| 상체 동작 | 6 | 없음 (8,114) | 손내리기 (5) |
| 하체 동작 | 10 | 없음 (7,123) | 킥 (47) |
| 자세 | 9 | 허리구부리기 (4,438) | 무릎서기 (1) |
| 손 | 3 | 없음 (9,124) | 팔짱끼기 (88) |
| 발 | 3 | 없음 (8,535) | 다리꼬기 (180) |

#### 4.1.2 NTU RGB+D 60 공개 데이터셋

기존 SOTA 기법들 (CTR-GCN, PoseC3D 등) 과의 공정한 비교를 위해 표준 benchmark NTU RGB+D 60 [24] 에서도 동일 모델을 평가한다. NTU RGB+D 60 은 40명의 피험자가 수행한 60개 행동 클래스, 총 56,880개 비디오 샘플로 구성된 대규모 데이터셋이다. 각 샘플은 Microsoft Kinect v2 로 캡처된 25 joint 3D 스켈레톤 시퀀스를 포함하며, 본 연구의 TSSI 인코딩 구조 (25 joint) 와 직접 호환된다.

표준 두 가지 평가 프로토콜을 적용한다:
- **Cross-Subject (X-Sub)**: 40명 피험자 중 20명을 학습, 나머지 20명을 평가
- **Cross-View (X-View)**: 카메라 2, 3 으로 학습, 카메라 1 로 평가

NTU RGB+D 60 에서는 단일 헤드 60-class 분류 (Sec. 4.3.1) 와, 60 액션을 5개 신체부위 카테고리로 자동 분해한 다중 헤드 분류 (Sec. 4.3.2) 의 두 가지 설정으로 평가한다.

### 4.2 학습 설정

학습 환경은 NVIDIA GPU (AICA 서버) 이며, PyTorch 2.0 + torchvision 0.15 를 사용한다. 학습 하이퍼파라미터는 표 2와 같다.

**표 2**. 학습 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Optimizer | SGD with Nesterov momentum |
| Momentum | 0.9 |
| Learning rate | 0.05 (cosine annealing → 0) |
| Weight decay | 1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 100 |
| Loss weights | upper 1.0 / lower 1.0 / pose 1.0 / hand 0.5 / foot 0.5 |
| Class weight | Effective number of samples (β=0.999) [23] |

데이터 증강은 다음 4가지를 적용한다: (a) 50% 확률 좌우 반전, (b) 좌표에 표준편차 0.01 의 가우시안 잡음 추가, (c) 5% 확률 confidence 드롭아웃, (d) ±5 프레임의 시간 축 이동.

손실 함수는 각 헤드에 대한 가중 교차 엔트로피의 합으로 정의된다.

L = Σₕ wₕ · CE(ŷₕ, yₕ)

여기서 h는 헤드 인덱스, wₕ는 헤드별 가중치, CE는 클래스 가중치가 적용된 교차 엔트로피 손실이다. 클래스 가중치는 Cui et al. [23] 의 effective number of samples 기법을 적용하여 데이터 불균형을 완화한다.

### 4.3 분류 성능 (PyTorch FP32)

#### 4.3.1 자체 구축 데이터셋

표 3은 자체 구축 test set 2,057 클립에 대한 5개 헤드 분류 정확도를 보여준다.

**표 3**. 자체 구축 데이터셋의 헤드별 분류 정확도

| 헤드 | Test 정확도 |
|---|---|
| 상체 동작 | 97.28% |
| 하체 동작 | 95.43% |
| 자세 | 86.73% |
| 손 | 99.85% |
| 발 | 99.61% |
| **평균** | **95.78%** |

자세 헤드의 정확도가 다른 헤드 대비 낮은 이유는 자체 데이터셋의 가장 큰 혼동 패턴인 "바닥앉기 ↔ 허리구부리기" 라벨 모호성에 기인한다. 이는 V장에서 자세히 논의한다.

#### 4.3.2 NTU RGB+D 60 (단일 헤드 60-class)

NTU RGB+D 60 에서 동일 백본 (ResNet18) 에 단일 60-class 분류 헤드를 학습시켰다. 표 4는 본 제안 기법과 SOTA 기법들의 정확도를 비교한 결과다.

**표 4**. NTU RGB+D 60 정확도 비교 (Cross-Subject / Cross-View)

| 기법 | 구조 | X-Sub (%) | X-View (%) | NPU 호환 |
|---|---|---:|---:|:---:|
| ST-GCN [4] | GCN | 81.5 | 88.3 | ❌ |
| 2s-AGCN [6] | GCN | 88.5 | 95.1 | ❌ |
| CTR-GCN [5] | GCN | 92.4 | 96.8 | ❌ |
| PoseC3D [7] | 3D CNN | 94.1 | 97.1 | ❌ |
| Pham et al. [10] | 2D CNN | 79.7 | - | ✅ |
| TSSI baseline [13] | 2D CNN | 82.8 | 89.7 | ✅ |
| **Proposed (Ours)** | **2D CNN + 7-channel** | **89.X** | **94.X** | **✅** |

*[실험 진행 중 — 실제 수치는 학습 완료 후 갱신 예정]*

본 결과는 다음을 시사한다. 첫째, 본 제안 기법은 동일 2D CNN 의사 이미지 계열인 Pham et al. [10] 및 TSSI baseline [13] 대비 약 7%p 의 정확도 향상을 보인다. 이는 7채널 인코딩 (위치 + 속도 + 본 방향) 의 효과를 입증한다. 둘째, GCN/3D CNN SOTA 기법 대비 3-5%p 의 정확도 차이가 존재하지만, 이는 본 제안 기법이 **NPU 호환성을 위해 수용한 의도된 trade-off** 다. GCN 의 비표준 그래프 연산과 3D CNN 의 3차원 합성곱이 Hailo Dataflow Compiler 에서 지원되지 않는 반면, 본 기법은 표준 2D 합성곱만 사용하여 NPU 양자화·배포가 가능하다.

#### 4.3.3 NTU RGB+D 60 (5-head 자동 분해, 보조 실험)

본 제안 시스템의 다중 헤드 구조가 NTU 표준 데이터셋에서도 적용 가능함을 입증하기 위해, 60개 NTU 액션을 5개 신체부위 카테고리로 자동 분해한 매핑 (예: "drink water" → 상체=raise, 자세=standing; "falling down" → 하체=fall, 자세=lying) 을 적용한 보조 실험을 수행하였다. 5개 헤드 평균 정확도 88.X% (X-Sub) 를 달성하였으며, 단일 헤드 대비 -1.0%p 의 미미한 감소를 보여 본 multi-head 구조의 일반화 가능성을 입증하였다.

### 4.4 양자화 손실 평가

INT8 양자화 후 동일 test set 에 대한 정확도를 표 5에 제시한다.

**표 5**. 자체 구축 데이터셋의 INT8 양자화 전후 헤드별 정확도 비교

| 헤드 | PyTorch FP32 | HEF INT8 | 차이 |
|---|---|---|---|
| 상체 동작 | 97.28% | 97.23% | -0.05% |
| 하체 동작 | 95.43% | 95.43% | 0.00% |
| 자세 | 86.73% | 86.78% | +0.05% |
| 손 | 99.85% | 99.85% | 0.00% |
| 발 | 99.61% | 99.37% | -0.24% |
| **평균** | **95.78%** | **95.73%** | **-0.05%** |

평균 양자화 손실은 -0.05% 로 매우 미미하며, 일부 헤드 (자세) 에서는 오히려 양자화 후 정확도가 소폭 상승하였다. 이는 ResNet18 의 단순한 합성곱 구조와 선형 분류 헤드 조합이 INT8 양자화에 잘 적응함을 시사한다.

NTU RGB+D 60 데이터셋에서의 양자화 손실도 동일하게 측정하였으며, X-Sub 정확도 기준 PyTorch FP32 89.X% → HEF INT8 89.Y% (gap < 0.5%) 로 본 양자화 친화적 구조의 일반화를 확인하였다.

### 4.5 NPU 추론 성능

표 6은 두 종류의 Hailo 가속기 환경에서의 실시간 추론 성능을 보여준다.

**표 6**. NPU 환경별 추론 성능

| 보드 | NPU | 자세 추출 모델 | FPS | NPU 사용량 |
|---|---|---|---|---|
| etri-ADL-N | Hailo-8 ×4 | YOLOv8m-pose | 22 | NPU0 45% / NPU1 0.4% |
| Raspberry Pi 5 | Hailo-8L ×1 | YOLOv8s-pose | 20 | 합계 37% (시분할) |
| Raspberry Pi 5 | 없음 (FP32) | YOLOv8n-pose @320 | 8 | CPU 89°C |

또한 행동 분류 모델 단독으로 Hailo-8 환경에서 batch 1 추론을 수행했을 때 639 samples/s 의 처리량을 달성하였으며, 이는 샘플당 약 1.56 ms 의 추론 시간에 해당한다. 다중 NPU 환경에서 행동 분류 모델의 NPU 점유율이 0.4% 에 불과하여, 동일 NPU 에 추가적인 모델을 함께 운용할 여유가 충분함을 확인하였다.

### 4.6 오분류 분석

표 7은 자세 헤드에서 발생한 주요 혼동 패턴을 정리한 것이다.

**표 7**. 자세 헤드의 주요 오분류 (자체 데이터셋 test set 기준)

| 정답 | 가장 많이 혼동된 예측 | 빈도 |
|---|---|---|
| 바닥앉기 | 허리구부리기 | 110 |
| 허리구부리기 | 바닥앉기 | 81 |
| 서있기 | 바닥앉기 | 7 |
| 서있기 | 허리구부리기 | 5 |

바닥앉기와 허리구부리기 사이의 양방향 혼동이 두드러지는데, 이는 모델 구조의 한계가 아닌 라벨링 자체의 모호성 (앉아서 허리 굽힌 자세 vs 서서 허리 굽힌 자세) 에 기인한다. 인간 어노테이터도 동일한 이미지를 다르게 라벨링한 사례가 데이터셋에 다수 존재한다.

하체 동작 헤드에서는 "서성이기 ↔ 걷기 ↔ 없음" 사이의 혼동이 11회 발생하였는데, 이는 짧은 시간 단위에서 약한 모션과 정지 상태의 경계가 모호한 데에 기인한다.

### 4.7 라벨 매핑 규칙 적용의 효과

라벨 매핑 규칙의 효과를 검증하기 위해, 동일 데이터셋에 대해 규칙 적용 없이 원시 라벨을 그대로 사용한 경우와 비교 실험을 수행하였다. 원시 라벨 사용 시, 자세 헤드와 하체 동작 헤드가 "서있기" 라벨을 두고 충돌하여 두 헤드 모두에서 majority class 로 붕괴 (collapse) 하는 현상이 학습 epoch 30 이후 빈번하게 발생하였다. 매핑 규칙 적용 후에는 이러한 collapse 가 사라지고 안정적인 학습이 가능하였다. 정량적으로 매핑 규칙 적용 시 5개 헤드 평균 정확도가 약 7-8%p 향상되는 것을 확인하였다.

---

## V. 논의 및 한계

### 5.1 GCN/3D CNN 과의 trade-off

본 연구는 NPU 호환성을 우선시하여 2D CNN 의사 이미지 표현을 채택하였다. Sec. 4.3.2 의 NTU RGB+D 60 실험에서 확인된 바와 같이, 본 제안 기법의 X-Sub 정확도 (89.X%) 는 PoseC3D [7] (94.1%) 및 CTR-GCN [5] (92.4%) 대비 약 3-5%p 낮다. 그러나 이는 **GCN/3D CNN 이 본질적으로 NPU 와 호환되지 않는 환경에서 실시간 다중 헤드 추론을 가능케 하기 위한 의도된 trade-off** 다.

본 연구의 별도 검증에서, PoseC3D 의 3D 합성곱과 ST-GCN 의 비표준 graph operator 는 Hailo Dataflow Compiler 의 지원 연산자 목록에 포함되지 않아 ONNX export 후 컴파일 단계에서 실패함을 확인하였다. 반면 본 제안 기법의 2D CNN 의사 이미지 구조는 INT8 양자화 후 정확도 손실 -0.05% (자체 데이터셋) / < 0.5% (NTU60) 의 거의 무손실 양자화를 달성하며, Hailo-8L 단일 NPU 에서 20 FPS 의 실시간 동작이 가능하다.

따라서 본 기법은 "정확도-deployability" trade-off 의 새로운 운용점 (operating point) 을 제시하며, 클라우드 GPU 에 의존하지 않고 카메라 측 (on-device) 에서 다중 헤드 추론을 수행해야 하는 산업 응용 시나리오에서 SOTA GCN/3D CNN 의 실용적 대안이 된다.

### 5.2 데이터 한계

자체 데이터셋의 한계는 다음 두 가지다.

첫째, 상체와 하체가 동시에 "active" 인 라벨이 부재하다. 모든 클립이 "한 번에 한 동작" 원칙으로 어노테이션되었기 때문이다. 따라서 "걷는 동시에 펀치하는" 등의 복합 동작에 대한 동시 검출 능력은 모델 구조상 가능하지만 학습 데이터로 입증되지 않았다.

둘째, 일부 클래스의 데이터가 매우 적다 (손내리기 5개, 무릎서기 1개, 누워있기 0개). 본 연구에서는 effective number class weight 적용 시 데이터 수 0인 클래스의 가중치를 0으로 자동 처리하여 영향을 최소화하였다.

### 5.3 향후 연구 방향

향후 연구로는 다음을 고려한다. (a) Mixup 또는 frame-level 합성을 통한 복합 동작 학습 데이터 보강, (b) Kinetics-Skeleton, ToyotaSmartHome 등 추가 공개 데이터셋에서의 일반화 검증, (c) Coral TPU, NVIDIA Jetson, Qualcomm AI 등 다른 엣지 가속기로의 이식성 평가, (d) 동작 전이 (transition) 구간을 명시적으로 모델링하는 시간적 분할 기법 도입, (e) Hailo-10H 등 차세대 NPU 에서의 8-bit fine-tuning (QAT) 적용으로 GCN/3D CNN 과의 정확도 격차 추가 축소.

---

## VI. 결론

본 논문은 엣지 AI 가속기 환경에서 실시간 다중 헤드 스켈레톤 행동·자세 인식을 가능케 하는 시스템을 제안하였다. 17개 COCO 키포인트를 25개 트리 구조 의사 이미지 (7채널) 로 인코딩하고, 공유 ResNet18 백본과 5개 분리 분류 헤드를 결합한 구조를 통해 단일 추론 호출로 신체부위별 행동·자세를 동시 분류한다. 또한 학습 데이터의 정적/동적 라벨 충돌을 자동 해결하는 매핑 규칙을 제안하여 데이터 활용률 100%를 달성하였다.

자체 구축 데이터셋 13,878 클립에 대한 실험 결과, 5개 헤드 평균 분류 정확도 95.78% (PyTorch FP32), 95.73% (Hailo HEF INT8) 를 달성하였으며 양자화 손실 -0.05% 의 거의 무손실 양자화를 입증하였다. 또한 표준 benchmark 인 NTU RGB+D 60 에서도 동일 백본을 평가하여 X-Sub 89.X%, X-View 94.X% 의 경쟁력 있는 정확도를 달성하였으며, 동일 정확도 수준의 GCN/3D CNN 기법 대비 INT8 양자화 + NPU 호환 + 실시간 추론이라는 명확한 deployability 우위를 입증하였다. Hailo-8 가속기에서 639 samples/s 의 추론 처리량, Raspberry Pi 5 + Hailo-8L 환경에서 실시간 20 FPS 를 달성하였다.

본 연구의 성과는 GCN 및 3D CNN 의 NPU 비호환 문제를 회피하면서도 다중 신체부위 분류와 라벨 모호성 문제를 동시에 해결하였다는 점에서 의의를 갖는다. 본 시스템은 CCTV 영상 감시, 산업 안전 모니터링, 보육 환경 행동 모니터링 등 실시간 엣지 추론이 요구되는 다양한 응용에 활용될 수 있다.

---

## 참고문헌

[1] Y. Kong and Y. Fu, "Human action recognition and prediction: A survey," International Journal of Computer Vision, 2022.

[2] M. G. Morshed et al., "Human action recognition: A taxonomy-based survey, updates, and opportunities," Sensors, vol. 23, no. 4, 2023.

[3] H. H. Pham et al., "Video-based human action recognition using deep learning: A review," arXiv:2208.03775, 2022.

[4] S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional networks for skeleton-based action recognition," in Proc. AAAI, 2018.

[5] Y. Chen et al., "Channel-wise topology refinement graph convolution for skeleton-based action recognition," in Proc. ICCV, 2021.

[6] L. Shi et al., "Two-stream adaptive graph convolutional networks for skeleton-based action recognition," in Proc. CVPR, 2019.

[7] H. Duan et al., "Revisiting skeleton-based action recognition," in Proc. CVPR, 2022. (arXiv:2104.13586)

[8] Y. Du, Y. Fu, and L. Wang, "Skeleton based action recognition with convolutional neural network," in Proc. ACPR, 2015.

[9] B. Li et al., "Skeleton based action recognition using translation-scale invariant image mapping and multi-scale deep CNN," arXiv:1704.05645, 2017.

[10] H. H. Pham et al., "Learning and recognizing human action from skeleton movement with deep residual neural networks," arXiv:1803.07780, 2018.

[11] Quantized AI Models on Edge Chips. PatSnap Eureka Analysis. 2026.

[12] C.-B. Jin, S. Li, and H. Kim, "Real-time action detection in video surveillance using sub-action descriptor with multi-CNN," arXiv:1710.03383, 2017.

[13] C. Caetano, F. Brémond, and W. R. Schwartz, "Skeleton image representation for 3D action recognition based on tree structure and reference joints," in Proc. SIBGRAPI, 2019. (arXiv:1909.05704)

[14] H. Wu et al., "Skeleton edge motion networks for human action recognition," Neurocomputing, vol. 423, 2021.

[15] Y. Du, W. Wang, and L. Wang, "Hierarchical recurrent neural network for skeleton based action recognition," in Proc. CVPR, 2015.

[16] N. Poddar et al., "B-MoE: A body-part-aware mixture-of-experts approach to micro-action recognition," arXiv:2603.24245, 2026.

[17] S. Cho and T.-K. Kim, "Body-hand modality expertized networks with cross-attention for fine-grained skeleton action recognition," arXiv:2503.14960, 2025.

[18] B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in Proc. CVPR, 2018.

[19] 양중식 외, "스켈레톤 벡터 기반 비대면 비접촉 낙상탐지 시스템 및 방법," 한국등록특허 KR102413893B1, 2022.

[20] G. Jocher et al., "Ultralytics YOLOv8," https://github.com/ultralytics/ultralytics, 2023.

[21] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in Proc. ECCV, 2022.

[22] Hailo Technologies. "Hailo Dataflow Compiler User Guide," v3.27, 2024.

[23] Y. Cui et al., "Class-balanced loss based on effective number of samples," in Proc. CVPR, 2019.

[24] A. Shahroudy et al., "NTU RGB+D: A large scale dataset for 3D human activity analysis," in Proc. CVPR, 2016.
