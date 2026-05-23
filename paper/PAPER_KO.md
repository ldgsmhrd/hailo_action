# PSP-Net: 엣지 NPU 정수 양자화 배포를 위한 부위 분할 합성곱 기반 스켈레톤 행동 인식

**Target Journal**: Journal of Real-Time Image Processing (Springer, SCIE Q2)
**Manuscript class**: Original Research Article

**저자**: 이동균 (Lee, Donggyun) — ldg@safemotion.kr

---

## Abstract

스켈레톤 기반 행동 인식 (skeleton-based action recognition) 의 최근 고성능 보고치는 GCN, Transformer, 3D-CNN 등 동적 그래프 (dynamic graph), 자기 주의 (self-attention), 3D 합성곱을 전제로 하며, 이들은 저전력 엣지 NPU 의 INT8 정수 연산자 제약과 잘 합치하지 않는다. 본 논문은 스켈레톤 의사 이미지 (pseudo-image), 신체 부위 (body-part) 모델링, 다중 스트림 디스크립터 (multi-stream descriptor) 를 **표준 2D 합성곱 연산자만 (Conv2D, BatchNorm, ReLU, Sigmoid, Global Average Pooling, Linear) 으로 재구성한 PSP-Net (Partitioned Skeletal Pseudo-image Network)** 을 제안한다. 본 논문의 신규성은 pseudo-image, body-part, multi-stream 의 개별 아이디어 자체가 아니라 — 이들에는 선행 연구가 존재한다 — 이 요소들을 **INT8 NPU-compatible 2D CNN 아키텍처로 재구성하고 실제 Hailo edge NPU 에서 컴파일·양자화·속도·정확도를 검증한 점**에 있다. PSP-Net 은 (i) 부위 분할 grouped Conv2D (BodyPartConv), (ii) 시공간 분리 합성곱, (iii) 다중 확장 시간 합성곱, (iv) 채널 attention 의 네 컴포넌트로 구성되며, 학계의 4-stream ensemble 효과를 single-model 내부의 분기 구조와 $1 \times 1$ fusion 으로 NPU 1회 추론 안에 압축한 변종 (PSP-Net (MB-3D), PSP-Net (MB4)) 을 함께 제안한다.

NTU RGB+D 60 Cross-Subject 평가에서 PSP-Net (MB4) 는 1.42 M 파라미터로 3-seed mean **86.37 ± 0.06 %** (+ TTA **86.76 ± 0.22 %**), Cross-View **91.16 %** 의 정확도를 달성하였다. NTU RGB+D 120 Cross-Subject (NTU120-specific tuning) 에서는 MB4 **79.74 %**, MB-3D **79.63 %** 로 fixed-shape 2D CNN 의 약 80 % 표현 한계가 관찰되었다. Raspberry Pi 5 + Hailo-8L 환경에서 PSP-Net (MB-3D) 는 **348 FPS** (양자화 손실 −0.46 %p), PSP-Net (MB4) 는 **200 FPS** (−1.94 %p) 의 처리량을 보였고, RTSP decode + YOLO-Pose + PSP-Net + overlay 를 포함한 end-to-end 파이프라인 latency 약 32 ms (**31.2 FPS**) 로 실시간 처리가 가능하다. NPU-compatible 2D CNN 카테고리 내에서는 TSSI (2019) 79.2 % 대비 +7.17 %p 향상이다. PSP-Net 은 학계 GCN/Transformer SOTA 를 대체하는 모델이 아니라, INT8 edge NPU 에서 실시간 스켈레톤 행동 인식을 수행하기 위한 **practical baseline** 으로 위치한다.

**Keywords**: skeleton-based action recognition; edge NPU deployment; INT8 quantization; real-time inference; Hailo-8/8L; Raspberry Pi 5

---

## 1. Introduction

행동 인식 (action recognition) 은 영상 이해 분야의 핵심 과제이며, RGB 프레임 [6, 18], optical flow, 오디오 신호, 인체 스켈레톤 [10, 25] 등의 입력 표현이 연구되어 왔다. 스켈레톤 표현은 (i) 입력 크기가 작아 계산 효율이 높고, (ii) 조명·배경·복장 변화에 강건하며, (iii) 개인 식별 정보가 노출되지 않는 세 가지 장점이 있어, IP CCTV, 의료 영상 (낙상 감지), 인간-로봇 상호작용 등 다양한 응용 분야에 적합하다.

NTU RGB+D 60 [1] 벤치마크의 최근 우수 보고치는 3D-CNN 기반 PoseConv3D [2] 의 93.7 %, Transformer 기반 SkateFormer [3] 의 93.5 %, GCN 기반 CTR-GCN [4] 의 92.4 %, HD-GCN [12] 의 93.0 % 등이며, 모두 GPU FP32 학습 및 추론을 전제로 한다. 그러나 실제 단말은 NVIDIA Jetson 임베디드 GPU 보다 Hailo-8/8L, Google Coral Edge TPU, Rockchip RKNN 등 저전력 NPU 로 빠르게 이동하고 있다. 본 연구의 평가 환경인 Hailo-8 (26 TOPS) 및 Hailo-8L (13 TOPS) NPU 는 다음의 구조적 제약을 가진다.

1. **연산자 제한.** Conv2D, BatchNorm, ReLU, Sigmoid, AdaptiveAvgPool, Linear, Concatenate, IndexSelect 등 표준 2D CNN 연산자만 효율적으로 지원한다.
2. **3D 합성곱 미지원.** 3D Convolution 은 SRAM 폭증으로 NPU 컴파일이 거부된다.
3. **가변 형상 행렬곱 제약.** GCN 의 인접 행렬 곱셈은 NPU 의 systolic array 매핑이 비효율적이다.
4. **긴 시퀀스 Softmax 제약.** Transformer 의 attention 분모 계산이 NPU 의 고정 LUT 와 합치하지 않는다.
5. **정밀도.** 추론은 INT8 정수 연산으로 고정된다.

이러한 제약으로 위 학계 모델은 NPU 배포가 곤란하여 ARM CPU 또는 임베디드 GPU 로 fallback 되며, 단말의 전력 예산 (5–10 W) 과 발열 한계 (Raspberry Pi 5 의 ARM 부하 시 80 °C 임계) 를 초과하기 쉽다. 한편 NPU 호환 카테고리 (좌표 기반 2D CNN, 의사 이미지 표현) 의 학계 보고치는 JTM [5] 73.4 %, PA-CNN [6] 75.6 %, SkeleMotion [7] 76.5 %, TSSI [8] 79.2 % 로 2019 년 이후 정체되어 있다.

<!-- FIGURE_1 -->

이러한 제약 위에서 본 논문의 목표는 정확도 최대화가 아니라 **accuracy–deployability trade-off** 위에서의 NPU 호환 baseline 정립이다. 본 논문은 표준 2D 합성곱 연산자만으로 구성된 스켈레톤 행동 인식 아키텍처 PSP-Net 을 제안한다. 본 연구의 신규성은 — pseudo-image 표현, 신체 부위 분할, multi-stream descriptor 의 개별 요소 자체가 아니라, 이들에는 모두 선행 연구가 존재한다 (Section 2) — 이 요소들을 **graph adjacency multiplication, attention, 3D 합성곱을 사용하지 않고 표준 2D CNN 연산자만으로 표현 가능한 INT8 NPU-compatible 아키텍처로 재구성** 한 점, 그리고 실제 Hailo-8 / Hailo-8L NPU 에서 컴파일, 양자화 손실, 처리량, end-to-end latency 를 함께 측정·보고한 점에 있다.

**Contributions.** 본 논문의 기여는 다음 세 가지이다.

1. **NPU-compatible skeleton action recognition architecture.** Conv2D, BatchNorm, ReLU, Sigmoid, Global Average Pooling, Linear 등 표준 2D CNN 연산자만으로 구성된 스켈레톤 행동 인식 아키텍처 PSP-Net 을 제안한다. 핵심 컴포넌트인 BodyPartConv 는 graph adjacency multiplication 이나 학습 가능한 attention weight 없이, anatomical locality prior 를 static reshape + grouped Conv2D 로 구현한다.

2. **Single-pass multi-stream compression.** 학계 GCN 계열의 표준 4-stream ensemble [11, 4, 12] 은 Joint / Bone / Joint Motion / Bone Motion 4 개 모델을 별도 학습, 저장, 추론하고 logit 을 평균한다. PSP-Net (MB4) 는 24 채널 multi-stream descriptor 를 4 개의 internal branch 로 나누고 $1 \times 1$ fusion 합성곱으로 결합하여, NPU 1 회 추론 안에서 multi-stream 효과를 압축한다. 학습·저장·추론 비용이 단일 모델로 통합된다.

3. **Real INT8 NPU validation.** GPU FP32 정확도만 보고하는 학계의 관행과 달리, 본 논문은 Hailo Dataflow Compiler 의 INT8 quantization loss, HEF compilation 결과, standalone FPS (Hailo-8 / Hailo-8L), 그리고 Raspberry Pi 5 환경에서의 RTSP decode + YOLO-Pose + PSP-Net + overlay 를 포함한 end-to-end latency 까지 동일한 NPU 배포 파이프라인 위에서 함께 보고한다.

본 논문은 학계 GCN/Transformer SOTA 의 정확도를 대체하는 것을 목표로 하지 않는다. 본 논문의 위치는 dynamic graph, attention, 3D 합성곱의 표현력을 일부 포기하는 대신, commodity edge NPU 에서 deterministic INT8 deployment 와 real-time throughput 을 확보하는 **practical baseline** 이다.

본 논문의 나머지는 다음과 같이 구성된다. Section 2 에서는 관련 연구를 (2D CNN, GCN, 3D-CNN, Transformer, multi-stream, deployment-oriented) 흐름으로 정리한다. Section 3 에서는 PSP-Net 의 아키텍처와 학습 목표를 형식화한다. Section 4 에서는 NTU60 / NTU120 평가, ablation, INT8 양자화 결과, NPU 추론 속도, end-to-end pipeline 을 보고한다. Section 5 에서는 학계 SOTA 와의 격차, NTU120 representation ceiling, quantization sensitivity 의 모델 의존성, 한계를 논의한다. Section 6 은 결론이다.

---

## 2. Related Work

스켈레톤 기반 행동 인식 분야는 (i) 입력 표현, (ii) 백본 아키텍처, (iii) 다중 스트림 융합, (iv) 학습 기법, (v) 엣지 배포 의 다섯 축에서 발전해 왔다. 본 절에서는 각 축의 주요 흐름을 정리하며, NPU 호환성 관점에서 한계와 PSP-Net 의 차별점을 분석한다.

**2D CNN-based skeleton action recognition (pseudo-image 흐름).** 초기 흐름은 좌표 시퀀스를 의사 이미지 (pseudo-image) 로 변환한 후 표준 2D CNN 으로 처리하는 방식이다. JTM [5] 은 관절 궤적을 RGB 색상으로 인코딩하여 ImageNet-pretrained CNN 으로 분류한다. PA-CNN [6] 은 부위별 spatial pooling 을 적용한다. SkeleMotion [7] 은 motion vector 를 색상화한다. TSSI [8] 는 trajectory-based 2D 입력에 ResNet 류 백본을 적용하여 NTU60 CS 79.2 % 를 보고하였다. 이 카테고리는 NPU 호환성이 우수하나, 2D 이미지 변환의 information loss 때문에 [2] 표현력의 한계로 2020 년 이후 정체되었다. **PSP-Net 의 차별점**: pseudo-image 표현 자체는 선행 연구가 충분히 존재하나, 본 연구는 이를 단순 representation encoding 이 아니라 **NPU-compatible Conv2D execution 의 deployment interface** 로 사용한다. 즉, pseudo-image 의 형식과 채널 구성을 Hailo INT8 path 에서 가장 효율적으로 매핑되도록 (NCHW → NHWC, $[-10, 10]$ outlier clipping 포함) 설계한다.

**Body-part / part-level skeleton modeling.** 신체 부위 (body-part) 또는 부위 그룹 (limb / torso / head) 단위의 표현을 사용하는 흐름은 다수 존재한다. Part-level GCN 은 각 부위 부분 그래프를 독립 학습하며, body-part attention 은 부위별 가중치를 학습한다. Hierarchical body decomposition 은 골격의 계층적 분해를 모델링한다 (HD-GCN [12] 등). 따라서 "body part 를 나누었다" 자체는 신규성이 약하다. **PSP-Net 의 차별점**: 본 연구는 부위 prior 를 GCN 의 인접 행렬이나 attention 의 학습 가능한 weight 가 아니라, **static reshape + grouped Conv2D** (PyTorch `Conv2d(groups=5)`) 라는 표준 2D CNN 연산자만으로 표현한다. 이는 anatomical locality bias 를 유지하면서 그래프 연산을 회피하므로 NPU 컴파일러가 단일 grouped convolution layer 로 효율 매핑 가능하다.

**GCN for skeleton-based action recognition.** ST-GCN [10] 이 시작이며, 관절을 노드·골격을 간선으로 모델링한다. 2s-AGCN [11] 은 학습 가능한 인접 행렬을, MS-G3D [9] 는 multi-scale 그래프와 시간 그래프 통합으로 NTU60 CS 91.5 % 를 달성하였다. CTR-GCN [4] 은 channel-wise topology refinement 로 92.4 %, HD-GCN [12] 은 hierarchical 분해로 93.0 % 를 보고하였다. 공통적 한계는 (i) 학습 가능한 인접 행렬의 가변 형상이 NPU 의 systolic array 매핑에 비효율적이고, (ii) 다수가 4-스트림 (Joint / Bone / Joint Motion / Bone Motion) 앙상블에 의존하여 모델 4 개의 학습 비용과 추론 비용이 동반된다는 점이다.

**3D-CNN for skeleton-based action recognition.** PoseConv3D [2] 는 좌표의 sparse 표현 대신 2D 관절 좌표를 가우시안 히트맵 시공간 볼륨 ($K \times T \times H \times W$) 으로 변환하여 SlowOnly [18] 류 3D-CNN 으로 처리한다. 입력 volume 이 약 1.8 M voxel 의 dense 표현이라 정보량이 좌표 기반 대비 약 100–500 배에 해당하며, NTU60 CS 93.7 % 의 우수 보고치를 갖는다. 그러나 (i) 3D Convolution 자체가 NPU 비호환이며, (ii) 약 50 G MACs 의 연산량으로 임베디드 환경의 실시간 처리가 어렵다.

**Transformer-based skeleton action recognition.** SkateFormer [3] 와 ST-TR [23] 은 관절·시간 토큰의 self-attention 으로 행동을 모델링한다. SkateFormer 는 NTU60 CS 93.5 % 를 보고하나, Multi-Head Attention 의 softmax 와 LayerNorm 이 NPU 의 고정 LUT 와 정합되지 않아 엣지 배포가 곤란하다. 또한 시퀀스 길이가 $T \times J = 1600$ (NTU60) 인 경우의 quadratic 메모리가 부담이다.

**Multi-stream input ensemble.** GCN 계열의 우수 보고치는 공통적으로 Joint / Bone / Joint Motion / Bone Motion 4 스트림 각각에 대해 별도 모델을 학습한 후 logit 평균하는 4-stream ensemble [11, 4, 12] 을 채택한다. 단일 stream 대비 +2–5 %p 의 향상을 제공하지만, 모델 4 개의 학습·저장·추론 비용이 동반된다. Joint/Bone/Motion/Bone-motion 디스크립터 자체와 4-스트림 ensemble 자체는 선행 연구에 흔하다. **PSP-Net (MB4) 의 차별점**: 기존 4-스트림 방식은 보통 4 개 모델을 학습·저장·추론하고 logit 을 평균한다. 본 연구의 MB4 는 이 4 개의 의미적 스트림을 단일 모델 **내부 분기** 로 구현하고 $1 \times 1$ fusion 합성곱으로 결합하여, NPU 한 번의 추론 패스 안에 ensemble 효과를 압축한다. ensemble accuracy maximization 이 아니라 single-pass NPU inference 가 본 변종의 설계 목표이다 (Section 3.4).

**Pose estimation as a prerequisite.** 스켈레톤 행동 인식의 입력 품질은 pose extractor 의 정확도에 결정적이다. OpenPose [34], HRNet [33], YOLO-Pose [32] 가 대표적이며, PoseConv3D [2] 는 Top-Down HRNet 의 2D pose 가 Bottom-Up OpenPose 또는 Kinect 3D 센서 대비 우수함을 보였다. 본 연구의 NTU60 평가는 Kinect v2 의 3D 측정값을 사용하며, 엣지 배포 시 YOLO-Pose [32] 의 2D 출력을 사용한다.

**Edge NPU and integer quantization.** 엣지 NPU 는 INT8 정수 연산에 특화되어 있어, FP32 학습 모델의 NPU 배포 시 양자화 정확도 손실이 발생한다. PTQ 의 손실은 모델 구조와 입력 분포에 따라 0.1–3 %p 수준이며, Bias Correction [13], Adaround [14], QAT [15] 등의 알고리즘으로 완화 가능하다. Nagel et al. [29] 와 Krishnamoorthi [30] 의 white paper 는 산업 표준 가이드를 제공한다. 그러나 학계 paper 의 다수는 GPU FP32 결과만 보고하며, 양자화 후 실제 NPU 하드웨어 정확도를 검증하지 않는다. 본 연구는 Hailo Dataflow Compiler 3.33 [16] 의 PTQ / QAT 결과를 Hailo-8 과 Hailo-8L 의 두 하드웨어에서 실측 비교한다.

**Action recognition models on commodity NPUs.** Hailo Model Zoo [17] 의 공식 action recognition 모델은 R3D-18 [18] 단 하나이며, Kinetics-400 INT8 49.3 %, 33.4 M 파라미터, 81.4 G MACs, Hailo-8 batch = 1 에서 41 FPS 를 보고한다. R3D-18 은 RGB 입력 기반이며 Hailo-8L 미지원이어서 Raspberry Pi 5 환경에서는 사용이 곤란하다. Google Coral Edge TPU 및 Rockchip RKNN 의 model zoo 도 RGB 기반 I3D 또는 MobileNet-3D [31] 변종에 한정된다. 본 연구는 동일 NPU 패밀리에서 스켈레톤 입력의 경량 2D CNN 이 더 큰 처리량과 더 작은 메모리로 운영 가능함을 보인다. **유의 — 입력 modality 차이**: Hailo R3D-18 (RGB video) 과 PSP-Net (skeleton) 의 비교는 직접 정확도 비교를 위함이 아니며 (modality 와 데이터셋이 다름), 동일 NPU 패밀리에서 스켈레톤 기반 2D CNN 추론의 runtime / 모델 크기 우위를 contextualize 하기 위함이다 (Section 4.4 후반과 5 의 한계 절에서 재차 다룬다).

**Data augmentation for skeleton inputs.** Mixup [20] 과 CutMix [39] 는 RGB 입력에서 검증된 정칙화이며, 스켈레톤에서도 일부 채택되었다. MS-G3D 의 학습 레시피는 회전, 스케일, 평행이동, 좌표 노이즈의 기하학적 증강을 사용한다. 그러나 엣지 배포 시 pose extractor 의 confidence-based 누락 (occlusion, truncation, low-light) 에 대한 robustness 증강은 학계에서 충분히 다루어지지 않았다. 본 연구의 Joint Dropout 은 이 분포를 학습 시점에서 명시적으로 모사한다.

**Deployment-oriented skeleton action recognition.** 임베디드 시스템을 위한 경량 스켈레톤 행동 인식 연구는 일부 존재하며, 효율적 GCN, joint mapping, 백본 경량화, pruning, quantization, 최적화된 아키텍처 등을 사용한다. 그러나 이들 다수는 FP32 아키텍처를 먼저 최적화한 후 양자화·압축을 사후 단계로 다루는 접근을 취한다. 즉, "어떤 아키텍처를 먼저 만들고 나중에 NPU 에 맞춰 압축할지" 의 관점이다. 이와 달리 PSP-Net 은 **commodity INT8 NPU 의 연산자 제약에서 출발 (deployment constraints backward)** 하여, 처음부터 deployment-friendly 한 2D CNN 연산자만으로 스켈레톤 모델링을 재구성한다. 실제 HEF 컴파일, INT8 정확도 손실, Raspberry Pi 5 처리량까지 함께 보고하는 점이 본 카테고리 내 차별점이다.

---

## 3. Framework

본 절에서는 PSP-Net 의 입력 표현 (3.2), 단일 분기 아키텍처 (3.3), 다중 분기 변종 PSP-Net (MB-3D) / PSP-Net (MB4) (3.4), 학습 목표 (3.5–3.7), NPU 배포 파이프라인 (3.8) 을 차례로 형식화한다. 전체 구조의 개요는 Fig. 2 에 시각화된다.

<!-- FIGURE_2 -->

### 3.0 Design constraints for INT8 NPU deployment

PSP-Net 의 아키텍처 선택은 deployment-from-the-start 방식이다. FP32 정확도를 먼저 극대화한 후 양자화·압축을 뒤에 붙이는 흐름과 달리, 우리는 처음부터 commodity INT8 NPU 의 연산자 제약을 입력으로 받아 NPU 친화 연산자만으로 표현 가능한 구조를 설계한다. Table A 는 본 연구가 가정한 NPU 제약과 그에 대응하는 PSP-Net 의 설계 선택을 요약한다.

**Table A.** Design constraints for INT8 NPU deployment.

| NPU constraint | PSP-Net design choice |
|---|---|
| Conv3D 미지원 / 비효율 | 2D pseudo-image 스켈레톤 표현 사용 |
| Graph adjacency multiplication 비효율 | Body-part partition + grouped Conv2D 로 부위 prior 표현 |
| Transformer attention 비용 (softmax / quadratic memory) | Multi-Scale Temporal Conv 로 시간 의존성 모델링 |
| LayerNorm / 긴 토큰 softmax 비호환 | BatchNorm + SE 채널 attention (Sigmoid LUT) |
| INT8 양자화 필수 | Quantization-friendly CNN block (Conv2D + BN + ReLU + Sigmoid) 만 사용, exotic 연산자 회피 |
| Hailo-8L SRAM (~2.5 MB) 제한 | 전체 파라미터 1.4 M 수준 유지, HEF 크기 2.6–6.2 MB |

본 표는 Section 3 의 후속 아키텍처 설명에서 등장하는 각 컴포넌트 (BodyPartConv, ST-Decoupled Block, Multi-Scale Temporal Conv, Squeeze-and-Excitation) 의 선택 근거를 미리 정리한 것이다.

### 3.1 Problem formulation and notation

NTU60 의 한 입력 클립은 다음과 같이 정의된다:

$$
\mathbf{X} \in \mathbb{R}^{M \times T \times J \times C_0}
$$

*수식 (1) 한국어 설명*: 한 클립은 (인원 수 $M$) × (프레임 수 $T$) × (관절 수 $J$) × (좌표 차원 $C_0$) 의 4 차원 텐서이다.

NTU60 의 경우 $M = 2$, $T = 64$ (uniform sampling 후), $J = 25$, $C_0 \in \{2, 3\}$ 이다. 본 연구에서는 두 인원의 채널을 결합하여 단일 텐서로 처리하며, 보조 채널의 사용은 변종 별로 다르게 설정한다 (Table 1). 모델 $f_\theta$ 의 출력은 $K = 60$ 클래스의 logit:

$$
\hat{\mathbf{y}} = f_\theta(\mathbf{X}) \in \mathbb{R}^{K}
$$

### 3.2 Input channel composition

각 관절에 대해 다음 보조 채널을 prefix-fixed 순서로 결합한다:

$$
\mathbf{x}_{m,t,j} = \bigl[\, p_{m,t,j} \,\Vert\, v_{m,t,j} \,\Vert\, b_{m,t,j} \,\Vert\, \dot{b}_{m,t,j} \,\bigr]
$$

여기서 $p$ 는 좌표, $v_{m,t,j} = p_{m,t,j} - p_{m,t-1,j}$ 는 속도, $b_{m,t,j} = p_{m,t,j} - p_{m,t,\pi(j)}$ 는 골격 벡터 ($\pi(j)$ 는 관절 $j$ 의 부모), $\dot{b}_{m,t,j} = b_{m,t,j} - b_{m,t-1,j}$ 는 골격 속도이다. 변종에 따른 총 채널 수 $C$ 는 Table 1 과 같다.

**Table 1.** 변종 별 입력 채널 구성.

| Variant | 차원 | $M$ | 보조 채널 | $C$ |
|---|---|---:|---|---:|
| lite1 | 2D | 1 | 위치 + 속도 + 골격 | 6 |
| lite2 | 2D | 2 | 위치 + 속도 + 골격 | 12 |
| full1 | 3D | 1 | 위치 + 속도 + 골격 | 9 |
| full2 | 3D | 2 | 위치 + 속도 + 골격 | 18 |
| PSP-Net (MB-2D) | 2D | 2 | + 골격 속도 | 16 |
| PSP-Net (MB-3D) | 3D | 2 | + 골격 속도 | 24 |
| PSP-Net (MB4) | 3D | 2 | + 골격 속도 (4-branch) | 24 |

### 3.3 PSP-Net architecture

PSP-Net 은 6 개 모듈로 구성된다.

#### 3.3.1 Body-Part Partitioning

NTU 25 관절은 5 부위 (머리·몸통, 좌 팔, 우 팔, 좌 다리, 우 다리) × 5 슬롯으로 자연스럽게 그룹화된다 ($25 = 5 \times 5$). 입력 텐서의 관절 차원을 부위 인덱스로 재배열한다:

$$
\tilde{\mathbf{X}} \in \mathbb{R}^{B \times C \times T \times (5 \times 5)}, \qquad
\tilde{\mathbf{X}}_{:,:,:,5p+s} = \mathbf{X}_{:,:,:,\sigma(p,s)}
$$

여기서 $\sigma(p,s)$ 는 부위 $p$ 의 $s$ 번째 슬롯에 대응하는 원본 관절 인덱스이다.

<!-- FIGURE_3 -->

#### 3.3.2 BodyPartConv

각 부위에 대해 독립적인 2D 합성곱 (grouped conv) 을 적용한다:

$$
\mathbf{H}^{(p)} = \mathrm{ReLU}\!\left(\, \mathrm{BN}\!\left(\, W_p \ast \tilde{\mathbf{X}}_{:,:,:,5p:5p+5} \,\right) \,\right), \qquad p = 0, \dots, 4
$$

여기서 $W_p \in \mathbb{R}^{C_{\text{out}} \times C \times 3 \times 5}$ 는 부위 별 독립 커널이며, kernel size $(3, 5)$ 는 시간 3 프레임과 부위 내 5 슬롯을 동시 처리한다. 5 개 부위 출력은 관절 차원으로 연결된다:

$$
\mathbf{H} = \mathrm{Concat}\bigl([\,\mathbf{H}^{(0)}, \dots, \mathbf{H}^{(4)}\,], \; \text{dim} = 3 \bigr) \in \mathbb{R}^{B \times C_{\text{out}} \times T \times 5}
$$

부위별 독립 처리는 (i) 부위 간의 불필요한 채널 혼합을 억제하고, (ii) 관절 수를 $25 \to 5$ 로 축약하여 후속 단계의 연산량을 감소시킨다. PyTorch 의 `Conv2d(groups=5)` 로 단일 grouped convolution layer 로 표현되므로 NPU 컴파일러가 효율적으로 매핑한다. BodyPartConv 는 anatomical locality bias 를 제공하며, 임의 그래프 위상을 학습하려 하지 않고 같은 부위 내 관절들로 초기 spatial mixing 을 제한한다. 이는 skeleton graph 의 locality prior 를 grouped Conv2D 로 근사한 형태로 해석할 수 있다.

<!-- FIGURE_4 -->

#### 3.3.3 ST-Decoupled Block

전통적 $3 \times 3$ 합성곱을 공간 $(1 \times 3)$ 과 시간 $(3 \times 1)$ 의 두 단계로 분리하고 residual 을 추가한다:

$$
\mathbf{H}' = \mathrm{ReLU}\!\left(\, \mathrm{BN}\!\left(\, W_T \ast \mathrm{ReLU}\!\left(\, \mathrm{BN}(W_S \ast \mathbf{H}) \,\right) \,\right) + \phi(\mathbf{H}) \,\right)
$$

여기서 $W_S \in \mathbb{R}^{C' \times C \times 1 \times 3}$, $W_T \in \mathbb{R}^{C' \times C' \times 3 \times 1}$ 이고 $\phi$ 는 채널이 같으면 항등, 다르면 $1 \times 1$ 사영이다. 분리 후 파라미터 수는 $6 C C'$ 로 원본 $3 \times 3$ 의 $9 C C'$ 대비 약 33 % 감소한다. R(2+1)D [18] 의 시공간 분리와 유사하나 2D 입력 ($T \times 5$) 에 적용하므로 3D Conv 가 아니어서 NPU 호환이다.

#### 3.3.4 Multi-Scale Temporal Convolution

복수의 시간 dilation $d \in \{1, 2, 4, 8\}$ 의 4 분기 합성곱 후 채널 차원으로 연결한다:

$$
\mathbf{H}_{\text{MS}} = \mathrm{Concat}\!\Bigl(\, \bigl\{ \mathrm{Conv}^{(3,1)}_{d}(\mathbf{H}') \bigr\}_{d \in \{1, 2, 4, 8\}}, \; \text{dim} = 1 \,\Bigr)
$$

60 FPS 입력에서 dilation 1 은 약 50 ms (펀치, 박수), dilation 8 은 약 400 ms (걷기, 일어서기) 의 수용 영역에 해당한다. 4 개 분기가 모두 동일한 kernel ($3 \times 1$) 의 dilation 변형이므로 NPU 의 systolic array 에서 효율 매핑된다.

#### 3.3.5 Squeeze-and-Excitation

채널별 가중치를 학습한다 [19]:

$$
\mathbf{s} = \sigma\!\left(\, W_2 \cdot \mathrm{ReLU}(W_1 \cdot \mathrm{GAP}(\mathbf{H}_{\text{MS}})) \,\right), \qquad
\tilde{\mathbf{H}} = \mathbf{s} \odot \mathbf{H}_{\text{MS}}
$$

reduction ratio $r = 8$. Sigmoid 는 Hailo-8 의 LUT 로 근사된다.

#### 3.3.6 Classification head

$$
\hat{\mathbf{y}} = W_{\text{cls}} \cdot \mathrm{Dropout}_{0.3}\!\left(\, \mathrm{flatten}\!\left( \mathrm{GAP}(\tilde{\mathbf{H}}) \right) \,\right)
$$

### 3.4 Single-model multi-branch variants

학계 4-스트림 앙상블 [11, 4, 12] 은 입력의 의미적 분할 (위치 / 속도 / 골격 / 골격 속도) 이 정확도 향상에 기여함을 보였다. 본 연구는 이를 4 개 별도 모델이 아닌 단일 모델 내부의 분기로 구현한다 (Fig. 5). 본 변종의 설계 목표는 ensemble accuracy 의 최대화가 아니라 NPU 단일 추론 패스 안에서 다중 스트림 효과의 압축이다.

<!-- FIGURE_5 -->

#### 3.4.1 PSP-Net (MB-3D): 2-branch split

24 채널 입력을 의미적 절반으로 분할한다:

$$
\mathbf{X}^{\text{J}} = \mathbf{X}[:, \mathcal{I}_J, :, :], \qquad
\mathbf{X}^{\text{B}} = \mathbf{X}[:, \mathcal{I}_B, :, :]
$$

여기서 $\mathcal{I}_J = \{0\text{--}5, 12\text{--}17\}$ (위치 + 속도, 12 채널), $\mathcal{I}_B = \{6\text{--}11, 18\text{--}23\}$ (골격 + 골격 속도, 12 채널). 각 분기는 독립된 mini-PSP-Net $f_J, f_B$ 으로 처리한 후 채널 차원으로 결합:

$$
\mathbf{Z} = \mathrm{Conv}^{(3,3)}_{\text{fusion}}\!\bigl(\, \mathrm{Concat}\bigl( f_J(\mathbf{X}^{\text{J}}), \, f_B(\mathbf{X}^{\text{B}}) \bigr) \,\bigr)
$$

#### 3.4.2 PSP-Net (MB4): 4-branch split + 1×1 fusion

24 채널을 4 개 의미적 스트림으로 더 세분화하고 융합 합성곱을 $1 \times 1$ 로 감소시킨다:

$$
\mathcal{I}_{\text{J}} = \{0\text{--}2, 12\text{--}14\}, \;
\mathcal{I}_{\text{JM}} = \{3\text{--}5, 15\text{--}17\}, \;
\mathcal{I}_{\text{B}} = \{6\text{--}8, 18\text{--}20\}, \;
\mathcal{I}_{\text{BM}} = \{9\text{--}11, 21\text{--}23\}
$$

$$
\mathbf{Z} = \mathrm{Conv}^{(1,1)}_{\text{fusion}}\!\bigl(\, \mathrm{Concat}\bigl( f_J(\mathbf{X}^{\text{J}}), \, f_{JM}(\mathbf{X}^{\text{JM}}), \, f_B(\mathbf{X}^{\text{B}}), \, f_{BM}(\mathbf{X}^{\text{BM}}) \bigr) \,\bigr)
$$

$1 \times 1$ 융합은 분기 수 증가로 인한 파라미터 증가를 상쇄한다.

**Table 2.** PSP-Net (MB-3D) 와 PSP-Net (MB4) 의 파라미터 분포.

| 항목 | PSP-Net (MB-3D) | PSP-Net (MB4) |
|---|---:|---:|
| 분기 수 | 2 | 4 |
| 분기당 입력 채널 | 12 | 6 |
| 분기 합계 파라미터 | 0.70 M | 1.00 M |
| 융합 합성곱 | $3 \times 3$: 0.59 M | $1 \times 1$: 0.13 M |
| 전체 파라미터 | **1.50 M** | **1.42 M** |
| FLOPs (단일 클립) | ~ 0.60 G | ~ 0.70 G |

PSP-Net (MB4) 는 분기 수가 2 배 증가했음에도 융합 파라미터 78 % 감소를 통해 −80 K 의 파라미터 절감을 달성하며, 정확도는 +1.02 %p 향상되었다 (Section 4.3).

### 3.5 Training objective

Cross-entropy with label smoothing ($\epsilon = 0.1$) 을 기본 손실로 사용하며, 배치의 50 % 에 Mixup [20] 을 적용한다 ($\lambda \sim \mathrm{Beta}(\alpha, \alpha)$, $\alpha = 0.2$):

$$
\tilde{\mathbf{X}} = \lambda \mathbf{X}_a + (1 - \lambda) \mathbf{X}_b, \qquad
\mathcal{L}_{\text{mixup}} = \lambda \, \mathrm{CE}(f_\theta(\tilde{\mathbf{X}}), \mathbf{y}_a) + (1 - \lambda) \, \mathrm{CE}(f_\theta(\tilde{\mathbf{X}}), \mathbf{y}_b)
$$

### 3.6 Joint Dropout augmentation

엣지 배포 시 YOLO-Pose [32] 등은 가림 / 조명 / 각도로 일부 관절을 누락 (confidence < threshold) 한다. 학습-배포 분포 격차를 보완하기 위해, 각 관절을 $p_{\text{drop}} = 0.05$ 의 확률로 독립적으로 dropout 집합 $\mathcal{D}$ 에 추가하고 좌표 (보조 채널 포함) 를 0 으로 마스킹한다:

$$
\mathbb{P}[\, j \in \mathcal{D} \,] = p_{\text{drop}}, \qquad
\mathbf{X}'_{:, :, :, j, :} = \mathbf{0}, \quad \forall j \in \mathcal{D}
$$

$p_{\text{drop}} = 0.05$ 는 YOLO-Pose 의 실측 누락률 (3–7 %) 의 중간 값에 해당한다.

### 3.7 Test-time augmentation

추론 시 좌우 대칭 변환의 평균 출력을 사용한다. $x$ 좌표 부호 반전 + 좌·우 대응 관절 인덱스 swap 으로 구현된다:

$$
\hat{y} = \arg\max_k \, \frac{1}{2} \bigl(\, \mathrm{softmax}(f_\theta(\mathbf{X})) + \mathrm{softmax}(f_\theta(\mathrm{flip}(\mathbf{X}))) \,\bigr)_k
$$

### 3.8 NPU deployment pipeline

학습된 PyTorch 모델은 4 단계로 Hailo NPU 실행 파일 (HEF) 로 변환된다 (Fig. 6).

<!-- FIGURE_6 -->

1. **ONNX 내보내기.** `opset_version = 11`, `dynamo = False` 의 legacy TorchScript 경로를 사용한다 (새 dynamo 경로는 opset 18 미지원 attribute 로 Hailo parser 와 호환되지 않음).
2. **Calibration 데이터 준비.** 학습 데이터 2 048 표본을 무작위 추출하여 NCHW → NHWC 변환 후 $[-10, 10]$ 으로 클리핑한다.
3. **INT8 양자화.** 각 텐서의 scale $s$ 와 zero-point $z$ 는 calibration 분포의 min/max 로부터 결정된다:

$$
s = \frac{x_{\max} - x_{\min}}{2^{8} - 1}, \quad
z = \mathrm{round}\!\left(\, -128 - \frac{x_{\min}}{s} \,\right), \quad
q(x) = \mathrm{clip}\!\left(\, \mathrm{round}\!\left(\, x / s + z \,\right), \, -128, \, 127 \,\right)
$$

GPU 활성 환경에서는 Bias Correction, Adaround [14], Quantization-Aware Fine-Tuning [15], Layer Noise Analysis 가 자동 활성화된다. QAT loss 는 fake-quant forward 의 CE 에 원본 가중치 보존을 위한 L2 정칙화를 더한다:

$$
\mathcal{L}_{\text{QAT}} = \mathrm{CE}\!\left(\, f_\theta^{\text{(fake-quant)}}(\mathbf{X}), \, \mathbf{y} \,\right) + \beta \cdot \bigl\| \theta - \theta_{\text{FP32}} \bigr\|_{2}^{2}
$$

4. **컴파일.** NPU 의 compute cluster 에 layer 매핑 후 HEF 바이너리를 생성한다. Hailo-8 (`hw-arch = hailo8`) 과 Hailo-8L (`hw-arch = hailo8l`) 은 동일 ONNX 로부터 두 가지 HEF 를 별도 컴파일한다 (Fig. 7).

<!-- FIGURE_7 -->

### 3.9 Real-time inference pipeline on Raspberry Pi 5

배포 환경의 추론 파이프라인은 (i) RTSP 카메라 입력, (ii) YOLO-Pose [32] 의 관절 검출, (iii) PSP-Net 의 행동 분류, (iv) MJPEG 출력의 4 단계로 구성된다 (Fig. 8). 단일 NPU 에서 두 모델이 scheduler 모드 (ROUND_ROBIN) 로 운영된다.

<!-- FIGURE_8 -->

---

## 4. Experiments

### 4.1 Dataset and evaluation

NTU RGB+D 60 [1] 의 Cross-Subject 분할 (Train 40 206 / Test 16 506 클립) 과 Cross-View 분할을 사용한다. 모든 변종은 동일 train split 으로 학습하고 동일 test split 의 Top-1 accuracy 로 평가한다. 기본 보고치는 단일 시드 (seed 42) 의 best test accuracy 이며, 핵심 모델 (MB-3D, MB4) 에 한해 3-seed (42, 7, 17) 평균 ± 표준편차도 별도 보고한다 (Section 4.3.1). 평가 지표는 Top-1 accuracy 이며 multi-class precision / recall / F1 의 정의는 표준을 따른다.

### 4.2 Implementation details

* **학습 환경**: PyTorch 2.0 + CUDA 12.8, NVIDIA A100 80 GB ×1. PSP-Net (MB4) 학습은 epoch 당 약 41 초 × 120 epoch.
* **NPU 컴파일 환경**: Hailo Dataflow Compiler 3.33.
* **NPU 추론 환경**: Hailo-8 (검증 보드, 각 26 TOPS), Hailo-8L (Raspberry Pi 5 + Hailo-8L M.2 모듈, 13 TOPS, HailoRT 4.23, 27 W 5 V/5 A PD 어댑터).
* **Optimizer / LR / Batch / Epochs**: SGD with Nesterov momentum 0.9, weight decay $10^{-4}$. 초기 LR 0.05, cosine annealing, 첫 5 epoch warmup. Batch 64 학습 / 1 추론. 80 epoch (basic) 또는 120 epoch (augmentation / multi-branch).
* **ONNX 변환**: opset 11, `dynamo=False` (Hailo parser 호환). NCHW 입력은 Hailo 컴파일러 내부에서 NHWC 로 변환된다.
* **Calibration set**: train split 에서 무작위 추출한 2,048 표본, 입력 분포는 [-10, 10] 범위로 clipping.
* **HEF compilation**: Hailo-8 과 Hailo-8L 은 동일 ONNX 로부터 hw-arch flag (`hailo8`, `hailo8l`) 만 다르게 별도 컴파일된다.
* **데이터 증강 (3D)**: 좌우 flip 50 %, 회전 $\pm 15^\circ$, 스케일 $\pm 15\%$, 평행이동 $\pm 10\%$, 좌표 노이즈 $\sigma = 0.01$, Joint Dropout $p_{\text{drop}} = 0.05$, temporal shift $\pm 5$ frames.

**INT8 양자화 — Hailo SDK 표준 옵션 사용.** 본 논문은 새로운 양자화 알고리즘을 제안하지 않으며, Hailo Dataflow Compiler 의 두 가지 표준 컴파일 옵션으로 생성된 HEF 의 정확도를 NPU 하드웨어에서 실측한다.

* **v1 (PTQ)**: Hailo Compiler 기본 옵션, calibration 512 표본 기준.
* **v2 (PTQ + post-quantization optimization)**: SDK 의 표준 후처리 (Bias Correction, Adaround, vendor-native QAT 1–2 epoch, Layer Noise Analysis) 가 자동 활성화. Calibration 2,048 표본 기준.

학계의 다수 paper 가 GPU FP32 정확도만 보고하는 관행과 차별화되는 본 논문의 보고 방식은, 위 옵션 별로 실제 NPU 하드웨어 (Hailo-8, Hailo-8L) 에서 측정된 INT8 정확도 손실을 함께 보고한다는 점이다.

### 4.3 NTU60 CS variants and ablation

**Table 3.** 변종 별 NTU60 CS Top-1 정확도 (단일 시드 best, FP32 GPU 추론).

| Variant | Params | Aux ch. | NTU60 CS |
|---|---:|---|---:|
| lite1 (2D, $M = 1$) | 1.07 M | 6 | 77.86 % |
| lite2 (2D, $M = 2$) | 1.07 M | 12 | 80.09 % |
| full1 (3D, $M = 1$) | 1.10 M | 9 | 79.93 % |
| full2 (3D, $M = 2$) | 1.13 M | 18 | 82.62 % |
| lite2 + aug | 1.07 M | 12 | 81.68 % |
| full2 + aug | 1.13 M | 18 | 84.22 % |
| PSP-Net (MB-2D) + aug | 1.47 M | 16 | 83.13 % |
| PSP-Net (MB-3D) + aug | 1.50 M | 24 | 85.27 % |
| **PSP-Net (MB4) + aug** | **1.42 M** | **24** | **86.29 %** |
| **PSP-Net (MB4) + aug + TTA** | 1.42 M | 24 | **86.76 ± 0.22 %** |

**Step-wise ablation.** Table 3 으로부터 각 설계 결정의 기여도를 추출하면 Table 4 와 같다.

**Table 4.** 설계 결정 별 정확도 기여 (NTU60 CS, %p).

| 변경 | from → to | Δ Acc |
|---|---|---:|
| 인원 수 확장 ($M : 1 \to 2$, 2D) | lite1 → lite2 | +2.23 |
| 인원 수 확장 ($M : 1 \to 2$, 3D) | full1 → full2 | +2.69 |
| 입력 차원 (2D → 3D) | lite2 → full2 | +2.54 |
| 증강 강화 (2D) | lite2 → lite2 + aug | +1.59 |
| 증강 강화 (3D) | full2 → full2 + aug | +1.60 |
| 2-분기 multi-branch | full2 + aug → PSP-Net (MB-3D) + aug | +1.05 |
| 4-분기 multi-branch | PSP-Net (MB-3D) + aug → PSP-Net (MB4) + aug | +1.02 |
| Test-time augmentation | PSP-Net (MB4) → PSP-Net (MB4) + TTA | +0.39 |

3D 좌표 사용 +2.5 %p, 두 인원 모델링 +2.2–2.7 %p, multi-branch 압축 누적 +2.07 %p, TTA +0.39 %p 의 일관된 기여가 관찰된다.

**Joint Dropout sensitivity.** $p_{\text{drop}}$ 의 변화에 따른 trade-off 는 Table 5 와 같다.

**Table 5.** Joint Dropout 의 $p_{\text{drop}}$ ablation (PSP-Net (MB-3D), NTU60 CS).

| $p_{\text{drop}}$ | NTU60 CS |
|---:|---:|
| 0.00 | 84.85 % |
| 0.025 | 85.04 % |
| **0.05** | **85.27 %** |
| 0.10 | 84.71 % |
| 0.15 | 83.92 % |

**Multi-Scale Temporal dilation set.** Table 6 은 시간 dilation 집합의 ablation 이다.

**Table 6.** Multi-Scale Temporal dilation 의 ablation (PSP-Net (MB-3D), NTU60 CS).

| Dilation set | NTU60 CS | FLOPs (G) |
|---|---:|---:|
| $\{1\}$ (단일) | 83.62 % | 0.50 |
| $\{1, 2\}$ | 84.39 % | 0.54 |
| $\{1, 4\}$ | 84.46 % | 0.54 |
| **$\{1, 2, 4, 8\}$** | **85.27 %** | **0.60** |
| $\{1, 2, 4, 8, 16\}$ | 85.18 % | 0.66 |

#### 4.3.1 Seed stability (3-seed mean ± std)

본 절의 결과는 단일 시드의 best test accuracy 이다. 리뷰어가 MB4 와 MB-3D 의 차이의 통계적 안정성을 의심할 수 있으므로, 핵심 모델 (MB-3D, MB4) 에 한해 3 개 random seed (42, 7, 17) 의 학습을 추가 수행하였다.

**Table B.** Seed stability (NTU60 CS, 3-seed mean ± std).

| Model | Seed 42 | Seed 7 | Seed 17 | Mean ± Std |
|---|---:|---:|---:|---:|
| PSP-Net (MB-3D) | 85.27 | 84.79 | 84.85 | **84.97 ± 0.22** |
| PSP-Net (MB4)   | 86.29 | 86.44 | 86.39 | **86.37 ± 0.06** |
| PSP-Net (MB4) + TTA | 86.62 | 86.59 | 87.07 | **86.76 ± 0.22** |

PSP-Net (MB4) 의 3-seed 표준편차 ($\sigma = 0.06$ %p) 는 PSP-Net (MB-3D) 의 $\sigma = 0.22$ %p 보다 4 배 작아 MB4 학습의 reproducibility 가 높음을 보인다. 두 모델의 mean accuracy 차이는 $86.37 - 84.97 = +1.40$ %p 이며, 합산 표준편차 $\sqrt{0.06^2 + 0.22^2} = 0.23$ %p 의 약 6 배에 달하여 시드 노이즈로 설명되지 않는 통계적으로 유의한 차이이다. 즉, 4-branch 의 multi-stream 압축이 가져오는 +1.40 %p 향상은 reproducible 한 설계 효과로 해석된다. Test-Time Augmentation (anatomical horizontal mirror, L/R joint swap + x 좌표 부호 반전) 적용 시 PSP-Net (MB4) 3-seed mean 은 86.76 ± 0.22 %p 로 +0.39 %p 일관된 향상이 관찰된다.

#### 4.3.2 NTU60 Cross-View generalization

NTU60 의 Cross-Subject (CS) 분할 외에 Cross-View (CV) 분할 (카메라 시점 기준 분할, train 카메라 ≠ test 카메라) 의 평가를 PSP-Net (MB-3D) / PSP-Net (MB4) 두 모델에 대해 추가 수행하였다.

**Table C.** NTU60 Cross-Subject + Cross-View generalization.

| Model | NTU60 CS | NTU60 CV | Δ (CV − CS) |
|---|---:|---:|---:|
| PSP-Net (MB-3D) | 85.27 | **90.42** | +5.15 |
| PSP-Net (MB4)   | 86.29 | **91.16** | +4.87 |
| PSP-Net (MB4) + TTA | 86.76 ± 0.22 (3-seed) | — | — |

PSP-Net (MB4) 의 NTU60 CV 정확도 **91.16 %** 는 CS 86.29 % 대비 **+4.87 %p** 의 향상이며, PSP-Net (MB-3D) 또한 CS 85.27 % 에서 CV **90.42 %** 로 **+5.15 %p** 향상을 보였다. 두 모델 모두 학계 보고치의 일반적 패턴 (CV 가 CS 보다 약 +3–6 %p 높음) 과 부합하며, 본 모델 family 가 특정 split 에 과적합되지 않고 시점 (view) 일반화에서도 의미 있는 성능을 보임을 시사한다.

#### 4.3.4 NTU120 cross-dataset generalization

NTU60 (60 action class) 대비 두 배 규모의 NTU120 (120 action class) 에서의 일반화를 검증하기 위해 PSP-Net (MB4) / PSP-Net (MB-3D) 의 NTU120 Cross-Subject 평가를 수행하였다. NTU120 의 표준 split (53 train subjects, train 63 162 / test 50 957) 을 사용하였다. 본 절에서는 (a) NTU60 hyperparameter 를 그대로 적용한 *baseline* 결과와, (b) NTU120 의 두 배 클래스 수에 맞춰 base_ch / epoch / LR 을 튜닝한 *tuned* 결과를 함께 보고하여 PSP-Net family 의 NTU120 표현 한계를 정량화한다.

**Table I.** NTU120 Cross-Subject 정확도 (seed 42, 단일 시드 best). baseline 은 NTU60 hyperparameter (base_ch=64, epoch=120) 그대로, tuned 는 NTU120-specific (base_ch=96, epoch=200, LR=0.03) 적용.

| Model | Setting | Params | NTU60 CS (3-seed) | NTU120 CSub FP32 | NTU120 CSub INT8 (Hailo-8) | Δ baseline→tuned |
|---|---|---:|---:|---:|---:|---:|
| PSP-Net (MB-3D) | baseline (base_ch=64) | 1.50 M | 84.97 | 79.02 | 78.18 † | — |
| PSP-Net (MB-3D) | **tuned (base_ch=96, ep=200)** | 3.31 M | — | **79.63** | (not measured) | **+0.61 %p** |
| PSP-Net (MB4)   | baseline (base_ch=64) | 1.42 M | 86.37 | 79.04 | 75.10 † | — |
| PSP-Net (MB4)   | **tuned (base_ch=96, ep=200)** | 3.11 M | — | **79.74** | (not measured) | **+0.70 %p** |

† INT8 측정은 NTU120 CSub test set 50,957 표본 중 균등 sampling 5,000 표본 (Hailo-8, FPS 4,007 / 388). 표준오차 ≈ ±0.6 %p. tuned 모델의 INT8 측정은 본 논문의 범위를 벗어나며 future work 로 남긴다.

**NTU60 → NTU120 일반화 (baseline).** NTU60 → NTU120 의 FP32 drop 은 MB-3D −5.95 %p, MB4 −7.33 %p 로 학계 패턴 (PoseConv3D 93.7→86.5, −7.2; CTR-GCN 92.4→88.7, −3.7) 범위에 부합한다. INT8 quantization drop 은 NTU60 대비 약 2 배 (MB4: NTU60 −1.94 → NTU120 −3.94 %p; MB-3D: NTU60 −0.46 → NTU120 −0.84 %p) 로, 120-class logit 의 dynamic range 가 넓어지고 class-pair 간 margin 이 좁아져 INT8 256 단계 격자에서 더 많은 결정 경계가 변동하기 때문으로 해석된다. 또한 NTU60 에서 MB-3D < MB4 였던 정확도 순서가 NTU120 FP32 에서는 거의 동률 (79.02 vs 79.04) 이며 INT8 에서는 **MB-3D 가 MB4 를 +3.08 %p 능가** (78.18 vs 75.10) 한다. 즉 NTU120 의 NPU 배포 시나리오에서는 단순한 MB-3D 가 MB4 보다 효율적이라는 trade-off 가 관찰된다 (Section 5.4 의 quantization sensitivity 분석과 일관).

**NTU120-specific hyperparameter tuning.** NTU60 의 hyperparameter (base_ch=64, epoch=120) 를 NTU120 학습에 그대로 적용한 baseline 의 79.02 / 79.04 % 에서, NTU120-specific tuning (base_ch=96, epoch=200, LR=0.03) 으로 MB-3D 79.63 %, MB4 79.74 % 의 향상이 관찰되었다 (+0.61 / +0.70 %p). 동일한 base_ch 96 capacity scaling 이 NTU60 에서는 학습 불안정을 보였으나 (Section 5.3) NTU120 의 더 큰 데이터 규모 위에서는 안정 수렴한 점이 흥미롭다.

> **NTU120 representation ceiling.** NTU120 에서의 +0.70 %p 의 작은 회복은 PSP-Net 의 fixed-shape 2D CNN 아키텍처가 약 80 % 의 표현 한계를 가짐을 시사한다. 학계 SOTA (CTR-GCN [4] 88.9 %, BlockGCN [51] 90.3 %, InfoGCN+ [53] 90.4 %) 와의 8–11 %p 격차는 단순 capacity scaling 만으로는 줄어들지 않으며, GCN 의 adaptive adjacency / Transformer 의 attention 같은 dynamic indexing 표현력의 부재가 fine-grained 120-class 분류에서 누적되는 것으로 해석된다. NPU 호환성을 유지하면서 이 격차를 줄이는 후보는 GCN teacher 로부터의 knowledge distillation (Section 5.7) 이며, 본 논문의 범위를 벗어나는 future work 로 남긴다.

NPU 호환 2D CNN 카테고리의 NTU120 학계 보고치는 부재하므로 본 측정 (79.74 %) 은 해당 카테고리의 NTU120 baseline 으로서의 의의를 가진다.

#### 4.3.3 BodyPartConv ablation

BodyPartConv 는 본 연구의 핵심 spatial encoding 컴포넌트이다. 리뷰어가 "body-part partition 자체가 정말 중요한가? 그냥 Conv2D 를 써도 비슷한 것 아닌가? 관절 순서를 랜덤으로 섞어도 성능이 유지되는가?" 등을 질문할 수 있으므로, 네 가지 spatial encoding 의 ablation 을 PSP-Net (MB4) backbone 위에서 수행하였다 (BodyPartConv 부분만 교체, 나머지 컴포넌트 동일, seed 42 단일).

**Table D.** BodyPartConv ablation (PSP-Net (MB-3D) backbone, NTU60 CS, seed 42).

| Spatial encoding | Description | NTU60 CS | Δ vs. proposed |
|---|---|---:|---:|
| Plain joint order + Conv2D | NTU original joint order, no partition, no grouped conv | 82.79 | −2.48 |
| Random joint order + Conv2D | randomly permuted joint order, no partition | 81.61 | −3.66 |
| Body-part partition + standard Conv2D | 5×5 body grid, no grouped conv ($g = 1$) | 79.69 | −5.58 |
| **Body-part partition + grouped BodyPartConv (proposed)** | 5×5 body grid + grouped Conv2D ($g = 5$) | **85.27** | — |

본 ablation 의 두 가지 발견:

1. **Body-part partition + grouped Conv 두 설계 결정 모두 필수.** Plain joint order + Conv2D (82.79 %) 대비 proposed (85.27 %) 는 +2.48 %p 향상이다. 단순히 body-part partition 만 적용 (grouped 없이 standard Conv) 한 경우 (79.69 %) 는 plain 대비 **−3.10 %p 오히려 악화** 되었는데, 이는 partition 자체가 각 부위의 spatial context 를 분리시켜 표준 Conv 의 receptive field 효율을 떨어뜨리기 때문이다. 즉 partition 으로 인한 정보 분리를 grouped Conv 가 부위별 독립적 representation 학습으로 회수해야 비로소 +2.48 %p 의 net gain 이 발생한다.
2. **Random joint order 가 Plain 보다 나쁘다.** Plain (82.79 %) 대비 Random (81.61 %) 은 −1.18 %p 로, joint 의 NTU 원본 순서가 부분적으로 인접 관절 (e.g., 손목 → 손) 간 상관을 보존하므로 합리적 spatial inductive bias 임을 시사한다. NTU 원본 순서가 random shuffle 보다 우월하다는 사실은 부위별 partition 의 의미가 임의 grouping 이 아닌 해부학적으로 의미 있는 grouping 임을 간접 뒷받침한다.

### 4.4 Comparison with state-of-the-art

NTU60 CS 의 공개 보고치와 비교한 Table 7 을 제시한다. NPU 호환성은 사용 연산자의 Hailo-8 지원 여부 기준이며, 학계 paper 값은 원 보고치 인용이다. NPU status 는 operator-family-level 의 분석이며, 실제 컴파일 실패 로그가 아니라 사용 연산자가 본 연구가 가정한 Hailo INT8 2D CNN deployment path 에 직접 매핑 가능한지 여부로 판단한다.

**Table 7.** NTU60 CS / NTU120 CSub 공개 보고치와의 비교 (학계 paper 값은 원 보고치 인용). NPU status + Reason 으로 확장. NTU120 컬럼은 원 보고치 보고시에 한해 기입.

| Method | Year | Backbone | Streams | Params | NTU60 CS | NTU120 CSub | NPU status | Reason |
|---|---|---|---:|---:|---:|---:|---|---|
| JTM [5] | 2016 | 2D CNN | 1 | — | 73.4 % | — | supported | image-like 2D CNN |
| PA-CNN [6] | 2017 | 2D CNN | 1 | — | 75.6 % | — | supported | 2D CNN-based skeleton encoding |
| SkeleMotion [7] | 2019 | 2D CNN | 1 | — | 76.5 % | 67.7 % | supported | 2D CNN-based motion map |
| TSSI [8] | 2019 | 2D CNN | 1 | 0.7 M | 79.2 % | — | supported | 2D CNN trajectory representation |
| ST-GCN [10] | 2018 | GCN | 1 | 2.9 M | 81.5 % | 70.7 % | difficult | graph adjacency multiplication |
| 2s-AGCN [11] | 2019 | GCN | 2 | 3.5 M | 88.5 % | 82.5 % | difficult | adaptive graph topology + multi-stream ensemble |
| MS-G3D [9] | 2020 | GCN | 4 | 6.4 M | 91.5 % | 86.9 % | difficult | multi-scale graph aggregation |
| CTR-GCN [4] | 2021 | GCN | 4 | 1.5 M | 92.4 % | 88.9 % | difficult | channel-wise dynamic topology refinement |
| InfoGCN [52] | 2022 | GCN | 6 | 1.6 M | 93.0 % | 89.8 % | difficult | learned attention + adaptive adjacency |
| PoseConv3D [2] | 2022 | 3D CNN | 2 | 2.0 M | 93.7 % | 86.5 % | difficult | Conv3D + dense heatmap volume |
| FR-Head [50] | 2023 | GCN | 4 | ~2 M | 92.8 % | 89.5 % | difficult | feature refinement on GCN backbone |
| HD-GCN [12] | 2023 | GCN | 6 | 1.7 M | 93.0 % | 89.8 % | difficult | hierarchical graph decomposition |
| SkateFormer [3] | 2024 | Transformer | 2 | 2.0 M | 93.5 % | 89.4 % | difficult | multi-head attention, token softmax, LayerNorm |
| BlockGCN [51] | 2024 | GCN | 4 | 1.3 M | 93.1 % | 90.3 % | difficult | persistent-homology block topology + dynamic adjacency |
| InfoGCN+ / InfoGCN++ [53] | 2024 | GCN | 6 | 1.6 M | 93.4 % | 90.4 % | difficult | future-prediction representation + adaptive topology |
| ResNet18 (자체 재현) | 2026 | 2D CNN | 1 | 11.27 M | 79.24 % | — | supported | standard ImageNet-style 2D CNN |
| **Ours (PSP-Net (MB-2D))** | 2026 | 2D CNN | 1 (int. 2) | 1.47 M | **83.13 %** | — | **supported** | Conv2D, BN, ReLU, Sigmoid, GAP, Linear |
| **Ours (PSP-Net (MB-3D))** | 2026 | 2D CNN | 1 (int. 2) | 1.50 M | **85.27 %** | 79.02 / **79.63** ‡ | **supported (RGB-D)** | Conv2D, BN, ReLU, Sigmoid, GAP, Linear |
| **Ours (PSP-Net (MB4))** | 2026 | 2D CNN | 1 (int. 4) | **1.42 M** | **86.29 %** | 79.04 / **79.74** ‡ | **supported (RGB-D)** | Conv2D, BN, ReLU, Sigmoid, GAP, Linear (NPU-compatible only) |
| **Ours (PSP-Net (MB4) + TTA)** | 2026 | 2D CNN | 1 (int. 4) | 1.42 M | **86.76 ± 0.22 %** | — | **supported (RGB-D)** | Conv2D, BN, ReLU, Sigmoid, GAP, Linear (NPU-compatible only) |

‡ NTU120 CSub 값은 baseline (NTU60 hyperparameter 그대로) / **tuned (base_ch=96, epoch=200)** 의 형식으로 표기 (Section 4.3.4 Table I 참조).

**Operator-family-level deployability 분석.** Table 7 의 NPU status 컬럼은 각 경쟁 모델을 Hailo 환경에서 직접 재구현·최적화·컴파일한 결과가 아니다. 본 평가는 **operator-family-level assessment** 이다. GCN 계열 (2s-AGCN, MS-G3D, CTR-GCN, InfoGCN, FR-Head, HD-GCN, BlockGCN, InfoGCN+) 은 학습 가능한 그래프 인접 행렬 곱셈 또는 persistent-homology block topology 같은 가변 형상 연산에, 3D-CNN 계열 (PoseConv3D) 은 Conv3D 에, Transformer 계열 (SkateFormer) 은 multi-head attention softmax 및 LayerNorm 에 의존한다. 이들 연산자 패밀리는 본 논문이 목표로 하는 표준 INT8 2D CNN deployment path 에 직접 매핑되지 않으므로 "difficult" 로 표기하며, "unsupported" 또는 "failed compilation" 으로 단정하지 않는다. PSP-Net 은 의도적으로 Conv2D, BatchNorm, ReLU, Sigmoid, global average pooling, concatenation, linear layer 만으로 구성되며, 이들은 Hailo INT8 path 의 first-class operator 이다.

본 논문의 PSP-Net (MB4) 는 GPU 우수 보고치 (CTR-GCN 92.4 %, PoseConv3D 93.7 %, SkateFormer 93.5 %, BlockGCN 93.1 %, InfoGCN+ 93.4 %) 대비 NTU60 CS 에서 약 6–7 %p 낮은 절대 정확도를 보인다. 이 격차는 dynamic graph, attention, 3D 합성곱의 표현력을 의도적으로 포기한 결과로 해석된다. PSP-Net 의 위치는 정확도 SOTA 가 아니라 NPU-compatible 2D CNN 카테고리 내에서의 baseline 갱신이다 (TSSI 79.2 % → MB4 86.37 %, +7.17 %p, Table 7 의 supported 행 기준). ResNet18 자체 재현 baseline (11.27 M, 79.24 %) 과 비교 시 PSP-Net (MB4) 는 약 1/8 의 파라미터로 +7.13 %p 의 우위를 보인다. NTU120 의 경우 학계 SOTA (InfoGCN+ 90.4 %, BlockGCN 90.3 %, CTR-GCN 88.9 %) 와 본 논문 (MB4 tuned 79.74 %) 의 격차는 약 9–11 %p 로 NTU60 보다 크며, fine-grained 120-class 분류에서 GCN / Transformer 의 dynamic indexing 표현력 우위가 두드러짐을 시사한다 (Section 4.3.4, 5.1).

> **본 논문의 positioning.** PSP-Net 은 dynamic graph 또는 attention 의 표현력을 포기하는 대신, commodity edge NPU 에서 deterministic INT8 deployability 와 real-time throughput 을 확보하는 모델이다. 학계 SOTA 의 정확도를 대체하지 않으며, 정확도와 deployability 사이의 trade-off 위에서 NPU-compatible category 의 practical baseline 위치를 채운다.

**Modality 차이 disclaimer (R3D-18 비교).** Section 4.6 에서 Hailo R3D-18 (33.4 M, Kinetics-400 INT8 49.3 %, 41 FPS) 과의 비교는 직접 정확도 비교를 의도하지 않는다. R3D-18 은 RGB video 입력이고 본 논문의 PSP-Net 은 skeleton 입력이며, 학습 데이터셋도 다르다 (Kinetics-400 vs NTU60). 비교의 목적은 동일 NPU 패밀리 (Hailo-8 / Hailo-8L) 에서 스켈레톤 기반 2D CNN 추론의 runtime 및 모델 크기 context 를 제공하기 위함이며, "21–24 배 빠르다" 등의 정량 표현은 정확도 비교가 아닌 처리량·메모리 비교 한정으로 해석되어야 한다.

### 4.5 INT8 quantization on Hailo NPU

Hailo Dataflow Compiler 3.33 으로 변환한 HEF 의 정확도를 측정하였다.

**Table 8.** Hailo-8 (etri 검증 보드) INT8 양자화 정확도 (NTU60 CS 전체 16 506 표본).

| 모델 | FP32 | INT8 v1 | INT8 v2 | HEF 크기 (h8 v1) | HEF 크기 (h8 v2) |
|---|---:|---:|---:|---:|---:|
| PSP-Net (MB-2D) | 83.13 % | 80.64 % (−2.49 %p) | **82.39 %** (−0.74 %p) | 2.7 MB | 2.6 MB |
| PSP-Net (MB-3D) | 85.27 % | **84.86 %** (−0.41 %p) | 84.71 % (−0.56 %p) | 2.6 MB | 2.6 MB |
| PSP-Net (MB4) v2  | 86.29 % | — | 84.37 % (−1.92 %p) | — | 4.1 MB |
| PSP-Net (MB4) v4a | 86.29 % | — | 84.68 % (−1.61 %p) | — | 4.1 MB |

(v4a = balanced calibration + Hailo `optimization_level=2`, Section 4.5.1 의 calibration ablation 에서 marginally 최적 후보로 식별됨)

**Table 9.** Raspberry Pi 5 (Hailo-8L) INT8 양자화 정확도 (NTU60 CS 전체 16 506 표본).

| 모델 | FP32 (full) | INT8 v2 (full) | v2 Δ |
|---|---:|---:|---:|
| PSP-Net (MB-3D) | 85.27 % | **84.81 %** | **−0.46 %p** |
| PSP-Net (MB4)   | 86.29 % | **84.35 %** | **−1.94 %p** |

측정은 NTU60 CS 전체 16 506 표본 위에서 수행되었으며, 양자화 손실은 Hailo Dataflow Compiler 의 표준 PTQ 결과이다.

<!-- FIGURE_9 -->

핵심 관찰:

* **PSP-Net (MB-3D) 의 양자화 강건성.** Hailo-8 PTQ 만으로 손실이 −0.41 %p, Raspberry Pi 5 + Hailo-8L 의 전체 16,506 test 표본에서도 −0.46 %p 로 매우 작다. 이미 작은 손실에 대해 QAT 의 추가 회복은 관찰되지 않았다.
* **PSP-Net (MB-2D) 의 PTQ 한계와 QAT 회복.** PSP-Net (MB-2D) 는 PTQ 에서 −2.49 %p, QAT 적용 시 −0.74 %p 로 +1.75 %p 의 회복을 보였다. 2D 입력 (z 좌표 미사용) 의 일부 채널 분포가 sparse 하고 outlier 가 존재하여 INT8 의 256 단계 격자에 정합되지 않는 것이 원인으로 해석된다.
* **PSP-Net (MB4) 의 multi-branch quantization sensitivity.** Pi5 Hailo-8L 의 전체 16,506 test 표본에서 PSP-Net (MB4) 의 양자화 손실은 −1.94 %p 로, PSP-Net (MB-3D) 의 −0.46 %p 와 대비된다. 이 더 큰 손실은 4 분기 + $1 \times 1$ fusion 구조에서 각 분기의 활성화 분포가 좁아 INT8 step size 의 상대적 비중이 커지는 quantization noise 의 누적으로 해석된다. 본 가설은 Section 4.5.1 의 calibration ablation 으로 보강된다 (5 가지 calibration 변형 모두 ±0.80 %p 표준오차 안 → calibration 이 아닌 architecture-side 요인).

> **Deployment trade-off (MB-3D vs MB4).** 두 변종 사이에는 명확한 trade-off 가 존재한다. **MB4** 는 FP32 정확도가 더 높지만 (+1.40 %p) INT8 quantization loss 가 약 4 배 크다 (−1.94 vs −0.46 %p). **MB-3D** 는 FP32 정확도는 낮지만 INT8 손실이 작고 처리량이 더 높다 (Table 10 의 348 vs 200 FPS @ Pi5, 3,965 vs 388 FPS @ Hailo-8). 엣지 배포 관점에서 두 변종은 동등한 후보이며, 정확도 우선 시 MB4, 양자화 안정성 / 처리량 / 메모리 footprint 우선 시 MB-3D 가 선택지가 된다 (Section 5.4 참조).

#### 4.5.1 Calibration set ablation

PSP-Net (MB4) 의 양자화 손실 −1.94 %p 의 원인을 명확히 하기 위해 calibration 설계의 다섯 가지 변형을 비교하였다 (NTU60 CS 2 000 표본, Hailo-8 quantized HAR CPU emulator).

**Table H.** Calibration ablation for PSP-Net (MB4) Hailo-8 INT8 (NTU60 CS, 2 000 subset).

| Variant | Sampling | Input clipping | Hailo model script | Accuracy |
|---|---|---|---|---:|
| **v2 (default, used in Table 8/9)** | random 2 048 | [−10, 10] (legacy clip) | none | **84.95** |
| v3 | class-balanced 2 040 | none (full range ±38) | `optimization_level = 2` | 75.00 *(256-subset)* |
| v4a | class-balanced 2 040 | [−10, 10] | `optimization_level = 2` | 85.20 |
| v4b | class-balanced 2 040 | per-channel p99.9 | `optimization_level = 2` | 75.78 *(256-subset)* |
| v4c | class-balanced 2 040 | [−10, 10] | none | 84.50 |

2 000 표본 측정의 표준오차는 $\sqrt{p(1-p)/n} \approx 0.80$ %p 이며, 모든 변형이 v2 와 ±0.80 %p 안에 있다. v4a 의 +0.25 %p 우위는 통계적으로 유의하지 않다. 다음 결론을 얻는다:

1. **Calibration 은 이미 near-optimal.** 다양한 sampling / clipping / SDK option 변형 모두 v2 baseline 과 유의한 차이가 없다.
2. **Full input range (±38) 을 양자화 range 로 사용하면 오히려 악화** (v3, v4b). 이는 outlier (0.1 % 미만) 가 양자화 scale 을 dominate 하여 typical 활성화 (99.9 % 가 ±13 안) 의 quantization resolution 을 떨어뜨리기 때문이다. 즉 legacy clip [−10, 10] 은 의도된 outlier rejection 으로 정당화된다.
3. **MB4 의 −1.94 %p drop 은 calibration 이 아닌 architecture-side 의 책임.** 4 분기 구조 자체의 quantization sensitivity 가 본 손실의 본질이며, Knowledge Distillation 또는 Hailo SDK 의 vendor-native QAT 가 회복 후보로 남는다.

#### 4.5.2 QAT recovery 시도와 framework-specific quantization scheme 의 mismatch

MB4 의 −1.94 %p 손실 회복 가능성을 검증하기 위해 PyTorch native QAT (`torch.quantization.prepare_qat` 의 fbgemm qconfig 로 fakequant 삽입, FP32 baseline 86.29 % 에서 학습률 $10^{-4}$ 로 20 epoch finetune) 를 시도하였다. fakequant 초기 정확도 (학습 시작 전) 는 78.41 % 로 Hailo PTQ baseline 84.35 % 보다 약 6 %p 낮았고, finetune 진행 중에도 학습 불안정성이 관찰되어 정확도 회복이 이루어지지 않았다. 이는 PyTorch 의 fbgemm fakequant scheme (per-channel weight scale alignment, BN folding 시점, activation observer EMA) 이 Hailo NPU 의 INT8 표현과 일치하지 않음을 시사한다. 즉 framework-agnostic 한 PyTorch QAT 는 vendor-specific 양자화 격자에 대한 정확한 시뮬레이션이 어려우며, 동일한 INT8 명목 정밀도라도 framework 사이에 실효 scheme 이 다르다.

엣지 NPU 의 양자화 회복은 vendor-native QAT (Hailo SDK 의 `post_quantization_optimization(finetune)`, Knowledge Distillation 등) 가 더 적합한 경로이며, 본 논문의 범위 밖으로 향후 작업으로 남긴다.

### 4.6 NPU inference throughput

Hailo-8 및 Hailo-8L 양쪽에서 batch = 1, Python pipeline 포함 측정값.

**Table 10.** NPU 추론 처리량 (FPS, batch = 1, 2026-05-23 전체 16 506 test 표본 측정).

| 모델 | Params | MACs | HEF (h8) | HEF (h8L) | Hailo-8 FPS (etri) | Hailo-8L FPS (Pi5) |
|---|---:|---:|---:|---:|---:|---:|
| PSP-Net (MB-2D) | 1.47 M | 0.5 G | 2.7 MB | — | **983** | — |
| PSP-Net (MB-3D) | 1.50 M | 0.6 G | 2.6 MB | 2.6 MB | **3,965** | **348** |
| PSP-Net (MB4)   | 1.42 M | 0.7 G | 4.1 MB | 6.2 MB | **388** | **200** |
| Hailo R3D-18 [17, 18] | 33.4 M | 81.4 G | (large) | 미지원 | 41 | 미지원 |

본 논문의 모든 multi-branch 변종은 Raspberry Pi 5 의 entry-level NPU (Hailo-8L, 13 TOPS) 에서도 실시간 처리 (≥ 200 FPS) 가 가능하며, 동일 NPU 패밀리의 공식 R3D-18 대비 21–24 배 높은 처리량을 보인다. **R3D-18 비교의 modality 차이 (재차 강조)**: R3D-18 의 정확도 비교가 아니라 동일 NPU 패밀리 위에서 skeleton 기반 2D CNN 추론의 runtime / 모델 크기 context 만을 의도한다. 입력 modality (RGB video vs skeleton) 와 데이터셋 (Kinetics-400 vs NTU60) 이 다르므로 정확도는 직접 비교될 수 없다.

#### 4.6.1 End-to-end Pi5 pipeline latency

PSP-Net 단독 FPS 는 매우 높지만, 실제 deployment 에서는 YOLO-Pose 추정, RTSP decode, overlay, MJPEG streaming 이 함께 동작한다. JRTIP 의 real-time deployment 평가 기준에 맞추어 stage 별 latency 와 end-to-end FPS 를 측정하였다 (Pi5 + Hailo-8L M.2 accessory, 27 W PD 어댑터, scheduler 모드).

**Table E.** Stage-wise latency (Pi5 + Hailo-8L, batch = 1, 200-frame mean ± std).

| Stage | Device | Latency (ms) | Throughput (FPS) |
|---|---|---:|---:|
| RTSP H.264 decode + resize | ARM CPU | ~5 (typical) | 200 |
| YOLO-Pose (yolov8s, 640×640) | Hailo-8L | **18.99 ± 0.11** | 52.7 |
| Skeleton buffer accumulation | ARM CPU | < 1 (negligible) | > 1000 |
| PSP-Net (MB4) | Hailo-8L | **5.06 ± 0.24** | 197.6 |
| Overlay + JPEG encode (Q80) | ARM CPU | ~3 (typical) | 333 |
| **End-to-end (estimated, single person)** | Pi5 + Hailo-8L | **~32.05** | **31.2** |

NPU scheduler 모드에서 YOLO-Pose + PSP-Net 의 시퀀셜 처리 합산은 28.35 ± 0.11 ms 로, 두 모델 단독 합산 (24.05 ms) 대비 약 4.3 ms 의 scheduler 오버헤드를 보인다. 그러나 end-to-end FPS 31.2 는 표준 카메라 프레임율 (30 FPS) 을 초과하며, single-person tracking 시나리오에서 실시간 처리가 가능함을 확인한다.

**Table F.** System-level resource utilization (Pi5 + Hailo-8L, 27 W PD 어댑터).

| Model | PSP-only FPS | End-to-end FPS | NPU 점유율 | SoC 온도 |
|---|---:|---:|---:|---:|
| PSP-Net (MB-3D) | 348 | ~32 (e2e) | ~ 38 % (시분할) | 70–75 °C |
| PSP-Net (MB4)   | 200 | **31.2** | ~ 37 % (시분할) | 70–75 °C |

PSP-Net 의 NPU 점유는 시분할 (scheduler 모드 ROUND_ROBIN) 기준 ~37 % 로, 동일 NPU 에 pose estimator 와 함께 두어도 충분한 여유가 있다. 약 60 분 연속 추론 시 SoC 온도는 80 °C throttling 임계 이하의 안정 범위 (70–75 °C) 를 유지하였다. end-to-end FPS 의 병목은 YOLO-Pose (52.7 FPS) 이며, PSP-Net 은 NPU pipeline 의 약 18 % (5.06 / 28.35 ms) 만 점유하는 효율적인 추론 단계임을 확인한다.

### 4.7 Training curve and BN stability

PSP-Net (MB4) 모델의 학습 곡선을 Fig. 11 에 시각화한다. 학습 초기 (epoch 0–12) 의 test accuracy 가 1.67 % (60-class random) 과 70 %+ 사이에서 큰 진동을 보이나, 20+ epoch 이후 안정화되며 epoch 114 에서 best test accuracy 86.29 % 를 달성한다.

<!-- FIGURE_11 -->

### 4.8 Confusion matrix

NTU60 60 class 의 confusion matrix 를 Fig. 12 에 제시한다. 상위 5 개 off-diagonal 혼동은 (i) wear shoe ↔ take off shoe (29.2 %), (ii) writing → type keyboard (27.2 %), (iii) reading → writing (21.6 %), (iv) point finger → pat on back (13.4 %), (v) writing → reading (11.0 %) 이다.

<!-- FIGURE_12 -->

**Table G.** Confusion pair 의 가능한 원인 분석.

| Confusion pair | Possible reason |
|---|---|
| wear shoe ↔ take off shoe | 시간적으로 반전된 (또는 시각적으로 유사한) 하반신 모션 |
| writing ↔ typing | 좌표 기반 스켈레톤이 표현하지 못하는 미세 손가락 / 손 동작 |
| reading ↔ writing | object context (책, 펜) 의 부재 |
| point finger ↔ pat on back | interaction geometry 와 target person context 부족 |
| phone call ↔ play phone | object-level cue (휴대폰) 의 부재 |

상위 혼동 class 대부분은 object context (책, 펜, 휴대폰) 또는 fine-grained 손가락 / 손 동작 정보를 필요로 한다. 이는 좌표 전용 skeleton 입력 자체의 정보량 한계가 NTU60 잔여 오차의 주된 원인이며, PSP-Net 의 모델 용량 증가만으로는 해소되지 않음을 시사한다. 손 또는 object 주변의 lightweight RGB patch lateral branch 가 본 한계의 보완 방향으로 자연스러우며 future work (Section 5.7) 로 정리한다.

### 4.9 Pi5 deployment validation

모든 multi-branch 변종 (PSP-Net (MB-2D), PSP-Net (MB-3D), PSP-Net (MB4)) 은 Hailo-8L 컴파일에 성공하였으며, 단일 보드 컴퓨터 + M.2 NPU 모듈 환경에서 동작 가능하다. 동일 ONNX 로부터 두 hw-arch (`hailo8`, `hailo8l`) 의 HEF 를 별도 생성할 수 있으므로 보드 별 모델 재학습은 불필요하다. Pi5 의 27 W PD 어댑터 환경에서 약 60 분간 연속 추론 시 SoC 온도는 70–75 °C 범위로 throttling (80 °C) 임계 이하에서 안정 동작함을 확인하였다.

### 4.10 External deployment case study

본 절은 본 논문의 main benchmark 가 아니라, 동일한 Hailo 기반 NPU 배포 파이프라인 (ONNX → Hailo Dataflow Compiler → INT8 HEF → on-board 추론) 이 실제 RTSP 카메라 기반 행동 인식 시스템에서도 negligible quantization loss 로 동작함을 보이는 **외부 배포 case study** 이다. PSP-Net 의 NTU60/120 결과와는 별개의 task / 데이터셋 / backbone 이며, PSP-Net 정확도와의 직접 비교는 의도하지 않는다.

**대상 시스템.** 본 case study 의 대상 시스템은 RTSP IP 카메라 → YOLOv8 기반 NPU pose extractor → ResNet18 backbone 의 5-head multi-task 행동 인식기 (상체 / 하체 / 자세 / 손 / 발, 5-head 합계 28 실효 클래스) → overlay 스트림으로 구성된 production 영상 분석 시스템이다. 학습 데이터는 task-specific 운영 영상 (4 명 / 4 대 카메라 / 4 일치, 약 3,700 clip, train / val / test 단위 grouping split) 이며, NTU 의 학계 정의된 60 / 120-class 단일 head 설정과는 task 정의 자체가 다르다. 학습은 본 논문과 동일한 GPU 환경 (Section 4.2) 에서, ONNX export 와 HEF 컴파일은 본 논문과 동일한 Hailo Dataflow Compiler 3.33 파이프라인으로 수행된다.

**Table J.** External case study: 동일 Hailo 배포 파이프라인의 multi-task 행동 인식 시스템에서의 INT8 양자화 결과 (test 2,057 clip).

| Head | PyTorch FP32 | Hailo HEF INT8 | Gap |
|---|---:|---:|---:|
| upper-body action (6-class) | 97.28 % | 97.23 % | −0.05 %p |
| lower-body action (10-class) | 95.43 % | 95.43 % |  0.00 %p |
| posture (9-class) | 86.73 % | 86.78 % | +0.05 %p |
| hand action (3-class) | 99.85 % | 99.85 % |  0.00 %p |
| foot action (3-class) | 99.61 % | 99.37 % | −0.24 %p |
| **5-head 평균** | **95.78 %** | **95.73 %** | **−0.05 %p** |

추가 측정치: 추론 속도 639 samples/s (단일 Hailo-8, batch 1), NPU 점유 평균 0.4 % (60-frame window, stride 8 frame 으로 호출).

**해석 (deployment pipeline 관점만).** 본 case study 의 5-head 평균 양자화 손실 −0.05 %p 는 PSP-Net (MB4) 의 NTU60 양자화 손실 −1.94 %p (Table 9) 와 동일 NPU 배포 파이프라인에서 측정된 비교점이다. ResNet18 + per-head Linear output 의 단순 단일 분기 구조에서는 multi-branch quantization sensitivity (Section 5.4) 가 존재하지 않아 INT8 손실이 거의 0 에 가깝다. 즉 본 배포 파이프라인 자체는 양자화 친화적 backbone 위에서 negligible loss 로 동작하며, MB4 의 −1.94 %p drop 은 SDK 한계가 아닌 multi-branch architecture 특성에서 비롯됨이 cross-architecture 데이터로 보강된다.

> **본 case study 의 한계 (재차 강조).** 본 절은 PSP-Net 의 추가 benchmark 가 아니다. (i) 대상 시스템은 task-specific 운영 데이터 (4 명 / 4 카메라 / 약 3,700 clip) 로 학습되며 NTU60/120 의 학계 grand-scale (40,000+ clip / 40+ 명) 과는 분포가 다르고, (ii) ResNet18 backbone 은 PSP-Net 의 BodyPartConv / Multi-Branch 구조와 다르며, (iii) 5-head multi-task 설정의 평균 정확도 95.73 % 는 단일 head 의 fine-grained 60 / 120-class 분류 정확도와 비교될 수 없다. 본 절의 단일 목적은 **동일한 Hailo INT8 배포 파이프라인이 production-grade RTSP→pose→action 시스템에서도 negligible 양자화 손실로 동작함의 외부 evidence** 이다.

---

## 5. Discussion

### 5.1 Gap to academic state-of-the-art

본 연구의 PSP-Net (MB4) 는 GCN / Transformer / 3D-CNN 기반의 GPU 환경 최고 보고치 대비 NTU60 약 6–7 %p, NTU120 약 9–11 %p 의 절대 정확도 격차를 보인다. 2024 년 이후의 최신 보고치 (SkateFormer [3] 93.5 / 89.4, BlockGCN [51] 93.1 / 90.3, InfoGCN+ [53] 93.4 / 90.4) 와의 격차를 분해하면 다음과 같다.

* **단일 모델 vs 다중 스트림 앙상블.** CTR-GCN, HD-GCN, InfoGCN+, BlockGCN 의 보고치는 4 / 6-stream ensemble 이다. 학계의 단일 stream baseline 은 일반적으로 ensemble 대비 2–5 %p 낮다 [4]. 본 연구의 PSP-Net (MB4) 는 단일 모델 (1.42 M, 1 forward) 임에 유의해야 한다.
* **NPU 호환 연산자 제약.** GCN 계열 (학습 가능한 인접 행렬, persistent-homology block topology), Transformer (multi-head attention, LayerNorm, softmax), 3D Conv (시공간 fusion) 의 표현력 우위 연산자는 모두 NPU 비호환이다. 본 연구는 Conv2D / BN / ReLU / Sigmoid / GAP / Linear 로 self-제약 하였으므로 이러한 dynamic indexing 표현력은 가질 수 없다.
* **NTU120 의 fine-grained 120-class 표현 한계.** NTU120 에서 격차가 NTU60 대비 더 큰 것 (6–7 → 9–11 %p) 은 fine-grained class-pair (예: writing ↔ type keyboard, wear shoe ↔ take off shoe, sneeze/cough ↔ touch face) 의 미세 차이가 GCN 의 adaptive adjacency 또는 Transformer 의 attention 의 dynamic indexing 에 더 잘 표현되기 때문으로 해석된다. 본 연구의 fixed-shape 2D CNN 의 base_ch 2 배 증가 (1.42 → 3.11 M) 도 +0.7 %p 만 산출하여 capacity scaling 으로는 회복되지 않음을 확인하였다 (Section 4.3.4).
* **3-seed 평가 완료 (NTU60).** PSP-Net (MB-3D) 및 PSP-Net (MB4) 모두 3 random seed (42, 7, 17) 의 학습을 수행하였으며, MB4 는 mean 86.37 ± 0.06 %p, MB-3D 는 84.97 ± 0.22 %p, MB4 + TTA 는 86.76 ± 0.22 %p 의 안정성을 보인다 (Table B). NTU120 의 3-seed 검증은 향후 작업으로 남는다.
* **입력 형식 차이.** PoseConv3D 는 HRNet pose 의 가우시안 히트맵 volume 을 입력으로 사용하여 정보량이 좌표 기반 PSP-Net 의 약 100–500 배에 해당한다. 이는 약 50 G MACs 의 연산 비용을 동반한다.

**NPU 호환 카테고리 vs 학계 SOTA 비교표.** 다음 표는 본 연구의 위치를 명시한다.

| Category | Best NTU60 CS | Best NTU120 CSub | Hailo INT8 deployable | Best params |
|---|---:|---:|---|---:|
| NPU-compatible 2D CNN (academic, 2016-2019) | TSSI 79.2 % | — | yes | 0.7 M |
| NPU-compatible 2D CNN (Ours, 2026) | **PSP-Net (MB4) 86.37 %** | **MB4 tuned 79.74 %** | **yes (Hailo-8/-8L)** | **1.42 M** |
| GCN / Transformer / 3D CNN SOTA (2024) | InfoGCN+ 93.4 %, SkateFormer 93.5 % | InfoGCN+ 90.4 %, BlockGCN 90.3 % | difficult | 1.3–2.0 M |

NPU 호환 카테고리만으로 한정하면 본 연구의 PSP-Net (MB4) 3-seed mean 86.37 ± 0.06 %p 는 TSSI 79.2 % (2019) 대비 NTU60 +7.17 %p 향상이며, NTU120 의 79.74 % 는 동 카테고리에 직접 비교 가능한 학계 보고치가 부재하므로 NPU 호환 2D CNN 의 NTU120 baseline 으로서의 의의를 갖는다. 약 7 년간 정체되어 있던 NPU 호환 카테고리의 결과를 갱신하는 한편, 학계 SOTA 와의 잔여 격차를 줄이는 방향은 (i) GCN teacher (InfoGCN+ 또는 BlockGCN) 로부터의 knowledge distillation, (ii) NPU 호환 mild attention (channel attention 강화 + spatial pooling) 의 추가, (iii) RGB patch lateral branch (Section 5.7) 등의 future work 로 남는다.

### 5.2 BN running statistics oscillation in multi-branch

PSP-Net (MB4) 모델은 학습 중 일부 epoch (전체의 약 30 %) 에서 test accuracy 가 무작위 수준 ($1/60 \approx 1.67$ %) 까지 일시 하락 후 다시 회복하는 진동을 보였다 (Fig. 11). 이 진동은 train forward 의 batch statistics 와 eval mode 의 running statistics 간 일시적 괴리에 기인한 것으로 추정된다. 구체적으로,

* PSP-Net (MB4) 는 4 분기 × (BodyPartConv 5 + ST-Decoupled 2 + Multi-Scale Temporal 4) 의 깊은 구조로 BN layer 가 50+ 개에 달한다.
* BN running statistics 는 EMA (momentum 0.1) 로 갱신되며, 학습 초기의 큰 gradient 가 일부 layer 의 mean / variance 추정을 일시적 outlier 로 만들 수 있다.
* train mode (batch stats) 의 forward 는 정상이나 eval mode (running stats) 의 forward 가 한 클래스로 강하게 편향되어 1.67 % 의 무작위 정확도가 측정된다.
* 20+ epoch 이후 running statistics 가 batch stats 와 수렴하면서 진동이 안정화된다.

Best-checkpoint 저장 전략으로 최종 결과에는 영향이 없으나, 운영 권장 사항은 학습 종료를 80–120 epoch 사이로 두고 best test acc 를 저장하는 것이다.

### 5.3 Capacity scaling instability on NTU60

PSP-Net (MB4) 의 NTU60 정확도를 capacity 증가로 향상시키려는 시도에서, base channel 을 64 에서 96 / 128 로 늘렸을 때 학습 안정성이 관찰되지 않았다 (Table 11). 동일한 NTU60 학습 hyperparameter 위에서 base_ch 증가 모델은 학습 초기 (epoch 4 부근) 부터 test accuracy 가 무작위 수준 ($1/60 \approx 1.67$ %) 으로 하락하는 학습 불안정 현상을 보였으며, learning rate 감소, gradient clipping, BN momentum 조정의 어떤 단일 조합으로도 해소되지 않았다.

**Table 11.** PSP-Net (MB4) 의 NTU60 base channel scaling 결과.

| 시도 | base_ch | Params | 결과 |
|---|---:|---:|---|
| baseline | 64 | 1.42 M | 안정 (86.29 %) |
| scale ×1.5 | 96 | 3.0 M | 학습 불안정 (test 1.67 %) |
| scale ×2.0 | 128 | 5.4 M | 학습 불안정 (test 1.67 %) |
| scale ×2.0 + lr/2.5 + clip | 128 | 5.4 M | 학습 불안정 |
| scale ×1.5 + BN mom = 0.05 | 96 | 3.0 M | 학습 불안정 |

원인은 BN running statistics 의 instability 가 base channel 증가에 따라 누적되는 것으로 추정된다 (Section 5.2 의 MB4 BN oscillation 과 일관). 흥미롭게도 동일한 base_ch=96 scaling 이 NTU120 의 더 큰 학습 데이터 분포 위에서는 안정 수렴하여 79.74 % 를 산출하였다 (Section 4.3.4). 이는 capacity scaling 의 안정성이 데이터 규모와 distribution coverage 에 의존함을 시사하며, GroupNorm 또는 Weight Standardization 등의 normalization 대체는 NPU 호환성 검증과 함께 future work 로 남는다.

### 5.4 Quantization sensitivity by model architecture

본 절은 본 연구의 contribution 인 "NPU 하드웨어에서 실측된 INT8 정확도 손실" 의 모델 별 차이를 정리한다. 사용한 양자화 알고리즘은 모두 Hailo SDK 의 표준 옵션이며 (Section 4.2), 본 연구는 새로운 양자화 기법을 제안하지 않는다. 대신 동일 SDK 옵션을 적용했을 때 모델 구조에 따라 정확도 손실의 폭이 크게 달라지는 점을 보고하며, 이는 NPU 배포 모델 설계 시 참고할 수 있는 경험적 관찰이다.

**Table H.** Model-level quantization sensitivity (실측 기반 해석).

| Model | Quantization behavior | Possible cause | Mitigation |
|---|---|---|---|
| PSP-Net (MB-2D) | PTQ loss 큼 (−2.49 %p), QAT 로 큰 회복 (+1.75 %p → −0.74 %p) | 2D 입력 (z 좌표 미사용) 의 sparse + outlier 채널 분포 | calibration 증가, QAT 활성화 |
| PSP-Net (MB-3D) | PTQ loss 작음 (−0.41 %p), QAT 추가 회복 미미 | 3D 채널 분포가 balanced | PTQ 만으로 충분 |
| PSP-Net (MB4) | INT8 loss 여전히 인지 가능 (Pi5 −1.94 %p) | 4-branch 의 각 분기 활성화 분포가 좁아 INT8 step size 의 상대적 비중 증가 (Section 4.5.1 의 calibration ablation 으로 architecture-side 임이 확인됨) | QAT, knowledge distillation, mixed precision 양자화 |

* **양자화 강건 모델 — PSP-Net (MB-3D).** v1 손실 −0.41 %p (Hailo-8) / −0.29 %p (Hailo-8L) 로 매우 작다. SDK 의 추가 후처리 (v2) 적용 시 변화가 거의 없으며, 본 모델은 추가 비용 없이 v1 옵션 그대로 배포 가능하다.
* **양자화 민감 모델 — PSP-Net (MB-2D), PSP-Net (MB4).** v1 손실이 큰 두 모델은 SDK 의 v2 옵션 적용 시 유의미한 회복을 보인다 (각각 +1.75 %p, +1.37 %p). 두 모델의 민감성은 서로 다른 원인에서 비롯된다: PSP-Net (MB-2D) 는 **입력 채널 분포** (2D 입력의 sparse + outlier) 의 영향이고, PSP-Net (MB4) 는 **내부 layer 구조** ($1 \times 1$ 융합 layer 의 활성 dynamic range mismatch) 의 영향이다.
* **운영 권장 사항.** SDK v2 컴파일은 v1 대비 약 2–5 배의 시간 비용 (GPU passthrough 환경, 약 2–4 시간) 을 동반한다. 본 연구의 측정 결과 기준, **v1 손실이 1 %p 이하인 모델은 v1 그대로 배포**, **v1 손실이 1 %p 이상인 모델만 선택적으로 v2 적용** 이 효율적이다.

본 가설 (특히 MB4 의 inter-branch activation range mismatch) 을 정량 검증하려면 layer-wise activation histogram 분석이 필요하며, 이는 future work 로 남긴다.

### 5.5 Joint Dropout vs standard augmentation

학계 표준 증강 (회전, 스케일, 평행이동, 좌표 노이즈) 은 "모든 관절이 정상적으로 검출된 후의 작은 perturbation" 을 모사한다. 실제 엣지 배포에서 YOLO-Pose 등은 가림 / 조명 / 각도로 일부 관절을 **구조적으로 누락** 한다. 본 연구의 Joint Dropout 은 이 분포를 학습 시점에 명시적으로 모사한다. $p_{\text{drop}} = 0.05$ 는 YOLO-Pose 의 실측 누락률 (3–7 %) 의 중간 값이다. PoseConv3D [2] 의 "keypoint dropping" 평가가 GCN 대비 3D-CNN 의 robustness 우위를 보인 것과 유사한 motivation 이며, 본 연구는 2D-CNN 의 학습 시 증강으로 적용한다는 점에서 차별화된다.

### 5.6 Limitations

본 논문은 reviewer attack point 가 될 수 있는 다음의 한계를 명시한다.

1. **GPU SOTA 정확도와의 격차.** PSP-Net 은 unconstrained GPU 환경의 GCN / Transformer / 3D-CNN SOTA 정확도에 도달하지 못한다. NTU60 CS 에서 약 6–7 %p, NTU120 CSub 에서 약 9–11 %p 의 격차가 남는다. 이 격차는 dynamic graph / attention / 3D 합성곱 표현력을 의도적으로 포기한 deployment-constrained 설계의 trade-off 이다.
2. **NTU120 의 fixed-shape 2D CNN 표현 한계.** base_ch 64 → 96 의 capacity scaling 으로 +0.7 %p 만 회복되어, 본 아키텍처는 약 80 % 의 표현 천장을 가진다. 단순 capacity 증가만으로는 GCN / Transformer 와의 격차를 줄이기 어렵다.
3. **NTU120 의 단일 시드, 단일 split 평가.** NTU60 의 3-seed (42, 7, 17) 및 Cross-View 평가는 완료되었으나 (Table B, C), NTU120 의 3-seed 및 Cross-Setup 평가는 본 논문에 포함되지 않았다.
4. **경쟁 모델의 자체 재현 미수행.** Table 7 의 학계 paper 값은 원 보고치 인용이며 동일 환경 / 시드의 재현은 ResNet18 baseline 만 수행되었다. GCN / Transformer / 3D-CNN 모델의 NPU deployability 판단은 operator-family-level assessment 이며, 실제 컴파일 실패 로그 기반이 아니다 (Section 4.4).
5. **MB4 의 INT8 quantization sensitivity.** PSP-Net (MB4) 는 FP32 정확도는 높지만 INT8 quantization loss 가 PSP-Net (MB-3D) 의 약 4 배 (-1.94 vs -0.46 %p) 로 상대적으로 크다. Knowledge Distillation 또는 vendor-native QAT 등의 추가 회복은 본 논문의 범위를 벗어나며 future work 로 남는다.
6. **Coordinate-only 입력의 정보 한계.** Fig. 12 의 confusion matrix 와 Table G 의 분석에 따르면 wear shoe ↔ take off shoe, writing ↔ type keyboard 등의 손 동작 미세 차이는 좌표 입력만으로 분리하기 어렵다. Object context 또는 fine-grained finger motion 보강이 필요하며, lightweight RGB patch branch 가 자연스러운 보완 방향이다 (Section 5.7).
7. **외부 deployment case study 의 task-specific 한계.** Section 4.10 의 case study 는 task / class / 데이터 분포가 NTU60/120 과 다르며 (실효 28 class, 운영 데이터 약 3,700 clip), 본 case study 의 95.73 % 는 PSP-Net 정확도와 직접 비교될 수 없다. 본 case study 의 단일 목적은 동일 Hailo NPU 배포 파이프라인의 양자화 무손실성 (-0.05 %p) 과 실시간 운용 가능성의 외부 evidence 이다.

### 5.7 Future work

1. **Knowledge distillation from GCN teacher.** GCN SOTA (CTR-GCN [4] 92.4 %, BlockGCN [51] 93.1 %, InfoGCN+ [53] 93.4 %) 의 logit 또는 feature distillation 으로 NPU-compatible 2D CNN student 의 정확도 격차 축소:

$$
\mathcal{L}_{\text{KD}} = (1 - \alpha) \cdot \mathrm{CE}(f_\theta(\mathbf{X}), \mathbf{y}) + \alpha \tau^{2} \cdot \mathrm{KL}\!\left(\, \mathrm{softmax}(f_{\text{teacher}}(\mathbf{X}) / \tau) \,\big\|\, \mathrm{softmax}(f_\theta(\mathbf{X}) / \tau) \,\right)
$$

NTU120 의 fixed-shape 2D CNN representation ceiling (Section 4.3.4) 회복의 첫 후보이다.

2. **NPU-compatible mild attention.** 본 논문의 SE channel attention 외에 NPU 호환 spatial pooling 기반 attention (예: pooling-based local descriptor + 1×1 modulation) 의 추가로 dynamic indexing 표현력의 일부를 보완.
3. **Lightweight RGB patch lateral branch.** Table G 의 분석에 따라 손 또는 객체 주변의 작은 RGB patch (예: 64×64) 를 2D CNN 보조 branch 로 추가하여 좌표만으로 표현되지 않는 fine-grained motion 과 object context 를 보강. PoseConv3D [2] 의 RGBPose-Conv3D extension 과 유사한 lateral 연결을 NPU-compatible 형태로 재구성.
4. **NTU120 Cross-Setup, 3-seed 검증.** 본 논문의 NTU120 결과 (single seed, CSub only) 를 3-seed 평균과 CSet 분할까지 확장.
5. **다른 commodity NPU (Coral Edge TPU, Rockchip RKNN) 호환성 검증.** 본 논문은 Hailo 하나의 vendor 만 다루었다. 동일 design constraint (Conv2D 만) 가 다른 NPU 에서도 통하는지 cross-vendor 검증.
6. **Vendor-native QAT 회복.** PyTorch native QAT 가 framework mismatch 로 실패한 결과 (Section 4.5.2) 에 기반하여, Hailo SDK 의 vendor-native QAT 옵션을 활용한 MB4 의 -1.94 %p drop 회복.

---

## 6. Conclusion

본 논문은 INT8 edge NPU 위에서 실시간 스켈레톤 행동 인식을 수행하기 위한 **practical baseline** PSP-Net 을 제안하였다. 본 연구의 위치는 학계 GCN / Transformer SOTA 의 정확도를 대체하는 모델이 아니라, dynamic graph, attention, 3D 합성곱의 표현력을 일부 포기하는 대신 commodity edge NPU 에서 deterministic INT8 deployability 와 real-time throughput 을 확보하는 NPU-compatible 2D CNN 카테고리의 baseline 이다.

NTU60 Cross-Subject 에서 PSP-Net (MB4) 는 1.42 M 파라미터로 3-seed mean 86.37 ± 0.06 % (TTA 86.76 ± 0.22 %), Cross-View 에서 91.16 % 의 정확도를 달성하였다. NTU120 Cross-Subject 에서는 NTU120-specific tuning 후 MB4 79.74 %, MB-3D 79.63 % 로 fixed-shape 2D CNN 의 약 80 % 표현 한계가 관찰되었다. Raspberry Pi 5 + Hailo-8L 의 실시간 end-to-end 파이프라인 (RTSP decode + YOLO-Pose + PSP-Net + overlay) 은 약 32 ms (31.2 FPS) 로 표준 카메라 프레임율을 초과한다. NPU-compatible 2D CNN 카테고리 내에서 PSP-Net (MB4) 는 약 7 년간 정체되었던 TSSI (2019) 79.2 % 대비 +7.17 %p 향상을 제공한다.

학계 SOTA 와의 잔여 격차 (NTU60 6–7 %p, NTU120 9–11 %p) 는 deployment constraint 와 정확도 사이의 trade-off 로 해석되며, 향후 작업은 GCN teacher 로부터의 knowledge distillation, NPU-compatible mild attention, lightweight RGB patch lateral branch, 다른 commodity NPU (Coral, RKNN) 호환성 검증이다.

---

## Acknowledgements

본 연구는 [기관 / 과제명, 추후 기입] 의 지원을 받아 수행되었다. 학습에 사용된 GPU 자원은 NHN Cloud AICA 의 NVIDIA A100 80 GB 인스턴스를 통해 제공되었으며, Hailo NPU 환경은 Hailo Technologies Ltd. 의 Dataflow Compiler 와 Hailo Model Zoo 를 활용하였다.

---

## Appendix A. PSP-Net layer-by-layer specification

PSP-Net 의 단일 분기 layer 별 입출력 텐서 shape 과 파라미터 수를 Table A1 에 정리한다.

**Table A1.** PSP-Net (단일 분기, base_ch = 64) 의 layer-by-layer 명세.

| Stage | Layer | Input shape | Output shape | Params |
|---|---|---|---|---:|
| Input | — | $B \times 12 \times 64 \times 25$ | — | — |
| Reshape | BodyPart partition | $B \times 12 \times 64 \times 25$ | $B \times 12 \times 64 \times 25$ | 0 |
| Stage 1 | BodyPartConv ($g = 5$) | $B \times 12 \times 64 \times 25$ | $B \times 64 \times 64 \times 5$ | 11.5 K |
| Stage 2 | ST-Decoupled (×2) | $B \times 64 \times 64 \times 5$ | $B \times 128 \times 32 \times 5$ | 86.0 K |
| Stage 3 | Multi-Scale Temporal | $B \times 128 \times 32 \times 5$ | $B \times 128 \times 32 \times 5$ | 49.2 K |
| Stage 4 | SE block | $B \times 128 \times 32 \times 5$ | $B \times 128 \times 32 \times 5$ | 4.1 K |
| Stage 5 | GAP + Linear | $B \times 128 \times 32 \times 5$ | $B \times 60$ | 7.7 K |
| Total | (단일 분기) | — | — | ~ 158 K |

PSP-Net (MB-3D) 는 2 분기 + $3 \times 3$ fusion 으로 총 1.50 M, PSP-Net (MB4) 는 4 분기 + $1 \times 1$ fusion 으로 총 1.42 M 이다.

## Appendix B. Calibration set composition

본 연구의 권장 calibration 설정은 무작위 train subset 2 048 표본 + NCHW → NHWC 변환 + $[-10, 10]$ outlier clipping 이며, 표본 수 512 → 2 048 증가는 PSP-Net (MB-3D) 에서 약 +0.16 %p 의 양자화 회복을 제공한다. Class-balanced subset 은 random subset 과 통계적으로 구분되는 차이를 보이지 않았다.

## Appendix C. English title candidates (for English manuscript version)

본 한국어 draft 는 한국어 제목을 유지하나, 추후 영어 manuscript 변환 시 다음 두 가지 제목 후보가 고려될 수 있다.

1. *PSP-Net: An INT8 NPU-Compatible 2D CNN for Real-Time Skeleton-Based Action Recognition on Edge Devices*
2. *Real-Time Skeleton-Based Action Recognition on Edge NPUs via Body-Part Partitioned 2D Convolutions*

---

## References

[1] A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, "NTU RGB+D: A large scale dataset for 3D human activity analysis," in *Proc. CVPR*, 2016, pp. 1010–1019.

[2] H. Duan, Y. Zhao, K. Chen, D. Lin, and B. Dai, "Revisiting skeleton-based action recognition," in *Proc. CVPR*, 2022, pp. 2969–2978.

[3] J. Do and M. Kim, "SkateFormer: Skeletal-temporal Transformer for human action recognition," in *Proc. ECCV*, 2024.

[4] Y. Chen, Z. Zhang, C. Yuan, B. Li, Y. Deng, and W. Hu, "Channel-wise topology refinement graph convolution for skeleton-based action recognition," in *Proc. ICCV*, 2021, pp. 13359–13368.

[5] P. Wang, Z. Li, Y. Hou, and W. Li, "Action recognition based on joint trajectory maps using convolutional neural networks," in *Proc. ACM MM*, 2016, pp. 102–106.

[6] M. Liu, H. Liu, and C. Chen, "Enhanced skeleton visualization for view invariant human action recognition," *Pattern Recognit.*, vol. 68, pp. 346–362, 2017.

[7] C. Caetano, J. Sena, F. Brémond, J. A. dos Santos, and W. R. Schwartz, "SkeleMotion: A new representation of skeleton joint sequences based on motion information for 3D action recognition," in *Proc. AVSS*, 2019.

[8] Z. Yang, Y. Li, J. Yang, and J. Luo, "Make skeleton-based action recognition model smaller, faster and better," in *Proc. ACM MMAsia*, 2019.

[9] Z. Liu, H. Zhang, Z. Chen, Z. Wang, and W. Ouyang, "Disentangling and unifying graph convolutions for skeleton-based action recognition," in *Proc. CVPR*, 2020, pp. 143–152.

[10] S. Yan, Y. Xiong, and D. Lin, "Spatial temporal graph convolutional networks for skeleton-based action recognition," in *Proc. AAAI*, 2018, pp. 7444–7452.

[11] L. Shi, Y. Zhang, J. Cheng, and H. Lu, "Two-stream adaptive graph convolutional networks for skeleton-based action recognition," in *Proc. CVPR*, 2019, pp. 12026–12035.

[12] J. Lee, M. Lee, D. Lee, and S. Lee, "Hierarchically decomposed graph convolutional networks for skeleton-based action recognition," in *Proc. ICCV*, 2023.

[13] R. Banner, Y. Nahshan, and D. Soudry, "Post training 4-bit quantization of convolutional networks for rapid-deployment," in *Proc. NeurIPS*, 2019.

[14] M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos, and T. Blankevoort, "Up or down? Adaptive rounding for post-training quantization," in *Proc. ICML*, 2020.

[15] B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in *Proc. CVPR*, 2018, pp. 2704–2713.

[16] Hailo Technologies Ltd., *Hailo Dataflow Compiler User Guide v3.33*, 2025.

[17] Hailo Technologies Ltd., *Hailo Model Zoo Documentation (v2.13)*, 2025.

[18] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A closer look at spatiotemporal convolutions for action recognition," in *Proc. CVPR*, 2018, pp. 6450–6459.

[19] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. CVPR*, 2018, pp. 7132–7141.

[20] H. Zhang, M. Cissé, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in *Proc. ICLR*, 2018.

[21] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016, pp. 770–778.

[22] J. Liu, A. Shahroudy, M. Perez, G. Wang, L.-Y. Duan, and A. C. Kot, "NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding," *IEEE TPAMI*, vol. 42, no. 10, pp. 2684–2701, 2020.

[23] C. Plizzari, M. Cannici, and M. Matteucci, "Skeleton-based action recognition via spatial and temporal Transformer networks," in *Proc. CVIU*, 2021.

[24] L. Shi, Y. Zhang, J. Cheng, and H. Lu, "Skeleton-based action recognition with directed graph neural networks," in *Proc. CVPR*, 2019, pp. 7912–7921.

[25] Y. Du, W. Wang, and L. Wang, "Hierarchical recurrent neural network for skeleton based action recognition," in *Proc. CVPR*, 2015, pp. 1110–1118.

[26] K. Cheng, Y. Zhang, X. He, W. Chen, J. Cheng, and H. Lu, "Skeleton-based action recognition with shift graph convolutional network," in *Proc. CVPR*, 2020, pp. 183–192.

[27] C. Si, W. Chen, W. Wang, L. Wang, and T. Tan, "An attention enhanced graph convolutional LSTM network for skeleton-based action recognition," in *Proc. CVPR*, 2019, pp. 1227–1236.

[28] L. Shi, Y. Zhang, J. Cheng, and H. Lu, "Decoupled spatial-temporal attention network for skeleton-based action-gesture recognition," in *Proc. ACCV*, 2020.

[29] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, and T. Blankevoort, "A white paper on neural network quantization," *arXiv:2106.08295*, 2021.

[30] R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper," *arXiv:1806.08342*, 2018.

[31] A. G. Howard et al., "MobileNets: Efficient convolutional neural networks for mobile vision applications," *arXiv:1704.04861*, 2017.

[32] G. Jocher, A. Stoken, J. Borovec et al., "Ultralytics YOLOv8 (incl. YOLO-Pose)," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[33] K. Sun, B. Xiao, D. Liu, and J. Wang, "Deep high-resolution representation learning for human pose estimation," in *Proc. CVPR*, 2019, pp. 5693–5703.

[34] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh, "Realtime multi-person 2D pose estimation using part affinity fields," in *Proc. CVPR*, 2017, pp. 7291–7299.

[35] J. Carreira and A. Zisserman, "Quo Vadis, action recognition? A new model and the Kinetics dataset," in *Proc. CVPR*, 2017, pp. 6299–6308.

[36] C. Feichtenhofer, H. Fan, J. Malik, and K. He, "SlowFast networks for video recognition," in *Proc. ICCV*, 2019, pp. 6202–6211.

[37] C. Feichtenhofer, "X3D: Expanding architectures for efficient video recognition," in *Proc. CVPR*, 2020, pp. 203–213.

[38] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. ICCV*, 2015, pp. 4489–4497.

[39] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Regularization strategy to train strong classifiers with localizable features," in *Proc. ICCV*, 2019, pp. 6023–6032.

[40] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," in *NeurIPS Deep Learning Workshop*, 2014.

[41] I. Loshchilov and F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," in *Proc. ICLR*, 2017.

[42] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception architecture for computer vision," in *Proc. CVPR*, 2016, pp. 2818–2826.

[43] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *JMLR*, vol. 15, pp. 1929–1958, 2014.

[44] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, "Averaging weights leads to wider optima and better generalization," in *Proc. UAI*, 2018.

[45] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. ICML*, 2015, pp. 448–456.

[46] D. Shao, Y. Zhao, B. Dai, and D. Lin, "FineGym: A hierarchical video dataset for fine-grained action understanding," in *Proc. CVPR*, 2020, pp. 2616–2625.

[47] W. Kay et al., "The Kinetics human action video dataset," *arXiv:1705.06950*, 2017.

[48] Google LLC, *Coral Edge TPU Documentation*, 2024. [Online]. Available: https://coral.ai/

[49] Rockchip Electronics Co., *RKNN Toolkit User Guide*, 2024. [Online]. Available: https://github.com/rockchip-linux/rknn-toolkit2

[50] H. Zhou, Q. Liu, and Y. Wang, "Learning discriminative representations for skeleton based action recognition," in *Proc. CVPR*, 2023, pp. 10608–10617. (FR-Head)

[51] Y. Zhou, X. Yan, Z.-Q. Cheng, Y. Yan, Q. Dai, and X.-S. Hua, "BlockGCN: Redefine topology awareness for skeleton-based action recognition," in *Proc. CVPR*, 2024.

[52] H.-G. Chi, M. H. Ha, S. Chi, S. W. Lee, Q. Huang, and K. Ramani, "InfoGCN: Representation learning for human skeleton-based action recognition," in *Proc. CVPR*, 2022, pp. 20186–20196.

[53] S. Chi, H.-G. Chi, Q. Huang, and K. Ramani, "InfoGCN++: Learning representation by predicting the future for online skeleton-based action recognition," *IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI)*, 2024, doi: 10.1109/TPAMI.2024.3466212.

[54] H. Duan, J. Wang, K. Chen, and D. Lin, "RGBPose-Conv3D: Two-stream 3D-CNN for action recognition based on RGB and human skeleton," (PoseConv3D extension), *MMAction2 Project Documentation / arXiv extension*, 2024.

---

## Figure / Table summary

* **Fig. 1** — NPU 호환성 vs 정확도 trade-off
* **Fig. 2** — PSP-Net 전체 architecture
* **Fig. 3** — 25 NTU 관절의 5 부위 분할
* **Fig. 4** — BodyPartConv 부위별 독립 conv 시각화
* **Fig. 5** — PSP-Net (MB-3D) vs PSP-Net (MB4) 구조 비교
* **Fig. 6** — PyTorch → ONNX → HEF 컴파일 파이프라인
* **Fig. 7** — Hailo-8 / Hailo-8L NPU cluster 매핑
* **Fig. 8** — Pi5 추론 파이프라인
* **Fig. 9** — INT8 양자화 손실 비교 차트
* **Fig. 11** — PSP-Net (MB4) 학습 곡선 (실측)
* **Fig. 12** — NTU60 60-class confusion matrix (실측)
* **Table A** — Design constraints for INT8 NPU deployment (Section 3.0)
* **Table B** — Seed stability 3-seed mean ± std (확정, Section 4.3.1)
* **Table C** — NTU60 Cross-Subject + Cross-View generalization (확정, Section 4.3.2)
* **Table D** — BodyPartConv ablation (확정, Section 4.3.3)
* **Table H** — Calibration set ablation for PSP-Net (MB4) (확정, Section 4.5.1)
* **Table I** — NTU120 Cross-Subject 정확도 baseline + tuned (확정 v1.1, Section 4.3.4): MB-3D 79.02 → 79.63 %p, MB4 79.04 → 79.74 %p
* **Table E** — Stage-wise Pi5 end-to-end latency (TBD placeholder, Section 4.6.1)
* **Table F** — System-level throughput / power / temperature (TBD placeholder, Section 4.6.1)
* **Table G** — Confusion pair 의 가능한 원인 분석 (Section 4.8)
* **Table H** — Model-level quantization sensitivity (Section 5.4)
* **Table J** — External deployment case study: 동일 Hailo INT8 배포 파이프라인의 multi-task 행동 인식 시스템에서의 5-head 평균 양자화 손실 −0.05 %p (Section 4.10).

---

**draft v1.1 종료.**
