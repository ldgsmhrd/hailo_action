# RESULTS — 측정 결과 요약

본 문서는 [PAPER_KO.md](../paper/PAPER_KO.md) 의 주요 결과를 빠르게 참조할 수 있도록 정리한 요약입니다.

## NTU60 Cross-Subject (학계 표준 평가)

### 3-seed mean ± std (paper Table B)

| Model | Params | Seed 42 | Seed 7 | Seed 17 | Mean ± Std |
|---|---:|---:|---:|---:|---:|
| PSP-Net (MB-3D) | 1.50 M | 85.27 | 84.79 | 84.85 | **84.97 ± 0.22** |
| PSP-Net (MB4)   | 1.42 M | 86.29 | 86.44 | 86.39 | **86.37 ± 0.06** |
| PSP-Net (MB4) + TTA | 1.42 M | 86.62 | 86.59 | 87.07 | **86.76 ± 0.22** |

> TTA = anatomical horizontal mirror (x 좌표 부호 반전 + L/R 관절 swap)

### Variants 비교 (paper Table 3, seed 42 best)

| Variant | Params | Aux ch. | NTU60 CS |
|---|---:|---|---:|
| PSP-Net lite1 (2D, M=1) | 1.07 M | 6 | 77.86 % |
| PSP-Net lite2 (2D, M=2) | 1.07 M | 12 | 80.09 % |
| PSP-Net full2 (3D, M=2) + aug | 1.13 M | 18 | 84.22 % |
| PSP-Net (MB-2D) + aug | 1.47 M | 16 | 83.13 % |
| PSP-Net (MB-3D) + aug | 1.50 M | 24 | 85.27 % |
| **PSP-Net (MB4) + aug** | **1.42 M** | **24** | **86.29 %** |

### BodyPartConv Ablation (paper Table D)

| Spatial encoding | NTU60 CS | Δ vs proposed |
|---|---:|---:|
| Plain joint order + Conv2D | 82.79 | −2.48 |
| Random joint order + Conv2D | 81.61 | −3.66 |
| Body-part partition + standard Conv2D | 79.69 | −5.58 |
| **Body-part partition + grouped BodyPartConv** | **85.27** | — |

Body-part partition 과 grouped Conv2D **두 설계 결정 모두 필수.** partition 만 적용 (grouped 없이) 시 plain 대비 −3.10 %p 악화 → grouped 가 회수해야 net gain.

## NTU60 Cross-View (paper Table C)

| Model | NTU60 CS | NTU60 CV | Δ (CV − CS) |
|---|---:|---:|---:|
| PSP-Net (MB-3D) | 85.27 | **90.42** | +5.15 |
| PSP-Net (MB4)   | 86.29 | **91.16** | +4.87 |

## NTU120 Cross-Subject (paper Table I)

| Model | Hyperparameter | Params | NTU120 CSub | Δ tuned |
|---|---|---:|---:|---:|
| PSP-Net (MB-3D) baseline | base_ch=64, ep=120 | 1.50 M | 79.02 | — |
| PSP-Net (MB-3D) tuned    | base_ch=96, ep=200 | 3.31 M | **79.63** | +0.61 %p |
| PSP-Net (MB4) baseline   | base_ch=64, ep=120 | 1.42 M | 79.04 | — |
| PSP-Net (MB4) tuned      | base_ch=96, ep=200 | 3.11 M | **79.74** | +0.70 %p |

> **NTU120 representation ceiling**: base_ch 1.5×, epoch 1.67× 늘려도 +0.7 %p 만 회복. PSP-Net family 의 fixed-shape 2D CNN 아키텍처는 약 80 % 의 표현 한계를 가짐 (paper Section 4.3.4).

## INT8 양자화 결과 (paper Table 8, 9)

### Hailo-8 (검증 보드, 전체 16,506 test)

| Model | FP32 | INT8 | Drop |
|---|---:|---:|---:|
| PSP-Net (MB-2D) | 83.13 | 82.39 (v2) | −0.74 %p |
| PSP-Net (MB-3D) | 85.27 | **84.86** (v1) | **−0.41 %p** |
| PSP-Net (MB4)   | 86.29 | 84.37 (v2) | −1.92 %p |
| PSP-Net (MB4) v4a | 86.29 | 84.68 (calib-tuned) | −1.61 %p |

### Hailo-8L / Raspberry Pi 5 (전체 16,506 test)

| Model | FP32 | INT8 | Drop |
|---|---:|---:|---:|
| PSP-Net (MB-3D) | 85.27 | **84.81** | **−0.46 %p** |
| PSP-Net (MB4)   | 86.29 | 84.35 | −1.94 %p |

> **MB-3D vs MB4 deployment trade-off**: MB4 는 FP32 정확도 +1.40 %p 우위지만 INT8 quantization loss 가 약 4 배 (−1.94 vs −0.46 %p). 양자화 안정성 / 처리량 우선 시 MB-3D, FP32 정확도 우선 시 MB4.

## NPU 추론 처리량 (paper Table 10)

| 모델 | Params | MACs | HEF size | Hailo-8 FPS | Hailo-8L FPS (Pi5) |
|---|---:|---:|---:|---:|---:|
| PSP-Net (MB-2D) | 1.47 M | 0.5 G | 2.7 MB | **983** | — |
| PSP-Net (MB-3D) | 1.50 M | 0.6 G | 2.6 MB | **3,965** | **348** |
| PSP-Net (MB4)   | 1.42 M | 0.7 G | 4.1 MB | **388** | **200** |
| (참고) Hailo R3D-18 | 33.4 M | 81.4 G | (large) | 41 | 미지원 |

> R3D-18 비교는 modality (RGB vs skeleton) / 데이터셋 (Kinetics-400 vs NTU60) 이 다르므로 정확도 직접 비교 X. 동일 NPU 패밀리의 runtime / 모델 크기 context 만 의도.

## Pi5 End-to-End Pipeline (paper Table E, F)

### Stage-wise latency (200 frame 평균 ± std)

| Stage | Device | Latency (ms) | Throughput (FPS) |
|---|---|---:|---:|
| RTSP H.264 decode + resize | ARM CPU | ~5 (typical) | 200 |
| YOLO-Pose (yolov8s, 640×640) | Hailo-8L | **18.99 ± 0.11** | 52.7 |
| Skeleton buffer | ARM CPU | < 1 | > 1000 |
| PSP-Net (MB4) | Hailo-8L | **5.06 ± 0.24** | 197.6 |
| Overlay + JPEG encode | ARM CPU | ~3 (typical) | 333 |
| **End-to-end** | Pi5 + Hailo-8L | **~32.05** | **31.2** |

### Resource utilization (1 시간 연속 추론)

| Model | PSP-only FPS | E2E FPS | NPU 점유 | SoC 온도 |
|---|---:|---:|---:|---:|
| PSP-Net (MB-3D) | 348 | ~32 | ~38 % (시분할) | 70–75 °C |
| PSP-Net (MB4)   | 200 | **31.2** | ~37 % (시분할) | 70–75 °C |

> 30 FPS 표준 카메라 프레임율 초과 → **JRTIP real-time 기준 충족.** YOLO-Pose 가 e2e bottleneck (~60 %), PSP-Net 은 NPU pipeline 의 ~18 % 만 점유.

## 학계 SOTA 와의 비교 (paper Table 7 발췌)

| Method | Year | Backbone | NTU60 CS | NTU120 CSub | NPU? |
|---|---|---|---:|---:|---|
| **NPU-compatible 2D CNN category** | | | | | |
| TSSI [8] | 2019 | 2D CNN | 79.2 % | — | ✅ |
| **PSP-Net (MB4)** | 2026 | 2D CNN | **86.37 %** | **79.74 %** | **✅** |
| **GPU SOTA (참고, NPU 비호환)** | | | | | |
| CTR-GCN [4] | 2021 | GCN ×4 | 92.4 % | 88.9 % | ❌ |
| PoseConv3D [2] | 2022 | 3D CNN | 93.7 % | 86.5 % | ❌ |
| SkateFormer [3] | 2024 | Transformer | 93.5 % | 89.4 % | ❌ |
| BlockGCN [51] | 2024 | GCN | 93.1 % | 90.3 % | ❌ |
| InfoGCN+ [53] | 2024 | GCN | 93.4 % | 90.4 % | ❌ |

> PSP-Net 은 **NPU 호환 2D CNN 카테고리 내** TSSI (2019) 79.2 % 대비 +7.17 %p 향상. GPU SOTA 와의 −6~11 %p 격차는 dynamic graph / attention 표현력 부재로 인한 deployment trade-off.

## 자세한 결과

전체 측정값과 분석은 [PAPER_KO.md](../paper/PAPER_KO.md) 참조.
