# 특허 분석 — Multi-task Skeleton Action Recognition

> 작성일 2026-05-20
> 대상 시스템: Hailo NPU 기반 멀티태스크 행동·자세 인식 (5-head ResNet18 + TSSI 인코딩)

---

## Executive Summary

본 시스템은 **9개 기술 요소**로 분해 가능하며, 정밀 prior art 조사 결과 **출원 가치가 있는 요소는 단 2개**입니다.

| 요약 항목 | 결과 |
|---|---|
| **출원 가치 있는 청구항 수** | 2 (메인 1, 보조 1) |
| **메인 청구항 등록 가능성** | **60%** |
| **권장 출원 형태** | 한국 → PCT 우선권 (12개월 이내) |
| **예상 비용** | 한국 ~250만원 / PCT 추가 ~500만원 |
| **권장 일정** | 2-3주 내 변리사 의뢰 → 1개월 내 출원 |

---

## 1. 기술 요소 전체 분해

시스템을 **9개 요소**로 분해하고 각 요소를 독립 평가합니다.

```
┌─ A. 입력 / 데이터 ───────────────────────────────┐
│  A-1  17 COCO keypoint 시퀀스                     │
│  A-2  25 joint TSSI 변환                          │
│  A-3  7채널 pseudo-image                          │
├─ B. 모델 구조 ────────────────────────────────┤
│  B-1  ResNet18 백본 (in_channels=7)              │
│  B-2  공유 백본 + 5-head 분해 구조               │
├─ C. 라벨 / 학습 ───────────────────────────────┤
│  C-1  v22 → 28 simplified 라벨 매핑              │
│  C-2  정적/동적 라벨 충돌 자동 해결 규칙          │
│  D-1  Multi-task weighted CE + class weight       │
├─ E. 양자화 ──────────────────────────────────┤
│  E-1  ONNX (legacy export) + Hailo INT8 PTQ      │
├─ F. 배포 ──────────────────────────────────┤
│  F-1  단일 NPU + VDevice scheduler 시분할         │
│  F-2  다중 NPU 분산 (Multi-process worker)        │
├─ G. 시스템 통합 ───────────────────────────────┤
│  G-1  RTSP → NPU → MJPEG 파이프라인               │
└─────────────────────────────────────────────┘
```

---

## 2. 요소별 prior art 정밀 조사

### A-1. 17 COCO keypoint 시퀀스

| 항목 | 내용 |
|---|---|
| **구현** | 60 frame × 17 joint × 3 (x, y, conf) |
| **Prior art** | COCO 데이터셋 표준 (Lin et al. 2014) / OpenPose / MediaPipe / YOLOv8-pose |
| **신규성** | ❌ 없음 — 산업 표준 |
| **출원 / 등록** | 0% / 0% |

---

### A-2. 25 joint TSSI 변환

| 항목 | 내용 |
|---|---|
| **구현** | 17 COCO → 25 joint (tree-structured order, depth-first traversal) |
| **Prior art (학술)** | • [Caetano et al., arXiv 1909.05704 (2019)](https://arxiv.org/pdf/1909.05704) — TSSI 원본 논문 <br> • NTU-RGBD 데이터셋의 25 joint 표준 <br> • [Sign Language Recognition (CVPRW 2023)](https://openaccess.thecvf.com/content/CVPR2023W/LatinX/papers/Laines_Isolated_Sign_Language_Recognition_Based_on_Tree_Structure_Skeleton_Images_CVPRW_2023_paper.pdf) |
| **신규성** | ❌ 없음 — 2019년부터 다수 publication |
| **출원 / 등록** | 0% / 0% |

---

### A-3. 7채널 pseudo-image

| 항목 | 내용 |
|---|---|
| **구현** | (x, y, confidence, vx, vy, bone_dx, bone_dy) 7채널 stack |
| **Prior art** | • 3채널 (x, y, z) — TSSI 표준 <br> • 4채널 (+confidence) — 일부 publication <br> • 5채널 (+velocity) — [Skeleton Edge Motion Networks (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0925231220315824) <br> • **7채널 (+bone direction 2개) 정확히 일치하는 publication 없음** |
| **신규성** | ⭐ 약한 신규성 (채널 조합 차이) |
| **진보성** | ⭐ 약함 — 자명한 확장으로 거절될 위험 |
| **출원 / 등록** | 30% / 15% |

---

### B-1. ResNet18 백본 (in_channels=7)

| 항목 | 내용 |
|---|---|
| **구현** | ImageNet pretrained, conv1 3→7 채널 partial copy, first_conv_stride=(2,1) |
| **Prior art** | He et al. ResNet 2015 / 전이학습 표준 / asymmetric stride 일부 publication |
| **신규성** | ❌ 없음 |
| **출원 / 등록** | 0% / 0% |

---

### B-2. 공유 백본 + 5-head 분해 ⚠️ 

| 항목 | 내용 |
|---|---|
| **구현** | 공유 ResNet18 → 5 Linear head (upper / lower / pose / hand / foot) |
| **Prior art 신규 발견** | 🚨 **[B-MoE: Body-Part-Aware Mixture-of-Experts (arXiv 2603.24245)](https://arxiv.org/pdf/2603.24245)** — "head, body, upper limbs, lower limbs" 4개 신체부위 expert. 매우 유사 구조 |
| | • [DynaPURLS (arXiv 2512.11941)](https://arxiv.org/pdf/2512.11941) — Part-aware skeleton-based zero-shot |
| | • [Multimodal Skeleton via Decomposition+Composition (Springer 2025)](https://link.springer.com/article/10.1007/s11633-025-1583-z) |
| | • [CN114821640B](https://patents.google.com/patent/CN114821640B/en) — Multi-stream GCN 특허 |
| **신규성** | ⭐⭐ 중간 — B-MoE 와 컨셉 매우 유사하나 5-head 분해 (hand + foot 분리), CNN 기반, INT8 양자화 통합 점은 차이 |
| **진보성** | ⭐ 약함 — B-MoE 가 광범위한 청구 가능 |
| **데이터 한계** | ⚠️ 학습 데이터에 상체+하체 동시 active 라벨 0개. "동시 검출 기능" 청구는 약함 → **구조 청구로 좁혀야** |
| **출원 / 등록** | 35% / 15% |

---

### C-1. v22 → 28 simplified 라벨 매핑

| 항목 | 내용 |
|---|---|
| **구현** | 안전모션 v22 30+ raw → 28 simplified 클래스 |
| **Prior art** | private 데이터셋 가공 작업 — 일반적 데이터 전처리 |
| **신규성** | ⭐ 약함 — 데이터 가공 작업 |
| **출원 / 등록** | 5% / 0% |

---

### C-2. **정적/동적 라벨 충돌 자동 해결 규칙** ⭐ 메인 청구

| 항목 | 내용 |
|---|---|
| **구현** | **원칙**: 정적 자세 라벨 → pose 헤드 / 동적 동작 라벨 → action 헤드 / 겹치면 pose 우선 <br> **예시**: raw 31 "서있기" (5348 clips) → pose=standing, action_lower=none |
| **Prior art** | • [Skeleton Noisy Labels (arXiv 2403.09975)](https://arxiv.org/pdf/2403.09975) — 노이즈 라벨 일반 처리 <br> • [Fine-grained action: dynamic/static subgroups](https://arxiv.org/pdf/2501.02593) — 모션 variance 기반 subgroup 구분 (학습 시 augmentation 용도, 라벨 충돌 해결 아님) <br> • **헤드 간 자동 라벨 충돌 해결 / 데이터 가공 방법론 publication 검색 안 됨** |
| **신규성** | ⭐⭐⭐⭐ **강함** — 직접 prior art 없음 |
| **진보성** | ⭐⭐⭐ **강함** — 라벨 모호성으로 인한 모델 성능 저하 문제를 명확한 규칙으로 해결 |
| **데이터 한계 영향** | ❌ 없음 — 라벨링 방법론 자체이므로 모델 성능과 독립적으로 입증 가능 |
| **실증** | ✅ 13878 clip 가공 결과 + train/val/test 분포 + 양자화 후 정확도 유지 |
| **출원 / 등록** | **80% / 60%** ⭐ |

---

### D-1. Multi-task weighted CE + class weight

| 항목 | 내용 |
|---|---|
| **구현** | 헤드별 weighted CE + effective number class weight (β=0.999) |
| **Prior art** | • Cui et al. "Class-Balanced Loss" (CVPR 2019) — effective number 원본 <br> • Multi-task weighted loss 광범위 |
| **신규성** | ❌ 없음 |
| **출원 / 등록** | 0% / 0% |

---

### E-1. ONNX + Hailo INT8 PTQ

| 항목 | 내용 |
|---|---|
| **구현** | torch.onnx.export (dynamo=False) + Hailo DFC + 1500 calibration |
| **Prior art** | • [Quant-Trim (arXiv 2511.15300)](https://arxiv.org/pdf/2511.15300) <br> • [PatSnap "Quantization Pipeline" 특허 클러스터 분석](https://www.patsnap.com/resources/blog/rd-blog/quantized-ai-models-on-edge-chips-patsnap-eureka/) — 2020~2026 다수 출원 <br> • Hailo SDK 자체 공식 워크플로우 |
| **신규성** | ❌ 없음 — 표준 PTQ + 알려진 워크어라운드 |
| **출원 / 등록** | 0% / 0% |

---

### F-1. 단일 NPU + scheduler 시분할

| 항목 | 내용 |
|---|---|
| **구현** | `HailoSchedulingAlgorithm.ROUND_ROBIN` + 두 HEF 동시 로드 |
| **Prior art** | Hailo SDK 공식 기능 — `VDevice.create_params()` API |
| **신규성** | ❌ 없음 — SDK 활용 |
| **출원 / 등록** | 0% / 0% |

---

### F-2. 다중 NPU 분산 (Multi-process worker)

| 항목 | 내용 |
|---|---|
| **구현** | Pose worker (NPU0) + Action worker (NPU1) + shared memory + Queue |
| **Prior art** | Hailo TID_05 공식 예제 동일 구조 |
| **신규성** | ❌ 없음 |
| **출원 / 등록** | 0% / 0% |

---

### G-1. RTSP → NPU → MJPEG 파이프라인

| 항목 | 내용 |
|---|---|
| **구현** | RTSP 입력 → NPU 추론 → MJPEG HTTP 송출 |
| **Prior art** | 보안 카메라 / NVR 표준 구성 |
| **신규성** | ❌ 없음 |
| **출원 / 등록** | 0% / 0% |

---

## 3. 종합 매트릭스

| 카테고리 | 요소 | 신규성 | 진보성 | 데이터한계영향 | **출원** | **등록** |
|---|---|:---:|:---:|:---:|:---:|:---:|
| A. 데이터 | A-1. 17 keypoint | ❌ | ❌ | - | 0% | 0% |
| | A-2. 25 TSSI | ❌ | ❌ | - | 0% | 0% |
| | A-3. 7채널 pseudo-image | ⭐ | ⭐ | 없음 | 30% | 15% |
| B. 모델 | B-1. ResNet18 백본 | ❌ | ❌ | - | 0% | 0% |
| | B-2. 5-head 분해 | ⭐⭐ | ⭐ | **약화** | 35% | 15% |
| C. 라벨 | C-1. 라벨맵 매핑 | ⭐ | ❌ | - | 5% | 0% |
| | **C-2. 정적/동적 충돌 해결** | ⭐⭐⭐⭐ | ⭐⭐⭐ | **없음** | **80%** | **60%** |
| D. 학습 | D-1. weighted loss | ❌ | ❌ | - | 0% | 0% |
| E. 양자화 | E-1. PTQ INT8 | ❌ | ❌ | - | 0% | 0% |
| F. 배포 | F-1. NPU scheduler | ❌ | ❌ | - | 0% | 0% |
| | F-2. Multi-NPU 분산 | ❌ | ❌ | - | 0% | 0% |
| G. 통합 | G-1. RTSP→MJPEG | ❌ | ❌ | - | 0% | 0% |

---

## 4. 권장 청구항 (구체적)

### 🏆 메인 청구항 — 출원 80% / 등록 60%

**제목**: *"신체부위별 멀티 헤드 행동·자세 분류 모델 학습을 위한 라벨 충돌 자동 해결 방법 및 그 시스템"*

**청구항 1 (방법)**

> 행동·자세 분류 학습 데이터를 가공하는 방법에 있어서,
>
> (a) 동일 비디오 클립에 대해 **복수의 신체부위 카테고리** (상체 동작, 하체 동작, 자세, 손 동작, 발 동작) 라벨이 부여된 원시 라벨 집합을 입력받는 단계;
>
> (b) 상기 원시 라벨 중 **정적 자세를 나타내는 라벨**을 자세 카테고리에 우선 할당하고, 동적 동작을 나타내는 라벨을 동작 카테고리에 할당하는 단계;
>
> (c) 동일 raw 라벨이 두 카테고리에 모두 매핑될 가능성이 있는 경우, **정적 자세 카테고리에 자동 흡수**하고 동작 카테고리에서는 "없음" 으로 처리하는 단계;
>
> (d) 상기 가공된 라벨로 공유 백본 + 복수의 분리 분류 헤드를 가지는 신경망 모델을 학습시키는 단계;
>
> 를 포함하는 라벨 충돌 자동 해결 방법.

**청구항 2 (구체화)**

> 청구항 1에 있어서, 상기 (c) 단계의 "정적 자세 카테고리 우선 흡수" 규칙은 다음을 포함한다:
> - "서있기", "앉아있기", "허리 굽힘" 등 자세 명사 라벨 → pose 헤드
> - "걷기", "달리기", "넘어짐" 등 동작 동사 라벨 → action 헤드
> - "허리 구부리기" (동작) 가 "허리 굽힘" (자세) 와 의미 중복 시 → pose 우선

**청구항 3 (시스템)**

> 입력 영상의 스켈레톤 시퀀스를 처리하여 행동·자세를 동시 출력하는 시스템에 있어서,
> 공유 백본과, 상기 백본에 연결된 복수의 분리 분류 헤드 (상체 동작, 하체 동작, 자세, 손, 발)를 포함하고,
> 청구항 1의 라벨 충돌 자동 해결 방법으로 가공된 학습 데이터를 사용하여 학습된 시스템.

---

### 🥈 보조 청구항 — 결합 시 +10%p

**청구항 4 (보조)**

> 청구항 1의 학습 데이터에 적용되는 입력 표현 방법에 있어서,
> COCO 17 keypoint 시퀀스를 25 joint 트리 구조로 확장하고,
> **7채널 pseudo-image (위치 2 + 신뢰도 1 + 속도 2 + 본 방향 2)** 로 인코딩하는 단계를 포함하는 방법.

---

## 5. 출원 / 등록 가능성 정량 평가

| 시나리오 | 청구항 | 출원 | 등록 |
|---|---|:---:|:---:|
| **A. 메인 단독** | 청구항 1-3 (C-2 만) | 80% | **60%** |
| **B. 메인 + 보조** | 청구항 1-4 (C-2 + A-3) | 75% | **65%** |
| **C. 메인 + 보조 + 5-head 구조** | 청구항 1-5 (C-2 + A-3 + B-2 구조) | 70% | **55%** |

→ **시나리오 B 권장**: 등록 가능성 가장 높고 청구항 너무 넓지 않음.

---

## 6. 출원 전략 / 일정 / 비용

### 단계별 일정

| 단계 | 기간 | 작업 |
|---|---|---|
| **0. 사전 조사** | 1주 | KIPRIS / Google Patents 정밀 조사 + 변리사 1차 자문 |
| **1. 명세서 작성** | 2주 | 변리사 협업, 청구항 정밀화 |
| **2. 한국 출원** | 즉시 | KIPO 출원 (우선권 확보) |
| **3. PCT 출원** | 출원 후 12개월 내 | 해외 진입 시 |
| **4. 심사 청구** | 출원 후 3년 내 | 심사 청구 후 18-24개월 심사 |
| **5. 등록** | 출원 후 36-48개월 | 등록료 납부 |

### 비용 예상

| 항목 | 비용 |
|---|---|
| 한국 출원료 | 약 80만원 |
| 변리사 명세서 작성 | 150-250만원 |
| 심사 청구료 | 약 30만원 |
| 등록료 (등록 시) | 약 30만원 |
| **한국 합계** | **약 290-390만원** |
| PCT 추가 (해외) | 약 500만원 |
| 각국 진입 (US/JP/CN 등) | 국가별 200-400만원 |

---

## 7. 권장 다음 단계

### 즉시 (1-2주)

1. **변리사 1차 자문 미팅** — 청구항 메인 1번 (C-2) 중심 검토
2. **KIPRIS 정밀 검색** 직접 수행:
   - 검색어: "행동인식 + 자세 + 라벨", "골격 + 멀티태스크", "skeleton + multi-head"
   - 분류 코드: G06V 40/20, G06N 3/04, G06V 40/10
3. **데이터 한계 보완**: Mixup 합성 데이터로 상체+하체 동시 검출 입증 (선택)

### 중기 (1-2개월)

1. 명세서 작성 (변리사)
2. 한국 출원 완료
3. 도면 작성 (시스템 아키텍처 + 라벨 충돌 해결 흐름도)

### 장기 (12개월 내)

1. PCT 우선권 결정 (해외 진출 필요성 검토)
2. 데이터 보강 (Mixup 합성) — 청구항 5번 추가 가능
3. 트루엔 등 경쟁사 특허 모니터링

---

## 8. 솔직한 평가

| Q | A |
|---|---|
| 우리 기술이 모두 새로운가? | ❌ 대부분 표준 (TSSI, ResNet18, INT8 PTQ, NPU 배포 등) |
| 진짜 우리만의 것은 무엇? | ✅ **C-2 (정적/동적 라벨 충돌 해결)** — 유일하게 prior art 미발견 |
| B-2 (5-head) 는 왜 약한가? | B-MoE (arXiv 2603.24245) 가 매우 유사. CNN vs MoE 차이로 청구 가능하나 등록 어려움 |
| A-3 (7채널) 단독 출원 가능? | ❌ 진보성 약함 — 기존 5채널 publication 의 자명한 확장 |
| 데이터 한계 (상체+하체 0개) 가 출원에 영향? | C-2 청구는 무관, B-2 구조 청구만 영향 → B-2 는 보조로만 활용 |
| 출원 가치 있나? | ✅ C-2 단독으로도 충분 — 60% 등록 가능성은 통상 출원 평균 대비 양호 |

---

## 9. 결론

> **메인 청구항 — 정적/동적 라벨 충돌 자동 해결 방법 (C-2)** 단독으로 한국 출원 권장.
>
> 청구항 1-3 (방법 + 시스템) + 보조 청구항 4 (7채널 인코딩) 묶음으로 등록 가능성 **65%**.
>
> 트루엔 같은 카메라 제조사 / 경쟁사 견제력 있는 IP 자산 확보 가능.

---

## 참고문헌

### 학술 publication
- [Caetano et al. — Skeleton Image Representation TSSI (arXiv 1909.05704)](https://arxiv.org/pdf/1909.05704)
- [B-MoE — Body-Part-Aware MoE (arXiv 2603.24245)](https://arxiv.org/pdf/2603.24245)
- [Skeleton Edge Motion Networks (ScienceDirect 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0925231220315824)
- [Sign Language with TSSI (CVPRW 2023)](https://openaccess.thecvf.com/content/CVPR2023W/LatinX/papers/Laines_Isolated_Sign_Language_Recognition_Based_on_Tree_Structure_Skeleton_Images_CVPRW_2023_paper.pdf)
- [DynaPURLS — Part-aware Zero-Shot (arXiv 2512.11941)](https://arxiv.org/pdf/2512.11941)
- [Multimodal Skeleton via Decomposition (Springer 2025)](https://link.springer.com/article/10.1007/s11633-025-1583-z)
- [Evolving Skeletons — dynamic/static (arXiv 2501.02593)](https://arxiv.org/pdf/2501.02593)
- [Skeleton Noisy Labels (arXiv 2403.09975)](https://arxiv.org/pdf/2403.09975)
- [Quant-Trim Cross-Platform Quantization (arXiv 2511.15300)](https://arxiv.org/pdf/2511.15300)

### 특허
- [CN114821640B — Skeleton GCN Multi-stream (Google Patents)](https://patents.google.com/patent/CN114821640B/en)
- [CN111184512B — Upper limb rehabilitation action recognition (Patsnap)](https://eureka.patsnap.com/patent-CN111184512B)

### 분석 자료
- [Quantized AI Models on Edge Chips (PatSnap)](https://www.patsnap.com/resources/blog/rd-blog/quantized-ai-models-on-edge-chips-patsnap-eureka/)
- [KIPRIS 한국특허정보 검색](https://www.kipris.or.kr/)
