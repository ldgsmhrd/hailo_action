# Multi-task Action Recognition 모델 최종 리포트

NPU 보드 (Hailo-8) 실시간 행동 인식 시스템 — ResNet18 백본 + 5 헤드 멀티태스크.

---

## 1. 데이터셋

### 원본 출처
AICA GPU 서버 `/home/ubuntu/safemotion/action_dataset/` — 안전모션 v22 어노테이션.

7 개 dataset 디렉터리 통합:

| dataset | 위치 |
|---|---|
| bus 20250521 | `bus/20250521/clip` |
| bus 20250523 | `bus/20250523/clip` |
| bus 20250526 | `bus/20250526/clip` |
| kids_cafe 20250512 | `kids_cafe/20250512/clip` |
| kids_cafe 20250513 | `kids_cafe/20250513/clip` |
| kids_cafe 20250515 | `kids_cafe/20250515/clip` |
| kids_cafe 20250516 | `kids_cafe/20250516/clip` |

### 데이터 가공

| 단계 | 결과 |
|---|---|
| 원본 unique clip | **13886** |
| NPY 변환 성공 | **13878** (실패 8, 프레임 수 부족) |
| Train / Val / Test 분할 | 9645 / 2176 / **2057** (7:1.5:1.5, person id 그룹화) |

### 입력 포맷
- 17 COCO keypoint → 25 TSSI joint reorder
- 60 frame 시퀀스 → pseudo-image `[7, 60, 25]`
- 7 채널: position(x,y), confidence, velocity(vx,vy), bone angle(dx,dy)
- 사람별 좌표 정규화 (bbox 기준 0~1)

### 클래스 매핑 규칙

사용자 공식 라벨 표 22행 + 자세 14행 기반.  
**원칙: "정적 = pose 가 담당, 동적 = action 이 담당, 겹치는 건 pose 로"**

- action_lower 의 raw 31(서있기), 32(앉아있기) 등 정적 라벨 → pose 헤드가 처리하므로 action_lower 에서는 none 으로
- action_upper 의 raw 1(허리구부리기), 2(허리펴기) → pose 의 standing-bending 으로 흡수
- action_upper 의 raw 7(가리키기), 21(양손들기) → hand 헤드가 담당
- 의미 모호 또는 데이터 0 인 raw 는 -1 (학습 제외)

데이터 활용률 **100%** — 13878 clip 모두 5 헤드 라벨 모두 추출.

---

## 2. 클래스별 데이터 수

### 상체 action_upper (6 클래스)

| idx | 한글(영어) | train | val | test |
|---:|---|---:|---:|---:|
| 0 | 없음 (none) | 8114 | 1770 | 1721 |
| 1 | 펀치 (punch) | 209 | 36 | 34 |
| 2 | 손흔들기 (wave) | 334 | 56 | 73 |
| 3 | 손뼉치기 (clap) | 397 | 129 | 86 |
| 4 | 손올리기 (raise) | 647 | 124 | 143 |
| 5 | 손내리기 (put-down) | 5 | 0 | 0 |

### 하체 action_lower (10 클래스)

| idx | 한글(영어) | train | val | test |
|---:|---|---:|---:|---:|
| 0 | 없음 | 7123 | 1617 | 1504 |
| 1 | 서성이기 (pacing) | 354 | 59 | 73 |
| 2 | 걷기 (walk) | 233 | 39 | 66 |
| 3 | 달리기 (run) | 225 | 42 | 41 |
| 4 | 점프-제자리 (jump-still) | 160 | 35 | 34 |
| 5 | 넘어짐 (fall) | 323 | 86 | 85 |
| 6 | 킥 (kick) | 47 | 9 | 13 |
| 7 | 점프-두발 (jump-2feet) | 251 | 47 | 47 |
| 8 | 외발점프 (jump-1leg) | 509 | 91 | 113 |
| 9 | 외발점프-제자리 (jump-1leg-still) | 481 | 90 | 81 |

### 자세 pose (9 클래스)

| idx | 한글(영어) | train | val | test |
|---:|---|---:|---:|---:|
| 0 | 바닥앉기 (sit) | 3045 | 704 | 636 |
| 1 | 의자앉기 (sit-chair) | 278 | 90 | 60 |
| 2 | 무릎꿇기 (kneel-down) | 212 | 20 | 39 |
| 3 | 무릎서기 (knee-standing) | 1 | 1 | 0 |
| 4 | 서있기 (standing) | 428 | 76 | 82 |
| 5 | 허리구부리기 (standing-bending) | 4438 | 923 | 978 |
| 6 | 누워있기 (lying) | 0 | 0 | 0 |
| 7 | 무릎기기 (crawl-pose) | 164 | 45 | 40 |
| 8 | 기타 (other) | 1140 | 256 | 222 |

### 손 hand (3 클래스)

| idx | 한글(영어) | train | val | test |
|---:|---|---:|---:|---:|
| 0 | 없음 | 9124 | 2021 | 1901 |
| 1 | 팔짱끼기 (cross-arms) | 88 | 24 | 24 |
| 2 | 양팔들기 (raise-both) | 494 | 70 | 132 |

### 발 foot (3 클래스)

| idx | 한글(영어) | train | val | test |
|---:|---|---:|---:|---:|
| 0 | 없음 | 8535 | 1869 | 1826 |
| 1 | 다리꼬기 (leg-cross) | 180 | 64 | 36 |
| 2 | 한쪽다리들기 (one-leg-raise) | 991 | 182 | 195 |

**총 31 출력 / 학습된 의미있는 클래스 28개** (데이터 0~5개인 손내리기/무릎서기/누워있기는 사실상 미학습)

---

## 3. 모델 구조

```
입력 [B, 7, 60, 25]  (pseudo-image, NCHW)
    │
    ▼
ResNet18 backbone (ImageNet pretrained)
  - conv1: in_channels=7 (3→7 가중치 partial copy)
  - first_conv_stride=(2, 1) (시간 다운샘플)
  - fc → Identity (feature 512)
    │
    ▼
nn.ModuleDict {
   'action_upper':  Linear(512, 6),
   'action_lower':  Linear(512, 10),
   'pose':          Linear(512, 9),
   'hand':          Linear(512, 3),
   'foot':          Linear(512, 3),
}
    │
    ▼
출력 dict {head_name: logits}
```

총 파라미터: 약 **11.2 M** (ResNet18 그대로 + 5 linear head)

---

## 4. 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | SGD + Nesterov momentum 0.9 |
| Learning rate | 0.05 (cosine annealing → 0) |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 100 |
| AMP | 비활성 (이전 학습에서 NaN loss 발생 경험) |
| GPU | NVIDIA (AICA 서버, CUDA_VISIBLE_DEVICES=3) |
| 학습 시간 | 약 11분 (100 epoch × 6.3s/epoch) |

### Augmentation
- horizontal flip (좌우 keypoint swap)
- coordinate noise σ=0.01
- confidence dropout 5%
- temporal shift ±5 frames

### Loss
헤드별 weighted Cross-Entropy + class weight (effective number of samples, β=0.999):
- action_lower 1.0 / action_upper 1.0 / pose 1.0 / hand 0.5 / foot 0.5

빈/희소 클래스 weight 자동 0 처리 → 학습 무시.

### 학습 진행
- Best 갱신 마지막: **epoch 99 → val avg 96.21%**
- 주기적 collapse (10~30 epoch 마다 한 번씩 hand head 가 majority class 로 무너짐) → 다음 epoch 즉시 회복
- 마지막 30 epoch 부터 매우 안정 (95.5~96.2% 진동)

---

## 5. 평가 결과 (PyTorch)

Test set 2057 clips 기준:

| Head | Test accuracy |
|---|---:|
| 상체 action_upper | **97.28%** |
| 하체 action_lower | **95.43%** |
| 자세 pose | **86.73%** |
| 손 hand | **99.85%** |
| 발 foot | **99.61%** |
| **평균** | **95.78%** |

### 클래스별 recall

#### 상체 (action_upper)
| 클래스 | recall | 헷갈린 top-3 |
|---|---:|---|
| 없음 | 98.0% | 펀치(21), 손뼉치기(8), 손올리기(5) |
| 펀치 | 82.4% | 없음(6) |
| 손흔들기 | 87.7% | 손올리기(9) |
| 손뼉치기 | **100%** | - |
| 손올리기 | 95.1% | 손흔들기(7) |
| 손내리기 | n=0 (미학습) | - |

#### 하체 (action_lower)
| 클래스 | recall | 헷갈린 top-3 |
|---|---:|---|
| 없음 | 97.3% | 서성이기(13), 걷기(6), 점프-제자리(6) |
| 서성이기 | 79.5% | 없음(11), 걷기(2), 킥(2) |
| 걷기 | 77.3% | 없음(11), 달리기(3) |
| 달리기 | 92.7% | 걷기(2) |
| 점프-제자리 | 94.1% | 없음(2) |
| 넘어짐 | 89.4% | 없음(8) |
| 킥 | 69.2% | 서성이기(3), 없음(1) |
| 점프-두발 | 89.4% | 없음(3), 걷기(2) |
| 외발점프 | **100%** | - |
| 외발점프-제자리 | **100%** | - |

#### 자세 (pose)
| 클래스 | recall | 헷갈린 top-3 |
|---|---:|---|
| 바닥앉기 | 79.4% | **허리구부리기(110)**, 서있기(17) |
| 의자앉기 | 95.0% | 바닥앉기(3) |
| 무릎꿇기 | 92.3% | 기타(3) |
| 무릎서기 | n=0 | - |
| 서있기 | 81.7% | 바닥앉기(7), 허리구부리기(5) |
| 허리구부리기 | 90.7% | **바닥앉기(81)** |
| 누워있기 | n=0 | - |
| 무릎기기 | 87.5% | 허리구부리기(4) |
| 기타 | 88.7% | 바닥앉기(9), 허리구부리기(7) |

#### 손 (hand)
| 클래스 | recall | 헷갈린 top-3 |
|---|---:|---|
| 없음 | 100% | - |
| 팔짱끼기 | 100% | - |
| 양팔들기 | 97.7% | 없음(3) |

#### 발 (foot)
| 클래스 | recall | 헷갈린 top-3 |
|---|---:|---|
| 없음 | 99.8% | 한쪽다리들기(4) |
| 다리꼬기 | 91.7% | 없음(3) |
| 한쪽다리들기 | 99.5% | 없음(1) |

### 주요 오분류 패턴

1. **바닥앉기 ↔ 허리구부리기** (pose, 양방향 81+110) — 라벨링 자체가 모호 (앉아서 허리 굽힌 자세 vs 서서 허리 굽힌 자세)
2. **서성이기/걷기 ↔ 없음** (lower, 각 11개씩) — 약한 모션 미감지
3. **킥 (69.2%)** — 데이터 47개로 부족, 서성이기와 혼동
4. **손흔들기 ↔ 손올리기** (upper, 9개) — 동작 유사

---

## 6. 변환 과정 (ONNX → HEF)

### Stage 1 — PyTorch → ONNX
- `torch.onnx.export(dynamo=False)` (legacy TracedExport — Hailo 호환)
- opset 11, do_constant_folding=True
- 출력 5개: action_upper(6) / action_lower(10) / pose(9) / hand(3) / foot(3)
- 산출 `action_resnet_mt.onnx` — **43 MB** (single file)

### Stage 2 — Hailo DFC
3 단계 (`recompile_mt.sh` 내부에서):

```
hailo parser onnx → action_resnet_mt.har
hailo optimize    → action_resnet_mt_quantized.har  (INT8 PTQ)
hailo compiler    → action_resnet_mt.hef
```

| 단계 | 시간 | 비고 |
|---|---:|---|
| Parse | 1.4s | ONNX → HAR |
| Optimize | ~3분 | calibration 1500 sample, PTQ |
| Compile | 8s | HAR → HEF |
| **총** | **~5분** | |

산출: **`action_resnet_mt.hef` — 7.0 MB**

### 양자화 — 스킵된 최적화

GPU 패스스루 없이 docker 에서 실행돼 optimization_level=0 으로 강등:
- Bias Correction skipped
- Adaround skipped
- QAT Fine-Tuning skipped
- LayerNorm/Matmul decomposition skipped

ResNet18 같은 단순 구조라 PTQ 기본만으로도 손실 거의 없음 → 재컴파일 불필요로 판단.

### HEF 출력 매핑

Hailo 컴파일러가 fc1~fc5 를 ModuleDict 순서와 무관하게 부여. shape 기반 매핑 필요:

| HEF 출력 | shape | 실제 head |
|---|---:|---|
| fc1 | 10 | action_lower |
| fc2 | 6 | action_upper |
| fc3 | 3 | foot |
| fc4 | 3 | hand |
| fc5 | 9 | pose |

---

## 7. 변환 후 검증 (HEF on Hailo-8)

Test set 2057 clips 그대로, 보드 Hailo NPU 직접 추론.

| Head | PyTorch | **HEF** | Gap |
|---|---:|---:|---:|
| 상체 action_upper | 97.28% | **97.23%** | -0.05% |
| 하체 action_lower | 95.43% | **95.43%** | 0.00% |
| 자세 pose | 86.73% | **86.78%** | **+0.05%** |
| 손 hand | 99.85% | **99.85%** | 0.00% |
| 발 foot | 99.61% | **99.37%** | -0.24% |
| **평균** | **95.78%** | **95.73%** | **-0.05%** |

### 추론 속도
- **639 samples/s** (단일 NPU, batch 1)
- → 한 샘플당 약 1.56 ms
- 평균 0.4% NPU 점유 (sliding stride 8 frame 적용 시)

INT8 양자화 거의 무손실. **PyTorch ≈ HEF**.

---

## 8. 실시간 시스템 구성

### 보드
- 호스트: etri-board (192.168.1.163)
- HW: etri-ADL-N (Hailo-8 ×4 NPU)
- 컨테이너: `arn-npu`

### Multi-process 아키텍처
```
[RTSP reader thread]
        ↓ shared memory (latest-only)
[Pose Worker  (Core 1, NPU0 — yolov8m_pose)]
        ↓ Queue (tracked detections + ByteTracker)
[Action Worker (Core 2, NPU1 — action_resnet_mt)]
        ↓ shared memory (annotated frame)
[MJPEG HTTP Server]
        ↓
http://192.168.1.163:9999/stream
```

### 표시
각 추적 ID 마다 5 헤드 라벨 동시 표시 (한글(영어) NN%):
```
상체 펀치(punch) 87%
하체 걷기(walk) 78%
자세 서있기(standing) 91%
손 없음(none) 100%
발 없음(none) 100%
```

상단 status bar: `FPS / Tracks / Frames / NPU0(Pose) % / NPU1(Action) %`

### 측정값 (실측)
- FPS 21~22
- NPU0 (Pose, yolov8m) 약 44~46%
- NPU1 (Action MT) 약 0.3~0.5%

---

## 9. 알려진 한계

### 데이터 측 한계
- **상체+하체 동시 active 라벨 0개** — 원본 데이터셋 자체가 "한 번에 한 액션" 정책. 펀치하면서 걷기 같은 복합 동작 실시간 감지 어려움.
- 손내리기 (5), 무릎서기 (1), 누워있기 (0) — 학습 데이터 부족으로 미학습.
- 킥 (47) — 학습 가능하지만 recall 69.2%로 약함.

### 라벨 모호성
- 바닥앉기 ↔ 허리구부리기 (pose) — 동일 자세를 다르게 라벨한 케이스 다수 → 모델도 자연스럽게 모호.
- 서성이기/걷기 ↔ 없음 — 약한 모션 경계 모호.

### 양자화
- INT8 손실은 무시할 수준이나 GPU 패스스루 환경에서 Bias Correction + Adaround + QAT 추가 시 이론상 안전마진 증가.

---

## 10. 테스트 영상

> **[영상 첨부 자리]**  
> URL: http://192.168.1.163:9999/stream  
> 또는 mp4 첨부: `_____________________`

녹화 시 포함할 내용:
1. 모델 로드 + 초기화 (`Multi-task head→vstream` 로그)
2. 사람 한 명 — 다음 동작 순차 시연
   - 가만히 서있기 → 자세=서있기 / 하체=없음 / 상체=없음
   - 앉기 → 자세=바닥앉기
   - 걷기 → 하체=걷기 / 자세=서있기
   - 달리기 → 하체=달리기
   - 점프 → 하체=점프-제자리
   - 펀치 → 상체=펀치 / 하체=없음
   - 손흔들기 → 상체=손흔들기
   - 손뼉치기 → 상체=손뼉치기
   - 손올리기 → 상체=손올리기
   - 양팔들기 → 손=양팔들기
   - 팔짱끼기 → 손=팔짱끼기
   - 다리꼬기 (앉아서) → 발=다리꼬기 / 자세=바닥앉기
   - 넘어짐 → 하체=넘어짐 (안전 핵심 이벤트)
3. 좌상단 상태 바 — FPS, NPU 사용률 변화 확인
4. 라벨 텍스트 한글 정상 렌더링 확인

녹화 명령 (보드에서):
```bash
ffmpeg -i http://localhost:9999/stream -t 60 -c:v libx264 -preset fast \
    /share/multitask_demo_$(date +%Y%m%d_%H%M).mp4
```
