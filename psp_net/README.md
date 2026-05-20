# PSP-Net — Partitioned Skeletal Pseudo-image Network

NPU 친화적 차세대 모델. 기존 ResNet18 baseline 대비:
- 파라미터 1/10 (11.2M → 1.07M)
- NPU 호환 100% 유지
- 정확도 +(3-6%p) 목표

## 구성 요소

| 컴포넌트 | 역할 | NPU 호환 |
|---|---|:---:|
| ① Body-part block pseudo-image | 17 COCO joint → 5 신체부위 × 5 슬롯 (25) 재배열 | ✅ |
| ② BodyPartConv | 부위마다 독립 conv (groups=5 효과) | ✅ |
| ③ S-T Decoupled Block | 공간 (1×3) + 시간 (3×1) 분리 | ✅ |
| ④ Multi-Scale Temporal | dilation 1/2/4/8 병렬 | ✅ |
| ⑤ Squeeze-Excitation | NPU 호환 채널 attention | ✅ |
| ⑥ 5-Head 분류 | 기존과 동일 | ✅ |

NPU 비호환 op (LayerNorm, Multi-Head Attention, Softmax over long seq) 일체 미사용.

## 폴더 구조

```
psp_net/
├── configs/
│   └── psp_config.py        # 학습 hyperparams
├── dataset/
│   ├── joint_grouping.py    # 17 → 25 신체부위 매핑
│   ├── encoder.py           # 7채널 pseudo-image 인코더 (body-part 순서)
│   └── psp_dataset.py       # 기존 NPY 재사용 + body-part 인코딩
├── models/
│   └── psp_net.py           # PSP-Net 본체
└── scripts/
    └── train_psp.py         # 학습 entry
```

## 사용법

### 학습 (AICA 서버)

```bash
cd multitask-action-recognition
CUDA_VISIBLE_DEVICES=0 python -m psp_net.scripts.train_psp
```

기존 NPY (`data/split_mt_v3/`) 그대로 재사용. PSP-Net 데이터셋이 17 keypoint → 25 body-part 인코딩 자동 변환.

### 모델 forward 검증

```bash
python -m psp_net.models.psp_net
# 출력: PSP-Net 파라미터: 1.07M / shape 확인
```

## 모델 흐름

```
입력 [B, 7, 60, 25]   (body-part 순서: head(5) | L-arm(5) | R-arm(5) | L-leg(5) | R-leg(5))
   ↓ BodyPartConv (in=7, out=64)         부위마다 독립 처리, joint 25→5
[B, 64, 60, 5]
   ↓ Cross-part 1×1 conv                 부위 간 정보 교환
[B, 64, 60, 5]
   ↓ STDecoupledBlock(64→128) + Pool     공간+시간 분리, 시간 축 다운샘플
[B, 128, 30, 5]
   ↓ STDecoupledBlock(128→256) + Pool
[B, 256, 15, 5]
   ↓ STDecoupledBlock(256→256)
[B, 256, 15, 5]
   ↓ MultiScaleTemporal (dilation 1,2,4,8)
[B, 256, 15, 5]
   ↓ SqueezeExcitation
[B, 256, 15, 5]
   ↓ AdaptiveAvgPool2d → flatten
[B, 256]
   ↓ 5 Linear head
{upper, lower, pose, hand, foot}
```

## 데이터 형식

- 입력 NPY: 기존 `data/split_mt_v3/{train,val,test}/action_lower/{class_idx}/*.npy` 그대로
- 각 NPY: `[T=60, 17, 3]` (x, y, conf)
- `.meta.json` 의 5 head 라벨도 그대로 사용

## 17 → 25 신체부위 매핑

| 슬롯 | 부위 | COCO 인덱스 |
|---|---|---|
| 0-4 | Head | 코, 왼눈, 오른눈, 왼귀, 오른귀 |
| 5-9 | L-arm | 왼어깨, 왼팔꿈치, 왼손목, PAD, PAD |
| 10-14 | R-arm | 오른어깨, 오른팔꿈치, 오른손목, PAD, PAD |
| 15-19 | L-leg | 왼골반, 왼무릎, 왼발목, PAD, PAD |
| 20-24 | R-leg | 오른골반, 오른무릎, 오른발목, PAD, PAD |

Padding 슬롯은 confidence=0 으로 채워져 자연 무시.

## 기존 ResNet18 baseline 과 비교 목표

| 측면 | ResNet18 (현재) | PSP-Net (목표) |
|---|---:|---:|
| 파라미터 | 11.2M | 1.07M (1/10) |
| 자체 데이터 정확도 | 95.78% | 96%+ |
| 양자화 손실 | -0.05% | -0.1% 이하 |
| NPU 호환 | ✅ | ✅ |
| Hailo-8 FPS | 639 | ~1500 (예상) |

## 다음 단계

1. AICA GPU 서버 학습 (~10분, A100 기준)
2. PyTorch test 평가
3. ONNX export
4. Hailo HEF 컴파일
5. NPU 검증
6. ResNet18 baseline 과 직접 비교
