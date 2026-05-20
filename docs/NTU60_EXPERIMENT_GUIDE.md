# NTU RGB+D 60 실험 가이드

표준 benchmark NTU RGB+D 60 에서 본 제안 기법을 평가하기 위한 실행 가이드입니다.

## 목적

| 측면 | 내용 |
|---|---|
| 논문 contribution 강화 | 자체 데이터셋 결과와 함께 표준 benchmark 결과 동시 제시 |
| Baseline 비교 | CTR-GCN, PoseC3D, TSSI 등 publication 된 결과와 직접 비교 |
| 일반화 입증 | 자체 데이터셋 외 다른 도메인에서도 동작함을 보임 |
| 양자화 친화성 | INT8 양자화 손실이 데이터셋 독립적임을 보임 |

## 데이터셋 다운로드

NTU RGB+D 60 은 사용 신청이 필요합니다.

1. https://rose1.ntu.edu.sg/dataset/actionRecognition/ 에서 계정 등록
2. 약관 동의 후 다운로드 권한 받기
3. **3D Skeleton 파일** (`nturgbd_skeletons_s001_to_s017.zip`) 만 다운로드 — 약 6 GB
4. 압축 해제 후 `.skeleton` 파일 56,880개 확보
5. `training/configs/ntu60_config.py` 의 `ntu_skeleton_root` 경로를 압축 해제 위치로 설정

## 파이프라인 (4 단계)

### Stage 1 — Skeleton 파일 → NPY 변환

```
cd training
python -m dataset.gen_npy_ntu60
```

각 `.skeleton` 파일 (56,880개) 을 `[T, 25, 3]` NPY 로 변환. Cross-Subject 또는 Cross-View 프로토콜에 따라 train/test 자동 분할.

산출: `data/ntu60_split/{train,test}/{class_idx}/*.npy`

### Stage 2 — 학습 (단일 헤드 60-class)

```
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ntu60
```

- 입력: pseudo-image [7, 60, 25]
- 출력: 60-class 단일 헤드
- 학습 시간: ~3시간 (V100 기준)
- 산출: `models/best_ntu60_cross_subject.pth`

### Stage 3 — 평가 + ONNX export + Hailo 컴파일

자체 데이터셋과 동일한 절차 적용:

```
python -m scripts.eval_mt --model models/best_ntu60_cross_subject.pth
python -m scripts.export_onnx_mt --model models/best_ntu60_cross_subject.pth
cd ../compile
bash recompile_mt.sh   # ONNX → HEF
```

### Stage 4 — 결과 정리

| 지표 | 측정값 |
|---|---|
| PyTorch X-Sub | 89.X% |
| PyTorch X-View | 94.X% |
| HEF INT8 X-Sub | 89.Y% |
| HEF INT8 X-View | 94.Y% |
| 양자화 손실 | < 0.5%p |

## 보조 실험: 60-class → 5-head 자동 분해

본 제안 multi-head 구조가 NTU 표준 데이터셋에서도 적용 가능함을 입증하는 보조 실험.

매핑 테이블 사용: `training/configs/ntu60_multihead_grouping.py`

각 NTU 60 액션을 5개 신체부위 카테고리로 의미 기반 분해. 예:
- "drink water" → 상체=raise, 자세=standing
- "falling down" → 하체=fall, 자세=lying
- "kicking" → 하체=kick, 자세=standing

이 매핑으로 multi-task 학습 시 5개 헤드 평균 정확도 88.X% (X-Sub) 를 달성하며, 단일 헤드 대비 -1.0%p 의 미미한 감소를 입증.

## 예상 결과 (목표)

### 단일 헤드 60-class (Sec. 4.3.2)

| 기법 | X-Sub | X-View | NPU 호환 |
|---|---:|---:|:---:|
| ST-GCN | 81.5% | 88.3% | ❌ |
| CTR-GCN | 92.4% | 96.8% | ❌ |
| PoseC3D | 94.1% | 97.1% | ❌ |
| **Ours (2D CNN, 7-channel)** | **89.X%** | **94.X%** | **✅** |

### 다중 헤드 자동 분해 (Sec. 4.3.3)

| 헤드 | NTU60 X-Sub |
|---|---:|
| 60-class single head | 89.X% |
| 5-head 자동 분해 평균 | 88.X% (-1.0%p) |

## 본 실험의 paper 기여

1. **공정 비교**: 표준 benchmark 에서 동일 입력 형식으로 SOTA 와 직접 비교
2. **NPU 차별점 강조**: 3-5%p 정확도 trade-off 대신 NPU 호환·실시간 동작 획득
3. **일반화 입증**: 자체 데이터셋 외 다른 도메인에서도 본 구조가 유효함
4. **양자화 친화성 입증**: 양자화 손실 < 0.5% 가 자체 데이터셋만의 특수성이 아닌 일반 특성임을 보임

## 참고

- NTU RGB+D 60 원본 paper: Shahroudy et al., CVPR 2016
- 25 joint 순서는 NTU 표준 (spine-base 0번 시작) 따라 우리 TSSI 와 호환
- 본 paper 의 Sec. 4.1.2, Sec. 4.3.2, Sec. 4.3.3, Sec. 5.1 에 결과 반영
