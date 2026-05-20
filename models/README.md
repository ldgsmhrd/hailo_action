# Models

학습 산출 / 컴파일 결과 모델 모음. **Git 에는 모델 바이너리 미포함** (`.gitignore` 제외).
필요 시 별도 storage (NAS / S3 / 사내 공유 등) 에서 받아 이 디렉터리에 배치.

## 파일

| 파일 | 크기 | 용도 | 비고 |
|---|---:|---|---|
| `action_resnet_mt.onnx` | 43 MB | PyTorch export | legacy TracedExport (Hailo 호환) |
| `action_resnet_mt.hef` | 7 MB | Hailo-8 (26 TOPS) | etri-board ×4 NPU |
| `action_resnet_mt_h8l.hef` | 14 MB | Hailo-8L (13 TOPS) | RPi5 AI Kit |
| `yolov8s_pose.hef` | 11 MB | Pose 검출 (Hailo-8) | YOLOv8s 640×640 |
| `yolov8s_pose_h8l.hef` | 22 MB | Pose 검출 (Hailo-8L) | 동일 모델, 다른 architecture |
| `yolov8m_pose.hef` | 31 MB | Pose 검출 (Hailo-8) | YOLOv8m (더 정확) |
| `yolov8n-pose.onnx` | 13 MB | CPU fallback | YOLOv8n 320×320 |
| `calibration_set.npy` | 85 MB | Hailo INT8 calib | 1500 sample × [60, 25, 7] |

## 모델 메타데이터

### action_resnet_mt

| 항목 | 값 |
|---|---|
| Backbone | ResNet18 (ImageNet pretrained) |
| 입력 | `[1, 7, 60, 25]` — pseudo-image NCHW |
| 출력 | 5 head: action_upper(6) / action_lower(10) / pose(9) / hand(3) / foot(3) |
| 파라미터 | 11.2 M |
| 학습 데이터 | 13878 clip (안전모션 v22) |
| 학습 환경 | AICA GPU, 100 epoch, SGD + cosine LR |
| Test 정확도 | PyTorch 95.78% / HEF 95.73% |

### yolov8 pose 변종

Pose 모델은 Hailo Model Zoo (v2.18.0) 공식 컴파일 HEF 사용.
다운로드: https://github.com/hailo-ai/hailo_model_zoo

| 모델 | COCO AP | 우리 측정 FPS (단일 NPU) |
|---|---:|---:|
| yolov8m_pose | 65.0 | etri-board 22 |
| yolov8s_pose | 60.0 | Pi5 (h8l) 20 |
| yolov8n-pose (ONNX, CPU) | 50.4 | Pi5 CPU 8 |

## 모델 받는 법

이 디렉터리는 `.gitignore` 로 바이너리 제외돼 있어요. 별도 받아서 여기에 배치:

```bash
# 예: 사내 NAS 또는 공유 storage 에서
scp <storage>/action_resnet_mt.onnx ./models/
scp <storage>/action_resnet_mt_h8l.hef ./models/
# ...
```

또는 학습부터 직접 생성:
1. `training/scripts/train_mt.py` → `best_mt.pth`
2. `training/scripts/export_onnx_mt.py` → `action_resnet_mt.onnx`
3. `compile/recompile_mt.sh` → `*.hef`

## 모델 재컴파일

ONNX 가 있으면 HEF 재생성 가능 — [`compile/README.md`](../compile/README.md) 참고.

학습 weight 변경 시 순서:
1. `training/scripts/train_mt.py` 로 학습 → `best_mt.pth`
2. `training/scripts/export_onnx_mt.py` → `models/action_resnet_mt.onnx`
3. `compile/recompile_mt.sh` (or `_8l.sh`) → `models/*.hef`
