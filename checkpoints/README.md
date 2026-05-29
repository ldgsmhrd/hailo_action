# Pretrained Checkpoints

Pre-trained PyTorch (`.pth`) 와 Hailo (`.hef`) 체크포인트는 본 repository 의 **GitHub Releases** 에서 별도 다운로드합니다.

## 다운로드 위치

[**GitHub Releases v1.0.0**](https://github.com/ldgsmhrd/hailo_action/releases/tag/v1.0.0)

본 저장소는 **MB4-3D**(3D 입력)와 **MB4-2D**(2D 입력) 두 모델만 제공합니다.

### PyTorch FP32 체크포인트
| 파일 | 모델 | 크기 | 정확도 | Config |
|---|---|---|---|---|
| `best_mb4_xsub.pth` | MB4-3D | ~5.8 MB | 86.29 % (NTU60 CS) | `configs/ntu60_psp_mb4.yaml` |
| `best_mb4_2d_xsub.pth` | MB4-2D | ~5.4 MB | 84.36 % (NTU60 CS) | `configs/ntu60_psp_mb4_2d.yaml` |
| `best_mb4_xview.pth` | MB4-3D | ~5.8 MB | 91.16 % (NTU60 CV) | `--benchmark xview` |
| `best_ntu120_mb4_tuned.pth` | MB4-3D | ~12 MB | 79.74 % (NTU120 CSub) | `configs/ntu120_psp_mb4_tuned.yaml` |

### Hailo HEF 체크포인트 (INT8, QAT)
| 파일 | 모델 | 크기 | 정확도 | 대상 |
|---|---|---|---|---|
| `psp_mb4_h8.hef` | MB4-3D | ~4.1 MB | 85.50 % | Hailo-8 |
| `psp_mb4_h8l.hef` | MB4-3D | ~6.2 MB | 85.50 % | Hailo-8L (Pi5) |
| `psp_mb4_2d_h8.hef` | MB4-2D | ~4.1 MB | 82.17 % | Hailo-8 |
| `psp_mb4_2d_h8l.hef` | MB4-2D | ~6.3 MB | 82.17 % | Hailo-8L (Pi5) |

> HEF 는 모두 **QAT (optimization_level=4 + finetune)** 로 컴파일. PTQ 대비 MB4-3D +0.30 %p, MB4-2D +1.31 %p 회복.

### Calibration / finetune set (HEF 재컴파일 용)
| 파일 | 크기 | 용도 |
|---|---|---|
| `calib_ft4k_mb4_3d.npy` | ~300 MB | MB4-3D QAT finetune (60×67=4,020, [-10,10] clip) |
| `calib_ft4k_mb4_2d.npy` | ~200 MB | MB4-2D QAT finetune (60×67=4,020, [-10,10] clip) |

## 다운로드 예시

```bash
mkdir -p checkpoints

# 단일 파일
wget https://github.com/ldgsmhrd/hailo_action/releases/download/v1.0.0/best_mb4_xsub.pth -P checkpoints/

# 전체 (release tag 기준)
gh release download v1.0.0 --repo ldgsmhrd/hailo_action --dir checkpoints/
```

## 사용 예시

### PyTorch 평가
```bash
python scripts/eval.py \
    --config configs/ntu60_psp_mb4.yaml \
    --ckpt checkpoints/best_mb4_xsub.pth
```

### NPU 평가 (HEF)
```bash
# Pi5 또는 Hailo-8 보드에서
python deploy/eval_on_npu.py \
    --hef checkpoints/psp_mb4_h8l.hef \
    --test-x data/ntu60/test_x.npy \
    --test-y data/ntu60/test_y.npy
```

## SHA256 체크섬

다운로드 무결성 검증:
```bash
sha256sum -c checkpoints/SHA256SUMS.txt
```

(SHA256SUMS.txt 는 release 함께 첨부)

## 라이선스

체크포인트는 Apache License 2.0 (코드와 동일) 하에 배포됩니다. NTU RGB+D 60 / 120 데이터셋 자체는 별도 라이선스 (학술 목적 제한) 를 따르며 본 repository 는 weight 만 배포합니다.

자세한 dataset terms 는 [NTU 공식 사이트](https://rose1.ntu.edu.sg/dataset/actionRecognition/) 참고.
