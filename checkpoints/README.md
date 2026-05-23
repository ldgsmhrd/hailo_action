# Pretrained Checkpoints

Pre-trained PyTorch (`.pth`) 와 Hailo (`.hef`) 체크포인트는 본 repository 의 **GitHub Releases** 에서 별도 다운로드합니다.

## 다운로드 위치

[**GitHub Releases v1.0.0**](https://github.com/ldg/psp-net/releases/tag/v1.0.0)

## 파일 목록

### PyTorch FP32 체크포인트
| 파일 | 크기 | 정확도 | Config |
|---|---|---|---|
| `best_ntu60_mb4_xsub.pth` | ~5.8 MB | 86.29 % | `configs/ntu60_psp_mb4.yaml` |
| `best_ntu60_mb_3d_xsub.pth` | ~6.1 MB | 85.27 % | `configs/ntu60_psp_mb_3d.yaml` |
| `best_ntu60_mb_2d_xsub.pth` | ~5.8 MB | 83.13 % | (configs/ntu60_psp_mb_2d.yaml) |
| `best_ntu60_mb4_xview.pth` | ~5.8 MB | 91.16 % | `--benchmark xview` |
| `best_ntu60_mb_3d_xview.pth` | ~6.1 MB | 90.42 % | `--benchmark xview` |
| `best_ntu120_mb4_tuned.pth` | ~12 MB | 79.74 % | `configs/ntu120_psp_mb4_tuned.yaml` |
| `best_ntu120_mb_3d_tuned.pth` | ~12 MB | 79.63 % | `configs/ntu120_psp_mb_3d_tuned.yaml` |

### Hailo HEF 체크포인트
| 파일 | 크기 | 정확도 | 대상 |
|---|---|---|---|
| `psp_mb4_h8.hef` | ~4.1 MB | 84.37 % | Hailo-8 (검증 보드) |
| `psp_mb4_h8l.hef` | ~6.2 MB | 84.35 % | Hailo-8L (Pi5) |
| `psp_mb_3d_h8.hef` | ~2.6 MB | 84.71 % | Hailo-8 |
| `psp_mb_3d_h8l.hef` | ~2.6 MB | 84.81 % | Hailo-8L (Pi5) |
| `psp_mb_2d_h8.hef` | ~2.7 MB | 82.39 % | Hailo-8 |
| `psp_mb4_ntu120_h8.hef` | ~4.1 MB | 75.10 % | Hailo-8 (NTU120) |
| `psp_mb_3d_ntu120_h8.hef` | ~2.7 MB | 78.18 % | Hailo-8 (NTU120) |

### Calibration set (HEF 컴파일 재현 용)
| 파일 | 크기 | 용도 |
|---|---|---|
| `calib_ntu60_2k.npy` | ~80 MB | 2,048 표본, [-10, 10] clipping |

## 다운로드 예시

```bash
mkdir -p checkpoints

# 단일 파일
wget https://github.com/ldg/psp-net/releases/download/v1.0.0/best_ntu60_mb4_xsub.pth -P checkpoints/

# 전체 (release tag 기준)
gh release download v1.0.0 --repo ldg/psp-net --dir checkpoints/
```

## 사용 예시

### PyTorch 평가
```bash
python scripts/eval.py \
    --config configs/ntu60_psp_mb4.yaml \
    --ckpt checkpoints/best_ntu60_mb4_xsub.pth
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
