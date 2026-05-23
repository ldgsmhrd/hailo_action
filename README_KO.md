# PSP-Net: 엣지 NPU 정수 양자화 배포를 위한 부위 분할 합성곱 기반 스켈레톤 행동 인식

> **Practical baseline for INT8 skeleton action recognition on commodity edge NPUs**
> *(Hailo-8 / Hailo-8L / Raspberry Pi 5)*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hailo-8 / 8L](https://img.shields.io/badge/Hailo-8%20%2F%208L-FF9900.svg)](https://hailo.ai/)

[English README](README.md) | **한국어 README**

---

## 한 줄 소개

> 스켈레톤 의사 이미지 (pseudo-image), 신체 부위 (body-part) 모델링, 다중 스트림 (multi-stream) 디스크립터를 **표준 2D Conv 연산자만으로 재구성** 하여 commodity edge NPU (Hailo-8 / Hailo-8L) 에서 **31.2 FPS 실시간 처리** 와 **deterministic INT8 deployment** 를 확보한 스켈레톤 행동 인식 모델.

## 핵심 결과

| 지표 | PSP-Net (MB-3D) | PSP-Net (MB4) |
|---|---:|---:|
| Params | 1.50 M | **1.42 M** |
| NTU60 CS (3-seed mean) | 84.97 ± 0.22 % | **86.37 ± 0.06 %** |
| NTU60 CS + TTA | — | **86.76 ± 0.22 %** |
| NTU60 CV | 90.42 % | **91.16 %** |
| NTU120 CSub (tuned) | 79.63 % | **79.74 %** |
| Hailo-8 INT8 (NTU60 CS) | 84.71 % (−0.56 %p) | 84.37 % (−1.92 %p) |
| Hailo-8L INT8 (NTU60 CS, Pi5) | 84.81 % (−0.46 %p) | 84.35 % (−1.94 %p) |
| Hailo-8 FPS (batch=1) | **3,965** | 388 |
| Hailo-8L FPS (Pi5) | **348** | 200 |
| **Pi5 e2e (RTSP→pose→action)** | — | **31.2 FPS** |

자세한 결과 → [docs/RESULTS.md](docs/RESULTS.md)
논문 전문 → [paper/PAPER_KO.md](paper/PAPER_KO.md)

## 본 논문의 위치

PSP-Net 은 **GCN / Transformer / 3D-CNN SOTA 의 정확도를 대체하는 모델이 아닙니다.** 본 모델의 위치는 dynamic graph, attention, 3D 합성곱의 표현력을 일부 포기하는 대신, commodity edge NPU 에서 **deterministic INT8 deployment 와 real-time throughput** 을 확보하는 **NPU-compatible 2D CNN 카테고리의 practical baseline** 입니다.

- 정확도 SOTA (CTR-GCN 92.4%, SkateFormer 93.5%, BlockGCN 93.1%) 대비 약 −6~7 %p (NTU60), −9~11 %p (NTU120) 의 정확도 격차는 NPU 호환성 제약에서 비롯된 trade-off 입니다.
- **NPU 호환 2D CNN 카테고리 내** 에서는 TSSI (2019) 79.2 % 대비 +7.17 %p 향상 (NTU60 CS) 이며, 약 7 년간 정체되어 있던 카테고리의 결과를 갱신합니다.

## Repository 구성

```
psp-net/
├── paper/                  논문 (한국어) + figures
├── psp_net/                모델 + dataset 코드
│   ├── models/             PSP-Net, MB-3D, MB4
│   └── dataset/            NTU60 / NTU120 loader
├── configs/                yaml config (NTU60 MB4, MB-3D, NTU120 tuned)
├── scripts/                preprocess / train / eval / export ONNX
├── deploy/                 Hailo HEF 컴파일 + Pi5 e2e 데모
├── checkpoints/            pretrained .pth / .hef (GitHub Releases 별도 다운로드)
└── docs/                   INSTALL / REPRODUCE / DEPLOY / RESULTS
```

## 빠른 시작 (Quick Start)

### 1. 환경 설정

```bash
git clone https://github.com/ldg/psp-net.git
cd psp-net
pip install -r requirements.txt
```

자세한 안내 → [docs/INSTALL.md](docs/INSTALL.md)

### 2. 데이터 준비 (NTU60)

NTU RGB+D 60 의 `nturgbd_skeletons_s001_to_s017.zip` 을 [공식 사이트](https://rose1.ntu.edu.sg/dataset/actionRecognition/) 에서 다운로드한 후:

```bash
unzip nturgbd_skeletons_s001_to_s017.zip -d data/ntu60/
python scripts/preprocess_ntu.py --raw-dir data/ntu60/nturgb+d_skeletons --out-dir data/ntu60/npy
```

NTU120 은 추가로 `nturgbd_skeletons_s018_to_s032.zip` 도 다운로드 후 같은 위치에 풀어줍니다.

### 3. 학습

```bash
# NTU60 PSP-Net (MB4)
python scripts/train.py --config configs/ntu60_psp_mb4.yaml

# NTU60 PSP-Net (MB-3D)  — 양자화 robust, NPU 배포 권장
python scripts/train.py --config configs/ntu60_psp_mb_3d.yaml

# NTU120 PSP-Net (MB4) tuned
python scripts/train.py --config configs/ntu120_psp_mb4_tuned.yaml
```

### 4. ONNX export

```bash
python scripts/export_onnx.py --ckpt checkpoints/best_mb4.pth --out psp_mb4.onnx
```

### 5. Hailo HEF 컴파일

```bash
# Hailo Dataflow Compiler docker 안에서
bash deploy/compile_hef.sh hailo8     # Hailo-8 용
bash deploy/compile_hef.sh hailo8l    # Hailo-8L (Pi5) 용
```

자세한 안내 → [docs/DEPLOY.md](docs/DEPLOY.md)

### 6. NPU 위에서 정확도 평가

```bash
# Hailo-8 / 8L 보드 위에서
python deploy/eval_on_npu.py \
    --hef psp_mb4_h8l.hef \
    --test-x data/test_x.npy \
    --test-y data/test_y.npy
```

### 7. Pi5 end-to-end 실시간 데모

```bash
# Raspberry Pi 5 + Hailo-8L M.2 위에서
python deploy/pi5_e2e_demo.py \
    --rtsp rtsp://camera-ip/stream \
    --pose-hef yolov8s_pose_h8l.hef \
    --action-hef psp_mb4_h8l.hef
```

## Pretrained Checkpoints

이 repository 의 GitHub Releases 에서 다운로드:

| 파일 | 크기 | 정확도 |
|---|---|---|
| `best_mb4_xsub.pth` (NTU60) | ~5.8 MB | 86.29 % FP32 |
| `best_mb_3d_xsub.pth` (NTU60) | ~6.1 MB | 85.27 % FP32 |
| `best_ntu120_mb4_tuned.pth` | ~12 MB | 79.74 % FP32 |
| `psp_mb4_h8.hef` (Hailo-8) | ~4.1 MB | 84.37 % INT8 |
| `psp_mb4_h8l.hef` (Hailo-8L) | ~6.2 MB | 84.35 % INT8 |
| `psp_mb_3d_h8.hef` | ~2.6 MB | 84.71 % INT8 |
| `psp_mb_3d_h8l.hef` | ~2.6 MB | 84.81 % INT8 |

자세한 안내 → [checkpoints/README.md](checkpoints/README.md)

## 인용

```bibtex
@article{lee2026pspnet,
  title={PSP-Net: Body-Part Partitioned Convolution for INT8-Quantized
         Skeleton Action Recognition on Edge NPUs},
  author={Lee, Donggyun},
  journal={Journal of Real-Time Image Processing},
  year={2026},
  publisher={Springer}
}
```

## License

Apache License 2.0 — 자세한 내용은 [LICENSE](LICENSE) 참고.

## 문의

- Issues: [GitHub Issues](https://github.com/ldg/psp-net/issues)
- Email: ldg@safemotion.kr
