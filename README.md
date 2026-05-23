# PSP-Net: Body-Part Partitioned Convolution for INT8-Quantized Skeleton Action Recognition on Edge NPUs

> **Practical baseline for INT8 skeleton action recognition on commodity edge NPUs**
> *(Hailo-8 / Hailo-8L / Raspberry Pi 5)*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hailo-8 / 8L](https://img.shields.io/badge/Hailo-8%20%2F%208L-FF9900.svg)](https://hailo.ai/)

**English README** | [한국어 README](README_KO.md)

---

## TL;DR

> PSP-Net reformulates skeleton pseudo-image, body-part modeling, and multi-stream descriptors using **standard 2D convolution operators only** (Conv2D, BatchNorm, ReLU, Sigmoid, GAP, Linear), enabling **deterministic INT8 deployment** and **31.2 FPS real-time inference** on commodity edge NPUs (Hailo-8 / Hailo-8L) for skeleton-based action recognition.

## Key Results

| Metric | PSP-Net (MB-3D) | PSP-Net (MB4) |
|---|---:|---:|
| Params | 1.50 M | **1.42 M** |
| NTU60 CS (3-seed mean) | 84.97 ± 0.22 % | **86.37 ± 0.06 %** |
| NTU60 CS + TTA | — | **86.76 ± 0.22 %** |
| NTU60 CV | 90.42 % | **91.16 %** |
| NTU120 CSub (tuned) | 79.63 % | **79.74 %** |
| Hailo-8 INT8 (NTU60 CS) | 84.71 % (−0.56 pp) | 84.37 % (−1.92 pp) |
| Hailo-8L INT8 (Pi5) | 84.81 % (−0.46 pp) | 84.35 % (−1.94 pp) |
| Hailo-8 FPS (batch=1) | **3,965** | 388 |
| Hailo-8L FPS (Pi5) | **348** | 200 |
| **Pi5 end-to-end (RTSP→pose→action)** | — | **31.2 FPS** |

Full results → [docs/RESULTS.md](docs/RESULTS.md)
Paper (Korean) → [paper/PAPER_KO.md](paper/PAPER_KO.md)

## Positioning

PSP-Net is **not a replacement for GCN / Transformer / 3D-CNN SOTA accuracy.** It is positioned as a **practical baseline** for the NPU-compatible 2D CNN category: trading away the expressiveness of dynamic graphs, attention, and 3D convolutions in exchange for **deterministic INT8 deployment and real-time throughput** on commodity edge NPUs.

- Absolute accuracy gap of approximately −6 to −7 pp (NTU60) and −9 to −11 pp (NTU120) versus academic SOTA (CTR-GCN 92.4%, SkateFormer 93.5%, BlockGCN 93.1%) is a trade-off inherent to NPU operator constraints.
- **Within the NPU-compatible 2D CNN category**, PSP-Net (MB4) updates a 7-year-stagnant baseline (TSSI 2019, 79.2%) by +7.17 pp on NTU60 CS.

## Repository Layout

```
psp-net/
├── paper/                  Paper (Korean) + figures
├── psp_net/                Model + dataset code
│   ├── models/             PSP-Net, MB-3D, MB4
│   └── dataset/            NTU60 / NTU120 loader
├── configs/                YAML configs (NTU60 MB4, MB-3D, NTU120 tuned)
├── scripts/                preprocess / train / eval / export ONNX
├── deploy/                 Hailo HEF compile + Pi5 e2e demo
├── checkpoints/            Pretrained .pth / .hef (see GitHub Releases)
└── docs/                   INSTALL / REPRODUCE / DEPLOY / RESULTS
```

## Quick Start

### 1. Environment

```bash
git clone https://github.com/ldg/psp-net.git
cd psp-net
pip install -r requirements.txt
```

Full setup → [docs/INSTALL.md](docs/INSTALL.md)

### 2. Data (NTU60)

Download `nturgbd_skeletons_s001_to_s017.zip` from the [official NTU RGB+D site](https://rose1.ntu.edu.sg/dataset/actionRecognition/), then:

```bash
unzip nturgbd_skeletons_s001_to_s017.zip -d data/ntu60/
python scripts/preprocess_ntu.py --raw-dir data/ntu60/nturgb+d_skeletons --out-dir data/ntu60/npy
```

For NTU120, additionally download `nturgbd_skeletons_s018_to_s032.zip` and extract into the same directory.

### 3. Training

```bash
# NTU60 PSP-Net (MB4)
python scripts/train.py --config configs/ntu60_psp_mb4.yaml

# NTU60 PSP-Net (MB-3D) — quantization robust, recommended for NPU deployment
python scripts/train.py --config configs/ntu60_psp_mb_3d.yaml

# NTU120 PSP-Net (MB4) tuned
python scripts/train.py --config configs/ntu120_psp_mb4_tuned.yaml
```

### 4. ONNX Export

```bash
python scripts/export_onnx.py --ckpt checkpoints/best_mb4.pth --out psp_mb4.onnx
```

### 5. Hailo HEF Compilation

```bash
# Inside the Hailo Dataflow Compiler docker container
bash deploy/compile_hef.sh hailo8     # for Hailo-8
bash deploy/compile_hef.sh hailo8l    # for Hailo-8L (Pi5)
```

Full guide → [docs/DEPLOY.md](docs/DEPLOY.md)

### 6. NPU Accuracy Evaluation

```bash
# On a Hailo-8 / 8L board
python deploy/eval_on_npu.py \
    --hef psp_mb4_h8l.hef \
    --test-x data/test_x.npy \
    --test-y data/test_y.npy
```

### 7. Pi5 End-to-End Real-time Demo

```bash
# On Raspberry Pi 5 + Hailo-8L M.2 module
python deploy/pi5_e2e_demo.py \
    --rtsp rtsp://camera-ip/stream \
    --pose-hef yolov8s_pose_h8l.hef \
    --action-hef psp_mb4_h8l.hef
```

## Pretrained Checkpoints

Available from the GitHub Releases page of this repository:

| File | Size | Accuracy |
|---|---|---|
| `best_mb4_xsub.pth` (NTU60) | ~5.8 MB | 86.29 % FP32 |
| `best_mb_3d_xsub.pth` (NTU60) | ~6.1 MB | 85.27 % FP32 |
| `best_ntu120_mb4_tuned.pth` | ~12 MB | 79.74 % FP32 |
| `psp_mb4_h8.hef` (Hailo-8) | ~4.1 MB | 84.37 % INT8 |
| `psp_mb4_h8l.hef` (Hailo-8L) | ~6.2 MB | 84.35 % INT8 |
| `psp_mb_3d_h8.hef` | ~2.6 MB | 84.71 % INT8 |
| `psp_mb_3d_h8l.hef` | ~2.6 MB | 84.81 % INT8 |

See [checkpoints/README.md](checkpoints/README.md) for download instructions.

## Citation

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

Apache License 2.0 — see [LICENSE](LICENSE).

## Contact

- Issues: [GitHub Issues](https://github.com/ldg/psp-net/issues)
- Email: ldg@safemotion.kr
