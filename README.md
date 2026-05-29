# PSP-Net MB4: Body-Part Partitioned Convolution for INT8-Quantized Skeleton Action Recognition on Edge NPUs

> **Practical baseline for INT8 skeleton action recognition on commodity edge NPUs**
> *(Hailo-8 / Hailo-8L / Raspberry Pi 5)*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hailo-8 / 8L](https://img.shields.io/badge/Hailo-8%20%2F%208L-FF9900.svg)](https://hailo.ai/)

**English README** | [한국어 README](README_KO.md)

---

## TL;DR

PSP-Net MB4 reformulates skeleton pseudo-image, body-part modeling, and multi-stream descriptors using **standard 2D convolution operators only** (Conv2D, BatchNorm, ReLU, Sigmoid, GAP, Linear), enabling **deterministic INT8 deployment** and **real-time inference (~200 FPS / 31 FPS end-to-end)** on commodity edge NPUs.

A **single 4-branch architecture** is provided in **two input modes** so it can be dropped into either camera setup without changing the network:

- **MB4-3D** — 3D skeleton input (x, y, z). For **depth cameras** (Kinect / Azure / RealSense) where joint depth is available.
- **MB4-2D** — 2D skeleton input (x, y only). For **standard RGB cameras** driven by a 2D pose estimator (e.g. YOLO-Pose), where no depth is available.

Both modes share the same backbone (4 branches: joint / joint-motion / bone / bone-motion → 1×1 fusion → single head); only the input channel count differs (24ch vs 16ch).

## Key Results (NTU RGB+D 60, Cross-Subject)

| | **MB4-3D** (3D input) | **MB4-2D** (2D input) |
|---|---:|---:|
| Input channels | 24 (x, y, z) | 16 (x, y) |
| Parameters | 1.42 M | 1.38 M |
| Target camera | Depth (Kinect / Azure) | RGB + 2D pose estimator |
| **NTU60 CS — FP32** | **86.29 %** (3-seed 86.37 ± 0.06) | **84.36 %** |
| **NTU60 CS — INT8 (QAT)** | **85.50 %** (−1.35 pp) | **82.17 %** (−2.63 pp) |
| NTU60 CS — INT8 (PTQ) | 85.20 % (−1.65 pp) | 80.86 % (−3.94 pp) |
| NTU60 CV — FP32 | 91.16 % | — |
| NTU120 CSub — FP32 (tuned) | 79.74 % | — |
| Hailo-8L FPS (Pi5) | ~200 | ~200 |
| Pi5 end-to-end (RTSP→pose→action) | 31.2 FPS | 31.2 FPS |

> **INT8 quantization.** Both models deploy in **INT8 (8-bit, fast)**. Quantization-Aware finetune (QAT, Hailo `optimization_level=4`) recovers accuracy over plain PTQ — strongly for MB4-2D (+1.31 pp) where the absence of the z axis raises per-channel information density and quantization sensitivity. See [docs/RESULTS.md](docs/RESULTS.md).

Paper (Korean) → [paper/PAPER_KO.md](paper/PAPER_KO.md)

## Positioning

PSP-Net MB4 is **not a replacement for GCN / Transformer / 3D-CNN SOTA accuracy.** It is a **practical baseline** for the NPU-compatible 2D CNN category: trading away dynamic graphs, attention, and 3D convolutions for **deterministic INT8 deployment and real-time throughput** on commodity edge NPUs.

- Within the NPU-compatible 2D CNN category, MB4-3D updates a 7-year-stagnant baseline (TSSI 2019, 79.2 %) by **+7.1 pp** on NTU60 CS.
- The accuracy gap of ≈ −6 to −7 pp versus academic SOTA (CTR-GCN 92.4 %, SkateFormer 92.6 %) is an inherent trade-off of NPU operator constraints — in exchange for 5 W / 200 FPS edge deployment.

## Choosing a Mode

| Your camera / pose source | Use |
|---|---|
| Depth camera (Kinect v2, Azure Kinect, RealSense) — 3D joints available | **MB4-3D** (higher accuracy) |
| RGB camera + 2D pose estimator (YOLO-Pose, RTMPose, MoveNet) — no depth | **MB4-2D** |

## Repository Layout

```
psp-net/
├── paper/                  Paper (Korean) + figures
├── psp_net/                Model + dataset code
│   ├── models/
│   │   ├── psp_blocks.py   Shared building blocks (BodyPartConv, STDecoupled, SE, StreamBranch)
│   │   ├── psp_mb4.py      MB4-3D  (24ch input)
│   │   └── psp_mb4_2d.py   MB4-2D  (16ch input)
│   └── dataset/            NTU60 / NTU120 loader (use_3d toggles 3D/2D)
├── configs/                ntu60_psp_mb4.yaml (3D), ntu60_psp_mb4_2d.yaml (2D), ntu120_psp_mb4_tuned.yaml
├── scripts/                preprocess / train / eval / export ONNX
├── deploy/                 Hailo HEF compile (PTQ + QAT) + Pi5 e2e demo
├── checkpoints/            Pretrained .pth / .hef (see GitHub Releases)
└── docs/                   INSTALL / REPRODUCE / DEPLOY / RESULTS
```

## Quick Start

### 1. Environment

```bash
git clone https://github.com/ldgsmhrd/hailo_action.git
cd hailo_action
pip install -r requirements.txt
```

Full setup → [docs/INSTALL.md](docs/INSTALL.md)

### 2. Data (NTU60)

Download `nturgbd_skeletons_s001_to_s017.zip` from the [official NTU RGB+D site](https://rose1.ntu.edu.sg/dataset/actionRecognition/), then:

```bash
unzip nturgbd_skeletons_s001_to_s017.zip -d data/ntu60/
python scripts/preprocess_ntu.py --raw-dir data/ntu60/nturgb+d_skeletons --out-dir data/ntu60/npy
```

For NTU120, additionally download `nturgbd_skeletons_s018_to_s032.zip` into the same directory.

### 3. Training

```bash
# MB4-3D — 3D input (depth cameras)
python scripts/train.py --config configs/ntu60_psp_mb4.yaml

# MB4-2D — 2D input (RGB cameras + 2D pose estimator)
python scripts/train.py --config configs/ntu60_psp_mb4_2d.yaml

# NTU120 (MB4-3D, tuned)
python scripts/train.py --config configs/ntu120_psp_mb4_tuned.yaml
```

The only difference between the two modes is `use_3d: true|false` in the config (24ch vs 16ch input).

### 4. ONNX Export

```bash
python scripts/export_onnx.py --ckpt checkpoints/best_mb4_xsub.pth      --out psp_mb4.onnx    --model mb4    --in-channels 24
python scripts/export_onnx.py --ckpt checkpoints/best_mb4_2d_xsub.pth   --out psp_mb4_2d.onnx --model mb4_2d --in-channels 16
```

### 5. Hailo HEF Compilation (INT8)

```bash
# Inside the Hailo Dataflow Compiler docker container.
# QAT (optimization_level=4 + finetune) — requires GPU passthrough (--gpus), recovers accuracy.
bash deploy/compile_hef.sh hailo8   mb4       # MB4-3D, Hailo-8
bash deploy/compile_hef.sh hailo8l  mb4       # MB4-3D, Hailo-8L (Pi5)
bash deploy/compile_hef.sh hailo8l  mb4_2d    # MB4-2D, Hailo-8L (Pi5)
```

ALLS scripts: `deploy/psp_mb4_qat.alls` (QAT, recommended) and `deploy/psp_mb4_ptq.alls` (plain PTQ, no GPU needed). Full guide → [docs/DEPLOY.md](docs/DEPLOY.md)

### 6. Pi5 End-to-End Real-time Demo

```bash
# On Raspberry Pi 5 + Hailo-8L M.2 module
python deploy/pi5_e2e_demo.py \
    --rtsp rtsp://camera-ip/stream \
    --pose-hef yolov8s_pose_h8l.hef \
    --action-hef psp_mb4_2d_h8l.hef     # MB4-2D pairs naturally with a 2D pose estimator
```

## Pretrained Checkpoints

Available from the GitHub Releases page of this repository:

| File | Size | Accuracy |
|---|---|---|
| `best_mb4_xsub.pth` (MB4-3D, NTU60) | ~5.8 MB | 86.29 % FP32 |
| `best_mb4_2d_xsub.pth` (MB4-2D, NTU60) | ~5.4 MB | 84.36 % FP32 |
| `best_ntu120_mb4_tuned.pth` | ~12 MB | 79.74 % FP32 |
| `psp_mb4_h8.hef` / `_h8l.hef` (MB4-3D, QAT) | ~4–6 MB | 85.50 % INT8 |
| `psp_mb4_2d_h8.hef` / `_h8l.hef` (MB4-2D, QAT) | ~4–6 MB | 82.17 % INT8 |

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

- Issues: [GitHub Issues](https://github.com/ldgsmhrd/hailo_action/issues)
- Email: ldg@safemotion.kr
