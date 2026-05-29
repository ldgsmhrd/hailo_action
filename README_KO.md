# PSP-Net MB4: 엣지 NPU INT8 양자화 배포를 위한 부위 분할 합성곱 기반 스켈레톤 행동 인식

> **엣지 NPU(Hailo-8 / Hailo-8L / Raspberry Pi 5)에서 INT8 스켈레톤 행동 인식을 위한 실용 베이스라인**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hailo-8 / 8L](https://img.shields.io/badge/Hailo-8%20%2F%208L-FF9900.svg)](https://hailo.ai/)

[English README](README.md) | **한국어 README**

---

## 한 줄 요약

PSP-Net MB4 는 스켈레톤 의사 이미지·부위 분할·다중 스트림 디스크립터를 **표준 2D 합성곱 연산자만으로**(Conv2D, BatchNorm, ReLU, Sigmoid, GAP, Linear) 재구성하여, 엣지 NPU 에서 **결정적 INT8 배포**와 **실시간 추론(~200 FPS / end-to-end 31 FPS)**을 가능하게 합니다.

**단일 4-branch 아키텍처**를 **두 가지 입력 모드**로 제공하여, 네트워크 변경 없이 어떤 카메라 환경에도 그대로 적용할 수 있습니다.

- **MB4-3D** — 3D 스켈레톤 입력 (x, y, z). 관절 깊이를 얻을 수 있는 **depth 카메라**(Kinect / Azure / RealSense) 용.
- **MB4-2D** — 2D 스켈레톤 입력 (x, y). 깊이 정보가 없는 **일반 RGB 카메라** + 2D 포즈 추정기(YOLO-Pose 등) 용.

두 모드는 동일한 backbone(4 branch: joint / joint-motion / bone / bone-motion → 1×1 fusion → 단일 head)을 공유하며, 입력 채널 수만 다릅니다(24ch vs 16ch).

## 주요 결과 (NTU RGB+D 60, Cross-Subject)

| | **MB4-3D** (3D 입력) | **MB4-2D** (2D 입력) |
|---|---:|---:|
| 입력 채널 | 24 (x, y, z) | 16 (x, y) |
| 파라미터 | 1.42 M | 1.38 M |
| 대상 카메라 | Depth (Kinect / Azure) | RGB + 2D 포즈 추정기 |
| **NTU60 CS — FP32** | **86.29 %** (3-seed 86.37 ± 0.06) | **84.36 %** |
| **NTU60 CS — INT8 (QAT)** | **85.50 %** (−1.35 %p) | **82.17 %** (−2.63 %p) |
| NTU60 CS — INT8 (PTQ) | 85.20 % (−1.65 %p) | 80.86 % (−3.94 %p) |
| NTU60 CV — FP32 | 91.16 % | — |
| NTU120 CSub — FP32 (tuned) | 79.74 % | — |
| Hailo-8L FPS (Pi5) | ~200 | ~200 |
| Pi5 end-to-end (RTSP→포즈→행동) | 31.2 FPS | 31.2 FPS |

> **INT8 양자화.** 두 모델 모두 **INT8(8-bit, 빠름)**로 배포됩니다. 양자화 인식 finetune(QAT, Hailo `optimization_level=4`)이 기본 PTQ 대비 정확도를 회복하며, 특히 **MB4-2D 에서 효과가 큽니다(+1.31 %p)** — z 축이 없어 채널별 정보 밀도와 양자화 민감도가 높기 때문입니다. 자세한 내용 → [docs/RESULTS.md](docs/RESULTS.md)

논문(한국어) → [paper/PAPER_KO.md](paper/PAPER_KO.md)

## 포지셔닝

PSP-Net MB4 는 **GCN / Transformer / 3D-CNN 의 SOTA 정확도를 대체하지 않습니다.** NPU 호환 2D CNN 카테고리의 **실용 베이스라인**입니다 — 동적 그래프·어텐션·3D 합성곱의 표현력을 양보하는 대신, 엣지 NPU 에서 **결정적 INT8 배포와 실시간 처리**를 확보합니다.

- NPU 호환 2D CNN 카테고리에서, MB4-3D 는 7년간 정체되어 있던 베이스라인(TSSI 2019, 79.2 %)을 NTU60 CS 기준 **+7.1 %p** 갱신했습니다.
- 학계 SOTA(CTR-GCN 92.4 %, SkateFormer 92.6 %) 대비 약 −6 ~ −7 %p 의 정확도 격차는 NPU 연산자 제약의 본질적 trade-off 이며, 그 대가로 5 W / 200 FPS 엣지 배포를 얻습니다.

## 모드 선택

| 카메라 / 포즈 소스 | 사용 모델 |
|---|---|
| Depth 카메라 (Kinect v2, Azure Kinect, RealSense) — 3D 관절 제공 | **MB4-3D** (정확도 우위) |
| RGB 카메라 + 2D 포즈 추정기 (YOLO-Pose, RTMPose, MoveNet) — 깊이 없음 | **MB4-2D** |

## 저장소 구조

```
psp-net/
├── paper/                  논문(한국어) + figures
├── psp_net/                모델 + 데이터셋 코드
│   ├── models/
│   │   ├── psp_blocks.py   공용 building block (BodyPartConv, STDecoupled, SE, StreamBranch)
│   │   ├── psp_mb4.py      MB4-3D  (24ch 입력)
│   │   └── psp_mb4_2d.py   MB4-2D  (16ch 입력)
│   └── dataset/            NTU60 / NTU120 로더 (use_3d 로 3D/2D 전환)
├── configs/                ntu60_psp_mb4.yaml (3D), ntu60_psp_mb4_2d.yaml (2D), ntu120_psp_mb4_tuned.yaml
├── scripts/                preprocess / train / eval / export ONNX
├── deploy/                 Hailo HEF 컴파일 (PTQ + QAT) + Pi5 e2e demo
├── checkpoints/            사전학습 .pth / .hef (GitHub Releases 참조)
└── docs/                   INSTALL / REPRODUCE / DEPLOY / RESULTS
```

## 빠른 시작

### 1. 환경

```bash
git clone https://github.com/ldgsmhrd/hailo_action.git
cd hailo_action
pip install -r requirements.txt
```

전체 설정 → [docs/INSTALL.md](docs/INSTALL.md)

### 2. 데이터 (NTU60)

[NTU RGB+D 공식 사이트](https://rose1.ntu.edu.sg/dataset/actionRecognition/)에서 `nturgbd_skeletons_s001_to_s017.zip` 다운로드 후:

```bash
unzip nturgbd_skeletons_s001_to_s017.zip -d data/ntu60/
python scripts/preprocess_ntu.py --raw-dir data/ntu60/nturgb+d_skeletons --out-dir data/ntu60/npy
```

NTU120 은 `nturgbd_skeletons_s018_to_s032.zip` 을 같은 디렉토리에 추가 압축 해제.

### 3. 학습

```bash
# MB4-3D — 3D 입력 (depth 카메라)
python scripts/train.py --config configs/ntu60_psp_mb4.yaml

# MB4-2D — 2D 입력 (RGB 카메라 + 2D 포즈 추정기)
python scripts/train.py --config configs/ntu60_psp_mb4_2d.yaml

# NTU120 (MB4-3D, tuned)
python scripts/train.py --config configs/ntu120_psp_mb4_tuned.yaml
```

두 모드의 유일한 차이는 config 의 `use_3d: true|false` (24ch vs 16ch 입력) 뿐입니다.

### 4. ONNX Export

```bash
python scripts/export_onnx.py --ckpt checkpoints/best_mb4_xsub.pth      --out psp_mb4.onnx    --model mb4    --in-channels 24
python scripts/export_onnx.py --ckpt checkpoints/best_mb4_2d_xsub.pth   --out psp_mb4_2d.onnx --model mb4_2d --in-channels 16
```

### 5. Hailo HEF 컴파일 (INT8)

```bash
# Hailo Dataflow Compiler docker 컨테이너 안에서.
# QAT (optimization_level=4 + finetune) — GPU passthrough(--gpus) 필요, 정확도 회복.
bash deploy/compile_hef.sh hailo8   mb4       # MB4-3D, Hailo-8
bash deploy/compile_hef.sh hailo8l  mb4       # MB4-3D, Hailo-8L (Pi5)
bash deploy/compile_hef.sh hailo8l  mb4_2d    # MB4-2D, Hailo-8L (Pi5)
```

ALLS: `deploy/psp_mb4_qat.alls` (QAT, 권장) / `deploy/psp_mb4_ptq.alls` (기본 PTQ, GPU 불필요). 전체 가이드 → [docs/DEPLOY.md](docs/DEPLOY.md)

### 6. Pi5 End-to-End 실시간 데모

```bash
# Raspberry Pi 5 + Hailo-8L M.2 모듈에서
python deploy/pi5_e2e_demo.py \
    --rtsp rtsp://camera-ip/stream \
    --pose-hef yolov8s_pose_h8l.hef \
    --action-hef psp_mb4_2d_h8l.hef     # MB4-2D 는 2D 포즈 추정기와 자연스럽게 연결
```

## 사전학습 체크포인트

본 저장소 GitHub Releases 페이지에서 제공:

| 파일 | 크기 | 정확도 |
|---|---|---|
| `best_mb4_xsub.pth` (MB4-3D, NTU60) | ~5.8 MB | 86.29 % FP32 |
| `best_mb4_2d_xsub.pth` (MB4-2D, NTU60) | ~5.4 MB | 84.36 % FP32 |
| `best_ntu120_mb4_tuned.pth` | ~12 MB | 79.74 % FP32 |
| `psp_mb4_h8.hef` / `_h8l.hef` (MB4-3D, QAT) | ~4–6 MB | 85.50 % INT8 |
| `psp_mb4_2d_h8.hef` / `_h8l.hef` (MB4-2D, QAT) | ~4–6 MB | 82.17 % INT8 |

다운로드 방법 → [checkpoints/README.md](checkpoints/README.md)

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

## 라이선스

Apache License 2.0 — [LICENSE](LICENSE) 참조.

## 문의

- Issues: [GitHub Issues](https://github.com/ldgsmhrd/hailo_action/issues)
- Email: ldg@safemotion.kr
