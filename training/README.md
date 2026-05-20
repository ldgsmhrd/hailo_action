# Training — Multi-task ResNet18

PyTorch 학습 파이프라인. AICA GPU 서버 환경에서 실행.

## 파일 구성

```
training/
├── configs/
│   └── aica_multitask_config.py        # 5 head 31 클래스 정의 + 학습 hyperparams
├── dataset/
│   ├── cvt_labelmap_kid.py             # v22 → kid simple 라벨 매핑
│   ├── cvt_labelmap_bus01.py           # v22 → bus01 (참고용)
│   ├── run_gen_npy_mt.py               # clip JSON → NPY (TSSI keypoint)
│   └── encoders/
│       ├── pseudo_image.py             # 17 COCO → 25 TSSI + 7 채널
│       └── skeleton.py                 # 관절 연결 정의
└── scripts/
    ├── model_mt.py                     # MultiTaskActionResNet (ResNet18 + 5 Linear head)
    ├── dataset_mt.py                   # MultiTaskActionDataset (.meta.json 로드)
    ├── train_mt.py                     # 학습 entry (class weight + cosine LR)
    ├── eval_mt.py                      # Test set 평가 (per-head accuracy)
    ├── eval_mt_confusion.py            # 클래스별 confusion + 헷갈린 top-3
    └── export_onnx_mt.py               # ONNX export (legacy TracedExport, Hailo 호환)
```

## 실행 순서

### 1. NPY 생성

```bash
python -m dataset.run_gen_npy_mt
```

- 입력: clip JSON (안전모션 v22 어노테이션)
- 출력: `data/npy_mt_v3/<class>/<name>.npy` + `<name>.meta.json`
- 결과: 13878 NPY (train 9645 / val 2176 / test 2057)

### 2. 학습

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mt
```

- Optimizer: SGD + Nesterov 0.9 + weight_decay 1e-4
- LR: 0.05 → cosine annealing → 0
- 100 epoch (~11분 on V100/A100)
- Loss: head 별 weighted Cross-Entropy + effective-number class weight (β=0.999)
- Best ckpt 저장: `models/best_mt.pth` (avg val accuracy 기준)

### 3. 평가

```bash
python -m scripts.eval_mt              # 헤드별 accuracy + per-class recall
python -m scripts.eval_mt_confusion    # confusion matrix + 헷갈린 top-3
```

### 4. ONNX export

```bash
python -m scripts.export_onnx_mt
```

- `dynamo=False` (legacy TracedExport) — Hailo parser 호환
- opset 11, do_constant_folding
- 출력 5개: action_upper(6) / action_lower(10) / pose(9) / hand(3) / foot(3)
- 산출: `models/action_resnet_mt.onnx` (43 MB single file)

## 학습 결과 예시

```
epoch 099  loss=0.053  val[avg=0.962 lower=0.967 upper=0.971 pose=0.879 hand=1.000 foot=0.993]
학습 완료. best val avg = 0.9621
```

Test 95.78% (자세 86.7% / 손 99.9% / 발 99.6% / 상체 97.3% / 하체 95.4%).
자세한 결과는 [`docs/MULTITASK_MODEL_REPORT.md`](../docs/MULTITASK_MODEL_REPORT.md) §5 참고.

## 의존성

```
torch>=2.0
torchvision
numpy
opencv-python
onnx
```

(GPU 학습 권장 — CPU 면 10배 이상 느림)
