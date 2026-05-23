# REPRODUCE — 학습 + 평가 재현

본 문서는 paper 의 핵심 결과 (NTU60 86.37% / NTU120 79.74% / Pi5 31.2 FPS) 를 재현하는 단계별 안내입니다.

## 1. 데이터셋 준비

### NTU RGB+D 60
1. [NTU RGB+D 공식 사이트](https://rose1.ntu.edu.sg/dataset/actionRecognition/) 에서 `nturgbd_skeletons_s001_to_s017.zip` (약 5.8 GB) 다운로드 (계정 필요).
2. 압축 해제:
   ```bash
   mkdir -p data/ntu60
   unzip nturgbd_skeletons_s001_to_s017.zip -d data/ntu60/
   # → data/ntu60/nturgb+d_skeletons/ 의 56,880 개 .skeleton 파일
   ```
3. NPY 변환 (약 1 분):
   ```bash
   python scripts/preprocess_ntu.py \
       --raw-dir data/ntu60/nturgb+d_skeletons \
       --out-dir data/ntu60/npy
   ```

### NTU RGB+D 120 (선택)
NTU120 은 NTU60 + 추가 데이터:
1. 추가로 `nturgbd_skeletons_s018_to_s032.zip` (약 4.5 GB) 다운로드.
2. 같은 `nturgb+d_skeletons/` 디렉토리에 풀기:
   ```bash
   unzip nturgbd_skeletons_s018_to_s032.zip -d data/ntu60/_tmp120/
   find data/ntu60/_tmp120 -name '*.skeleton' \
       -exec mv -t data/ntu60/nturgb+d_skeletons/ {} +
   # → 114,480 개 .skeleton (NTU60 56,880 + NTU120 추가 57,600)
   ```
3. NPY 재변환 (NTU120 모드):
   ```bash
   python scripts/preprocess_ntu.py \
       --raw-dir data/ntu60/nturgb+d_skeletons \
       --out-dir data/ntu120/npy \
       --dataset ntu120
   ```

## 2. 학습

### NTU60 PSP-Net (MB4) — 핵심 결과 86.29 % (seed 42)
```bash
python scripts/train.py --config configs/ntu60_psp_mb4.yaml
```
- 학습 시간: 약 80 분 (A100 80 GB, 120 epoch)
- 출력: `checkpoints/best_ntu60_mb4_xsub_s42.pth`

### NTU60 PSP-Net (MB-3D)
```bash
python scripts/train.py --config configs/ntu60_psp_mb_3d.yaml
```

### NTU60 Cross-View 변종
```bash
python scripts/train.py --config configs/ntu60_psp_mb4.yaml --benchmark xview
```

### NTU120 PSP-Net (MB4) tuned
```bash
python scripts/train.py --config configs/ntu120_psp_mb4_tuned.yaml
```
- 학습 시간: 약 160 분 (A100, 200 epoch, base_ch=96)

### 3-seed 재현 (NTU60)
Paper Table B 의 3-seed mean 재현:
```bash
for seed in 42 7 17; do
    python scripts/train.py --config configs/ntu60_psp_mb4.yaml --seed $seed
done
# 평균 계산: scripts/eval_3seed.py (TBD)
```

## 3. 평가

### GPU FP32 평가 (학습 직후)
학습 스크립트가 매 epoch test set 평가를 자동 수행합니다. 별도 평가:
```bash
python scripts/eval.py \
    --config configs/ntu60_psp_mb4.yaml \
    --ckpt checkpoints/best_ntu60_mb4_xsub_s42.pth
```

### TTA (anatomical horizontal mirror) 평가
```bash
python scripts/eval.py \
    --config configs/ntu60_psp_mb4.yaml \
    --ckpt checkpoints/best_ntu60_mb4_xsub_s42.pth \
    --tta
```
Expected: TTA mean 86.76 ± 0.22 % (3-seed avg).

## 4. 핵심 결과 매칭 표

본 repository 의 학습 결과가 paper Table B / Table I 와 일치하는지:

| 모델 | Config | Expected | Tolerance |
|---|---|---:|---|
| PSP-Net (MB4) NTU60 CS | `configs/ntu60_psp_mb4.yaml` | 86.37 ± 0.06 % (3-seed) | ± 0.5 %p |
| PSP-Net (MB4) + TTA | + `--tta` flag | 86.76 ± 0.22 % | ± 0.5 %p |
| PSP-Net (MB-3D) NTU60 CS | `configs/ntu60_psp_mb_3d.yaml` | 84.97 ± 0.22 % | ± 0.5 %p |
| PSP-Net (MB4) NTU60 CV | `--benchmark xview` | 91.16 % | ± 0.5 %p |
| PSP-Net (MB4) NTU120 CSub | `configs/ntu120_psp_mb4_tuned.yaml` | 79.74 % | ± 1.0 %p |

> **Note**: 정확한 재현은 random seed, CUDA non-determinism, PyTorch 버전, GPU 모델 등의 변수로 ± 0.3~0.5 %p 정도의 차이가 발생할 수 있습니다.

## 5. NPU 컴파일 + 평가

이 단계는 → [DEPLOY.md](DEPLOY.md) 참고.

## 6. Pre-trained checkpoint 사용

학습을 건너뛰고 paper 의 checkpoint 로 평가만 하려면:
```bash
# GitHub Releases 에서 다운로드
wget https://github.com/ldg/psp-net/releases/download/v1.0.0/best_ntu60_mb4_xsub.pth -O checkpoints/

# 평가
python scripts/eval.py \
    --config configs/ntu60_psp_mb4.yaml \
    --ckpt checkpoints/best_ntu60_mb4_xsub.pth
```
자세한 안내 → [checkpoints/README.md](../checkpoints/README.md)
