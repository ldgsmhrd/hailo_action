# Evaluation

학습된 모델 (PyTorch) 와 컴파일된 HEF (NPU) 의 정확도 비교 + 클래스별 분석.

## 파일

| 파일 | 환경 | 용도 |
|---|---|---|
| `eval_mt_confusion.py` | AICA GPU | PyTorch 모델 confusion matrix + 헷갈린 top-3 |
| `hef_test_eval.py` | 보드 (Hailo NPU) | HEF batch inference + accuracy gap |

## eval_mt_confusion.py — PyTorch 평가

```bash
cd training
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_mt_confusion
```

출력 예시:
```
[자세] pose  —  9 클래스
idx  한글(영어)            train   val  test  recall  헷갈린 top-3
  0  바닥앉기              3045   704   636   79.4%  허리구부리기(110), 서있기(17), 기타(3)
  1  의자앉기               278    90    60   95.0%  바닥앉기(3)
  ...
```

각 클래스마다 train/val/test 카운트 + per-class recall + 어느 클래스로 잘못 분류됐는지 top-3.

## hef_test_eval.py — HEF 평가

PyTorch test set 을 인코딩해서 보드로 보내 HEF 추론.

### 절차

#### 1. AICA 서버에서 test batch 생성

```python
# 'training/dataset/' 안의 test 분할을 pseudo-image 로 인코딩 → npz
python -c "
import sys; sys.path.insert(0, 'training')
from scripts.dataset_mt import MultiTaskActionDataset
from dataset.encoders import PseudoImageEncoder
import numpy as np

ds = MultiTaskActionDataset(split_root='data/split_mt_v3', split='test', T=60,
                            encoder=PseudoImageEncoder(order='tssi', channels=('pos','velocity','angle')))
xs = np.empty((len(ds), 7, 60, 25), dtype=np.float32)
labels = {k: np.empty(len(ds), dtype=np.int64) for k in ds.labels[0]}
for i in range(len(ds)):
    x, l = ds[i]
    xs[i] = x
    for k in labels: labels[k][i] = l[k].item()
np.savez_compressed('test_batch.npz', x=xs, **labels)
print('saved test_batch.npz')
"
```

#### 2. 보드로 전송 + 실행

```bash
scp test_batch.npz <board>:/share/test_batch.npz
ssh <board>
docker exec arn-npu python3 /app/eval/hef_test_eval.py
```

#### 3. 출력 예시 (실제 결과)

```
============================================================
HEF (INT8 quantized) test results — N=2057
============================================================
  action_upper  : 97.23%  per-class: 0:0.98(1721) 1:0.85(34) ...
  action_lower  : 95.43%  per-class: ...
  pose          : 86.78%  ...
  hand          : 99.85%  ...
  foot          : 99.37%  ...
============================================================
  평균          : 95.73%

=== quantization gap (HEF - PyTorch) ===
  action_upper  : PyTorch 97.28% → HEF 97.23% (gap -0.05%)
  action_lower  : PyTorch 95.43% → HEF 95.43% (gap +0.00%)
  pose          : PyTorch 86.73% → HEF 86.78% (gap +0.05%)
  hand          : PyTorch 99.85% → HEF 99.85% (gap +0.00%)
  foot          : PyTorch 99.61% → HEF 99.37% (gap -0.24%)
```

## 결과 요약

| Head | PyTorch | HEF | Gap |
|---|---:|---:|---:|
| 상체 action_upper | 97.28% | 97.23% | -0.05% |
| 하체 action_lower | 95.43% | 95.43% | 0.00% |
| 자세 pose | 86.73% | 86.78% | **+0.05%** |
| 손 hand | 99.85% | 99.85% | 0.00% |
| 발 foot | 99.61% | 99.37% | -0.24% |
| **평균** | **95.78%** | **95.73%** | **-0.05%** |

INT8 PTQ 양자화 손실 거의 없음. 추론 속도 **639 samples/s** (단일 NPU batch 1).

## 주요 confusion 패턴

1. **바닥앉기 ↔ 허리구부리기** (pose) — 양방향 110+81 — 라벨 모호
2. **서성이기/걷기 ↔ 없음** (action_lower) — 약한 모션 미감지
3. **킥 (recall 69%)** — 데이터 47개로 부족
4. **손흔들기 ↔ 손올리기** (action_upper) — 동작 유사

자세한 분석: [`docs/MULTITASK_MODEL_REPORT.md`](../docs/MULTITASK_MODEL_REPORT.md) §5
