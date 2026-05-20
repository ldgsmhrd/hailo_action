# HEF 컴파일

ONNX → Hailo HEF (INT8 양자화) 변환.

## 파일

- `recompile_mt.sh` — Hailo-8 풀버전 (26 TOPS) 용
- `recompile_mt_8l.sh` — Hailo-8L (13 TOPS, RPi5 AI Kit) 용
- 입력: `../models/action_resnet_mt.onnx` (43MB)
- 입력 calib: `../models/calibration_set.npy` (1500 sample, INT8 양자화 보정용)
- 출력: HEF 파일

## 사전 준비 — Hailo Dataflow Compiler Docker

```bash
# Hailo 사이트에서 다음 두 파일 다운로드:
#   - hailo8_ai_sw_suite_2025-10.tar.gz (8.7GB Docker 이미지)
#   - hailort-pcie-driver_X.Y.Z_all.deb

# Docker 이미지 로드
docker load -i hailo8_ai_sw_suite_2025-10.tar.gz

# 확인
docker images | grep hailo8_ai_sw_suite
```

## 컴파일 실행

```bash
# Hailo-8 풀버전 용
cd compile  # 또는 hailo/shared_with_docker 디렉터리
docker run --rm \
  -v $(pwd):/local/work \
  hailo8_ai_sw_suite_2025-10:1 \
  bash -c "cd /local/work && bash recompile_mt.sh"

# Hailo-8L 용 (RPi5 AI Kit)
docker run --rm \
  -v $(pwd):/local/work \
  hailo8_ai_sw_suite_2025-10:1 \
  bash -c "cd /local/work && bash recompile_mt_8l.sh"
```

## 3 단계 (스크립트 내부)

```
1. hailo parser onnx       → action_resnet_mt.har  (중간 표현)
2. hailo optimize          → action_resnet_mt_quantized.har  (INT8 PTQ)
3. hailo compiler          → action_resnet_mt.hef
```

소요 시간 (~CPU only): ~5분
- Parse: 1.4s
- Optimize: ~3분 (calibration 1500 sample)
- Compile: ~10s

## 산출물

| 파일 | 크기 | 대상 |
|---|---|---|
| action_resnet_mt.hef | 7 MB | Hailo-8 (26 TOPS) |
| action_resnet_mt_h8l.hef | 14 MB | Hailo-8L (13 TOPS) |

## 양자화 — 스킵된 최적화

GPU 패스스루 없이 docker 에서 실행 시 optimization_level=0 으로 강등:
- Bias Correction skipped
- Adaround skipped
- QAT Fine-Tuning skipped

ResNet18 + Linear head 같은 단순 구조라 PTQ 기본만으로도 손실 거의 없음 — gap -0.05%.

GPU 활용해서 더 안전마진 확보하려면 docker run 시 `--gpus all` + nvidia-container-toolkit.

## HEF 검증

컴파일 후 보드에서 test set 정확도 비교:

```bash
cd ../eval
python3 hef_test_eval.py   # 보드에서 실행, test_batch.npz 필요
```

PyTorch vs HEF gap < 0.5% 면 성공.

## 보드별 HEF 사용

| 보드 | hw-arch | 사용 HEF |
|---|---|---|
| etri-board (Hailo-8 ×4) | hailo8 | action_resnet_mt.hef |
| RPi5 + Hailo-8L AI Kit | hailo8l | action_resnet_mt_h8l.hef |

`hailortcli fw-control identify` 로 보드 아키텍처 확인 가능.
