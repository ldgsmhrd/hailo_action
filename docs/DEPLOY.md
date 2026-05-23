# DEPLOY — Hailo HEF 컴파일 + 보드 배포

PyTorch checkpoint → ONNX → Hailo HEF → 보드 추론 의 4 단계 파이프라인을 안내합니다.

## 단계 0: 사전 요구사항

1. 학습된 PyTorch checkpoint (`*.pth`) — [REPRODUCE.md](REPRODUCE.md) 또는 release 다운로드
2. Hailo Dataflow Compiler docker — [INSTALL.md](INSTALL.md) 의 컴파일 환경 setup 참고
3. (배포 단계) Hailo-8 보드 또는 Raspberry Pi 5 + Hailo-8L M.2

## 단계 1: PyTorch → ONNX

```bash
python scripts/export_onnx.py \
    --ckpt checkpoints/best_ntu60_mb4_xsub.pth \
    --out psp_mb4.onnx \
    --model mb4 \
    --in-channels 24 \
    --num-classes 60
```

옵션:
- `--model {psp_net, mb_3d, mb4}` — 변종 선택
- `--in-channels` — 6 (2D, 단일 body), 12 (2D, 양 body), 9 (3D, 단일), 18 (3D, 양), 24 (3D + bone motion)
- `--num-classes` — 60 (NTU60), 120 (NTU120)
- `--opset 11` (default) — Hailo parser 호환

**중요**: PyTorch 2.0+ 에서는 `dynamo=False` 가 자동 적용됩니다 (Hailo parser 호환). 새 dynamo export 는 Hailo 가 인식 못 합니다.

## 단계 2: ONNX → Hailo HEF

Hailo Docker 컨테이너 안에서:

```bash
# 컨테이너 진입
docker run --rm -it \
    -v $(pwd):/work \
    hailo8_ai_sw_suite_2025-10:1 \
    /bin/bash

# 컨테이너 내부에서 compile_hef.sh 실행
cd /work
bash deploy/compile_hef.sh hailo8     # Hailo-8 용 (26 TOPS)
bash deploy/compile_hef.sh hailo8l    # Hailo-8L 용 (13 TOPS, Pi5)
```

### compile_hef.sh 의 단계
1. **Parse**: ONNX → HAR (Hailo internal IR)
   ```
   hailo parser onnx psp_mb4.onnx --hw-arch hailo8 \
       --har-path psp_mb4.har --tensor-shapes "input=[1,24,64,25]"
   ```
2. **Optimize**: PTQ + (선택) post-quantization optimization
   ```
   hailo optimize psp_mb4.har --calib-set-path calib.npy \
       --output-har-path psp_mb4_quantized.har --hw-arch hailo8
   ```
   - calib.npy: train split 에서 추출한 2,048 표본 (NHWC `[N, T, J, C]` float32)
   - 출력: `psp_mb4_quantized.har`
3. **Compile**: HAR → HEF (Hailo binary executable)
   ```
   hailo compiler psp_mb4_quantized.har --hw-arch hailo8 --output-dir .
   ```

### Calibration set 생성

```bash
python scripts/build_calibration.py \
    --npy-root data/ntu60/npy \
    --benchmark xsub \
    --n-samples 2048 \
    --clip-range "-10,10" \
    --out calib.npy
```

자세한 calibration 설명 → paper Section 4.5.1 Table H.

### 컴파일 시간
- v1 (PTQ default): 약 30 분
- v2 (PTQ + Adaround + bias correction + QAT): 2–4 시간 (vendor-native QAT 옵션, GPU passthrough 활성화 시)

## 단계 3: NPU 정확도 평가

### Hailo-8 (검증 보드)
```bash
python deploy/eval_on_npu.py \
    --hef psp_mb4_h8.hef \
    --test-x data/ntu60/test_x.npy \
    --test-y data/ntu60/test_y.npy \
    --chunk 256
```

### Hailo-8L (Raspberry Pi 5)
```bash
# Pi5 에서
python deploy/eval_on_npu.py \
    --hef psp_mb4_h8l.hef \
    --test-x data/ntu60/test_x.npy \
    --test-y data/ntu60/test_y.npy \
    --chunk 256
```

전체 16,506 표본 평가: 약 1.5 분 (Hailo-8L Pi5), 약 0.5 분 (Hailo-8).

Expected 결과:
| HEF | Expected acc | Expected FPS |
|---|---:|---:|
| `psp_mb_3d_h8.hef` | 84.71 % | ~3,965 |
| `psp_mb_3d_h8l.hef` | 84.81 % | ~348 |
| `psp_mb4_h8.hef` | 84.37 % | ~388 |
| `psp_mb4_h8l.hef` | 84.35 % | ~200 |

## 단계 4: Raspberry Pi 5 end-to-end 데모

실시간 RTSP 카메라 → YOLO-Pose → PSP-Net → overlay 파이프라인:

```bash
# Pi5 에서
python deploy/pi5_e2e_demo.py \
    --rtsp rtsp://192.168.1.50:554/stream \
    --pose-hef yolov8s_pose_h8l.hef \
    --action-hef psp_mb4_h8l.hef \
    --n-frames 200
```

Expected (200 frame 평균):
- YOLO-Pose: ~19 ms (52 FPS)
- PSP-Net (MB4): ~5 ms (200 FPS)
- End-to-end: ~32 ms (**31.2 FPS** real-time)

자세한 측정 결과 → [RESULTS.md](RESULTS.md) 의 Pi5 end-to-end pipeline 절.

## Troubleshooting

| 증상 | 원인 / 해결 |
|---|---|
| `IndexError: list index out of range` (Hailo parser) | PyTorch 2.0+ 의 dynamo export 사용 시. `dynamo=False` 로 재export. |
| `UnsupportedModelError: zero dimension` | tensor-shapes 가 NHWC 로 지정됨. 학습 ONNX 는 NCHW (`[1, C, T, J]`) 로 지정 필요. |
| `HAILO_OUT_OF_PHYSICAL_DEVICES` | 다른 프로세스가 NPU 점유. `pkill -f hailo` 후 재시도. |
| Pi5 NPU 인식 실패 | 27 W PD 어댑터 필요. 부족 시 NPU 전원 부족. |
| Pi5 SoC 80 °C throttling | Active cooler 또는 heatsink + fan 필요. |
| `Calibration set shape mismatch` | NCHW vs NHWC 확인. Hailo 내부는 NHWC 이며 calib set 도 NHWC 형식이어야 함. |
