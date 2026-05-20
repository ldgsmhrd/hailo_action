# CLAUDE.md

이 파일은 Claude / Codex / 다른 AI 어시스턴트 (그리고 새로 합류한 개발자) 가 빠르게 프로젝트 컨텍스트를 잡기 위한 가이드입니다.

---

## 프로젝트 한 줄

NPU (Hailo-8 / Hailo-8L) 보드에서 RTSP 카메라 영상을 받아 **스켈레톤 키포인트 → pseudo-image → ResNet18 멀티태스크 (5 head)** 로 행동/자세/손/발 동작을 동시에 실시간 인식.

---

## 파이프라인 (3 단계)

```
[학습 / training]              [컴파일 / compile]                  [배포 / deploy]
   AICA GPU 서버         →     Hailo Docker             →          보드 (etri / Pi5)
   PyTorch ResNet18            ONNX → INT8 → HEF                  RTSP → NPU → MJPEG
   ↓                           ↓                                   ↓
  best_mt.pth (44MB)          action_resnet_mt.hef (7MB)         http://<ip>:9999/stream
  → action_resnet_mt.onnx
```

각 단계는 독립적이라 단계별 README 와 코드는 그대로 두고 진행 가능.

---

## 가장 자주 헷갈리는 것 5가지

### 1. **헤드 / 클래스 매핑은 두 곳에서 다르다**

| 위치 | 순서 |
|---|---|
| PyTorch 모델 | ModuleDict 삽입 순서: `upper / lower / pose / hand / foot` |
| ONNX export | 위 순서 그대로 (`output_names` 명시) |
| **Hailo HEF** | **Hailo 컴파일러가 fc1~fc5 를 임의로 재배치** — shape 기반 매핑 필요 |

HEF 매핑 (`deploy/npu/stream_viewer_single.py` 박혀 있음):
```python
FC_TO_HEAD = {
    'fc1': 'action_lower',   # shape 10
    'fc2': 'action_upper',   # shape 6
    'fc3': 'foot',           # shape 3
    'fc4': 'hand',           # shape 3
    'fc5': 'pose',           # shape 9
}
```
**새 HEF 컴파일 시** `hailortcli parse-hef` 로 shape 확인하고 매핑 검증 필수.

### 2. **NCHW vs NHWC — HEF 와 ONNX 가 다르다**

| 형식 | 입력 shape |
|---|---|
| ONNX | NCHW `[1, 7, 60, 25]` |
| HEF | NHWC `[1, 60, 25, 7]` |

`stream_viewer_single.py` 의 `pseudo` 변수가 NCHW 로 들어오면 `np.transpose(0, 2, 3, 1)` 으로 NHWC 변환 후 HEF inference.

### 3. **단일 NPU 보드는 scheduler 모드 필수**

Pi5 + Hailo-8L 처럼 NPU 1개에 pose + action 두 모델 동시 로드하려면:

```python
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
vdevice = VDevice(params=params)
# 두 HEF 모두 configure
# activate() 호출 안 함 — InferVStreams 만 사용 (스케줄러 자동 활성화)
```

`stream_viewer_single.py` 가 이 패턴 사용. 다중 NPU 보드용 `stream_viewer.py` 는 NPU 별 분리된 VDevice 라 다름.

### 4. **ONNX export 는 legacy 사용 (dynamo=False)**

PyTorch 2.0+ 의 새 `torch.export` (dynamo) 로 export 한 ONNX 는 **Hailo parser 가 못 읽음** (`kernel_shape` attribute 누락 에러).

```python
torch.onnx.export(
    wrapped, dummy, out_path,
    opset_version=11,
    dynamo=False,   # ← 반드시 False
)
```

### 5. **학습 데이터 상체+하체 동시 active = 0개**

원본 어노테이션 정책이 "한 번에 한 액션" 라 multi-task 모델이 cross-head correlation 학습해버려서 펀치+걷기 같은 복합 동작은 실시간에서 동시 감지 어려움. 모델 구조 문제 아니라 **데이터 한계**.

해결책 (시도 안 함):
- mixup 합성 데이터
- 분리 학습 (5 모델)
- 새 라벨링

---

## 빠른 명령어 치트시트

### 학습 (AICA 서버)

```bash
cd training
python -m dataset.run_gen_npy_mt              # 13878 NPY 생성
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mt    # 100 epoch ~11분
python -m scripts.eval_mt                     # test 평가
python -m scripts.export_onnx_mt              # ONNX export
```

### HEF 컴파일 (Hailo Docker)

```bash
cd compile
docker run --rm -v $(pwd)/..:/work hailo8_ai_sw_suite_2025-10:1 \
  bash -c "cd /work/compile && bash recompile_mt.sh"      # Hailo-8
# 또는
  bash -c "cd /work/compile && bash recompile_mt_8l.sh"   # Hailo-8L
```

### 보드 실행

```bash
# etri-board (Hailo-8 ×4)
docker exec arn-npu python3 /app/npu/stream_viewer.py \
  --rtsp rtsp://... --action-hef /app/models/action_resnet_mt.hef

# Raspberry Pi 5 (Hailo-8L)
sudo apt install hailo-all     # 한 줄로 driver + runtime
python3 deploy/npu/stream_viewer_single.py \
  --rtsp rtsp://... --pose-hef ... --action-hef .../action_resnet_mt_h8l.hef

# CPU only (벤치)
python3 deploy/npu/stream_viewer_cpu.py --rtsp ... --pose-imgsz 320
```

### 보드 상태 확인 (Pi5)

```bash
hailortcli scan                  # NPU 인식
hailortcli fw-control identify   # architecture 확인 (hailo8 vs hailo8l)
vcgencmd get_throttled           # 0x0=정상, 0x5xxxx=under-voltage
vcgencmd measure_temp            # 80°C 이하 권장
```

---

## 보드별 실측 성능

| 보드 | NPU | Pose 모델 | FPS | NPU 사용량 |
|---|---|---|---:|---|
| etri-ADL-N | Hailo-8 ×4 | yolov8m_pose | 22 | NPU0 45% / NPU1 0.4% |
| RPi5 | Hailo-8L ×1 | yolov8s_pose | 20 | 합계 37% (시분할) |
| RPi5 | 없음 (FP32) | yolov8n_pose @320 | 8 | CPU 89°C (위험) |
| RPi5 | 없음 (INT8) | yolov8n_pose @320 | 5 | ARM ONNX INT8 효율 ↓ |

---

## 학습된 클래스 (28개)

```
상체 (5)  : 없음 / 펀치 / 손흔들기 / 손뼉치기 / 손올리기
하체 (10) : 없음 / 서성이기 / 걷기 / 달리기 / 점프-제자리 / 넘어짐 / 킥
            / 점프-두발 / 외발점프 / 외발점프-제자리
자세 (7)  : 바닥앉기 / 의자앉기 / 무릎꿇기 / 서있기 / 허리구부리기 / 무릎기기 / 기타
손 (3)    : 없음 / 팔짱끼기 / 양팔들기
발 (3)    : 없음 / 다리꼬기 / 한쪽다리들기
```

원본 31 출력 중 데이터 0~5개로 미학습된 클래스 3개 (손내리기, 무릎서기, 누워있기) 제외.

---

## 양자화 손실

| Head | PyTorch | HEF | Gap |
|---|---:|---:|---:|
| 평균 | 95.78% | 95.73% | **-0.05%** |

ResNet18 + Linear head 라 PTQ 기본만으로도 거의 무손실. QAT / Bias Correction / Adaround 같은 추가 최적화 안 함 (GPU 패스스루 없는 Docker 환경 때문). 필요 시 GPU + nvidia-container-toolkit 환경에서 재컴파일하면 +alpha 가능.

---

## 의존성 함정

### Pi5 (Hailo-8L) 환경에서 자주 막힘

| 증상 | 원인 / 해결 |
|---|---|
| `HAILO_OUT_OF_PHYSICAL_DEVICES (74)` | 다른 프로세스가 NPU 점유. `hailort.service` 끄거나 pkill |
| `HAILO_DEVICE_IN_USE (73)` | multi-process 충돌. scheduler 모드로 단일 process 통합 |
| 부팅 직후 1~2분 안에 꺼짐 | 전원 부족. 27W 5V/5A PD 어댑터 필요 |
| `ModuleNotFoundError: torch` | smtrack 이 torch 의존. `pip install torch --index-url ...cpu` |
| `ModuleNotFoundError: termcolor` | mmengine 의존. `pip install termcolor` |
| `'ByteTrackerRunner' object has no attribute 'runner_inference'` | method 이름 다름. `run_tracker(det_bboxes=..., det_labels=..., frame_id=...)` |

### ONNX export 후 Hailo parser 에러

| 증상 | 해결 |
|---|---|
| `IndexError: list index out of range` at `get_kernel_shape` | dynamo export → `dynamo=False` 로 재export |

---

## 모델 / 데이터 위치

| 자산 | 위치 |
|---|---|
| 학습 데이터 (clip JSON) | AICA `/home/ubuntu/safemotion/action_dataset/` |
| 학습 결과 (best_mt.pth) | AICA `/home/ubuntu/safemotion/ResNet-ActionRecognition/models/` |
| Hailo Docker 이미지 (8.7GB) | 로컬 `/home/ldg/smartnvr-backend/hailo/hailo8_ai_sw_suite_2025-10.tar.gz` |
| 모델 바이너리 (HEF/ONNX) | **`.gitignore` 제외** — 별도 storage 에서 받아 `models/` 에 배치 |
| 보드 (etri) 코드 | `etri-board:/home/etri/action_recognition_npu/` |
| 보드 (Pi5) 코드 | `picam:~/action_recognition_npu/` |

---

## 자주 묻는 질문

**Q. Action 인식이 안 되는 거 같아요**
A. tracking 끊겼는지 확인. Pose 검출은 frame 마다 되지만 action 추론은 **60 frame 누적 후 sliding stride 8** 마다 호출. 사람이 화면에 2초 이상 머물러야 첫 결과 나옴.

**Q. FPS 가 너무 낮아요**
A. NPU 사용 중인지 먼저 확인. `lsof /dev/hailo0` 비어있으면 CPU 로 fallback 된 것. `hailortcli scan` 으로 NPU 보이는지 확인.

**Q. 한글이 깨져요**
A. NotoSansCJK 또는 NanumGothic 폰트 설치. `stream_viewer*.py` 의 `_FONT_CANDIDATES` 가 자동 탐색하지만 폰트 자체가 없으면 OpenCV로 fallback → 한글 ?  표시.

**Q. fall 인데 standing 으로 나와요**
A. `_is_upright_pose` 후처리가 의도적으로 override. bbox 세로/가로 비율 1.5 이상 또는 머리-발 수직 거리 50% 이상이면 자동 standing. 책상 작업 false alarm 방지용.

---

## 학습 결과 상세

전체 학습 / 평가 / 양자화 검증 결과는 [`docs/MULTITASK_MODEL_REPORT.md`](docs/MULTITASK_MODEL_REPORT.md) 참고.

오분류 분석 (헷갈린 top-3): [`eval/README.md`](eval/README.md) + `eval_mt_confusion.py` 실행 결과.
