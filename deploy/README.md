# Deploy — 실시간 추론 + MJPEG 스트림

보드에서 RTSP 카메라 입력 → pose+action 추론 → 라벨 그린 MJPEG 스트림 송출.

## 3가지 변종

| 파일 | 보드 | NPU | 구조 |
|---|---|---|---|
| `stream_viewer.py` | etri-board | Hailo-8 ×4 | Multi-process (pose / action 분리) |
| `stream_viewer_single.py` | RPi5 + Hailo-8L AI Kit | Hailo-8L ×1 | Single-process + VDevice scheduler |
| `stream_viewer_cpu.py` | NPU 없음 | (CPU) | YOLOv8n-pose ONNX + ONNX Runtime |

## 공통 모듈

```
deploy/npu/
├── init_hailo.py        # VDevice + HEF configure helper (single/multi 자동 분기)
├── pose_extractor.py    # YOLOv8 pose raw output → 17 keypoint + bbox
├── pseudo_image.py      # 17 COCO → 25 TSSI + 7 채널 인코딩 (학습과 동일)
└── (action_classifier.py)
```

---

## A) etri-board 실행 (다중 NPU)

NPU 4 개 사용 가능한 보드. Pose / Action 워커 분리해서 NPU 1개씩 할당.

```bash
cd deploy/npu
python3 stream_viewer.py \
  --rtsp rtsp://admin:pw@192.168.1.175:554/Streaming/Channels/101 \
  --port 9999 \
  --pose-hef /app/models/yolov8m_pose.hef \
  --action-hef /app/models/action_resnet_mt.hef
```

- Reader thread (RTSP) → PoseWorker (NPU0) → Queue → ActionDrawWorker (NPU1) → MJPEG
- 실측: FPS 22 / NPU0 45% / NPU1 0.4%

Docker 안에서 실행 권장 (HailoRT 환경 격리):
```bash
docker exec -d arn-npu bash -c "cd /app && python3 /app/npu/stream_viewer.py --rtsp ... --action-hef /app/models/action_resnet_mt.hef ..."
```

---

## B) RPi5 + Hailo-8L 실행 (단일 NPU)

NPU 1개에 pose + action 동시 로드 — VDevice scheduler 가 시분할 처리.

```bash
cd deploy/npu
PYTHONPATH=$HOME:$HOME/action_recognition_npu:$HOME/action_recognition_npu/src:$HOME/action_recognition_npu/npu \
  python3 stream_viewer_single.py \
  --rtsp rtsp://admin:pw@192.168.1.175:554/Streaming/Channels/101 \
  --port 9999 \
  --pose-hef ~/models/yolov8s_pose_h8l.hef \
  --action-hef ~/models/action_resnet_mt_h8l.hef
```

핵심: `init_hailo.py` 의 `VDevice.create_params()` 에 `HailoSchedulingAlgorithm.ROUND_ROBIN` 설정 + `activate()` 호출 없이 `InferVStreams` 만 사용.

- 실측: FPS 20 / NPU 합계 37% (시분할)
- 한글 라벨 표시 — `/usr/share/fonts/truetype/nanum/NanumGothic.ttf` 자동 탐색

### Pi5 설치 한 줄

```bash
sudo apt install hailo-all     # driver + runtime + python binding 모두
# 재부팅 후 /dev/hailo0 + hailortcli scan 으로 확인
```

### Pi5 추가 설정 — 전원

Hailo 추론 시 피크 18W → 일반 5V/3A 어댑터로는 under-voltage 발생. **공식 27W (5V/5A) PD 어댑터 권장**.

```bash
vcgencmd get_throttled    # 0x0 = 정상, 0x50000 이면 throttling 발생
vcgencmd measure_temp     # 80°C 이하 권장
```

---

## C) CPU 단독 (NPU 없음 — 벤치마크)

ONNX Runtime CPU + ARM NEON.

```bash
cd deploy/npu
python3 stream_viewer_cpu.py \
  --rtsp rtsp://admin:pw@192.168.1.175:554/Streaming/Channels/101 \
  --port 9999 \
  --pose-imgsz 320 \
  --action-onnx ~/models/action_resnet_mt.onnx
```

- 실측: FPS 8 (FP32), 5 (INT8 dynamic) — ARM 의 INT8 효율 떨어짐
- 카메라 제품으로는 부적합 (발열 89°C, 전원 부족)

---

## 출력

각 변종 모두 동일:

| URL | 내용 |
|---|---|
| `http://<board-ip>:9999/stream` | MJPEG 실시간 스트림 |
| `http://<board-ip>:9999/snapshot` | 단일 JPEG 스냅샷 |
| `http://<board-ip>:9999/health` | health check |

같은 LAN 의 모든 기기 (브라우저, 모바일, VLC) 에서 바로 접근 가능. 외부 접근은 SSH 터널 (`ssh -L 9999:localhost:9999`) 권장.

---

## 라벨 표시 (5 head 동시)

각 track 마다 5 줄로 한글(영어) NN% 형식:

```
상체 펀치(punch) 87%
하체 걷기(walk) 78%
자세 서있기(standing) 91%
손 없음(none) 100%
발 없음(none) 100%
```

상태 바 (상단): `FPS:NN | Tracks:N | Frames:NN | NPU(Pose):NN% | NPU(Act):NN%`

---

## HEF 출력 매핑 주의

Hailo 컴파일러가 fc1~fc5 를 PyTorch ModuleDict 순서와 **무관하게** 부여.
`stream_viewer*.py` 에 shape-based 매핑 박혀 있음:

| HEF 출력 | shape | 실제 head |
|---|---:|---|
| fc1 | 10 | action_lower |
| fc2 | 6 | action_upper |
| fc3 | 3 | foot |
| fc4 | 3 | hand |
| fc5 | 9 | pose |

새 HEF 컴파일 시 매핑 검증 필요 (`hailortcli parse-hef` 로 출력 shape 확인).

---

## 의존성 — Pi5 기준

```bash
sudo apt install hailo-all
sudo pip3 install --break-system-packages \
  opencv-python numpy Pillow \
  torch torchvision \
  lap cython_bbox scipy filterpy mmengine \
  addict termcolor onnxruntime ultralytics onnx
```

font (한글 라벨):
```bash
sudo apt install fonts-nanum
```

자세한 보드별 설치는 [`docs/MULTITASK_MODEL_REPORT.md`](../docs/MULTITASK_MODEL_REPORT.md) §8 참고.
