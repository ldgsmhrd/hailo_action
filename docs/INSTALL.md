# INSTALL — 환경 설정

## 1. 학습 환경 (GPU)

### 권장 사양
- OS: Ubuntu 22.04 LTS (or 20.04)
- Python: 3.10+
- CUDA: 12.8+
- GPU: NVIDIA A100 80 GB 권장 (RTX 3090 / 4090 도 동작)
- RAM: 32 GB+
- 디스크: 50 GB+ (NTU60 + NTU120 raw + npy)

### 설치

```bash
# Python venv
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# PyTorch 는 CUDA 버전 맞춰 별도 설치 권장:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

학습 시 약 41–60 초 / epoch (A100, batch 64). 전체 120 epoch 약 80–120 분.

## 2. Hailo NPU 컴파일 환경

### Hailo Dataflow Compiler (DFC)
NPU HEF 컴파일은 Hailo 공식 Docker 이미지를 사용합니다.

```bash
# Hailo 공식 사이트에서 docker tar.gz 다운로드 (계정 필요)
# https://hailo.ai/developer-zone/software-downloads/
docker load -i hailo8_ai_sw_suite_2025-10.tar.gz

# 컴파일 컨테이너 실행
docker run --rm -it \
    -v $(pwd):/work \
    hailo8_ai_sw_suite_2025-10:1 \
    /bin/bash
```

자세한 컴파일 방법 → [DEPLOY.md](DEPLOY.md)

## 3. Raspberry Pi 5 + Hailo-8L 배포 환경

### 하드웨어
- Raspberry Pi 5 (8 GB RAM 권장)
- Hailo-8L M.2 accessory (13 TOPS, 2230 form factor)
- 27 W PD 어댑터 (5 V / 5 A) — 부족 시 NPU 인식 실패 가능
- (선택) Active cooler / heatsink — 80 °C throttling 방지

### 소프트웨어 설치

```bash
# Raspberry Pi OS Bookworm 64-bit 기준
sudo apt update
sudo apt install hailo-all     # HailoRT + driver + firmware 한 번에

# 인식 확인
hailortcli scan                 # 디바이스 보여야 함
hailortcli fw-control identify  # architecture: hailo8l 확인
```

### OS / 펌웨어 권장
- Raspberry Pi OS Bookworm 64-bit (December 2023+)
- HailoRT 4.23+
- Firmware: matching driver version

### 트러블슈팅
- `HAILO_OUT_OF_PHYSICAL_DEVICES (74)` — 다른 프로세스가 NPU 점유. `pkill -f hailort` 또는 reboot.
- `HAILO_DEVICE_IN_USE (73)` — multi-process 충돌. scheduler 모드로 통합.
- 부팅 직후 1–2 분 안에 꺼짐 — 전원 부족. 27 W PD 어댑터로 교체.
- SoC 80 °C 초과 — passive cooling 부족. heatsink 또는 active fan 추가.

## 4. 검증

설치 완료 후 sanity check:

```python
# PyTorch + GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"

# Hailo (Pi5 / Hailo-8 보드에서)
python -c "from hailo_platform import VDevice; print('Hailo OK')"
```
