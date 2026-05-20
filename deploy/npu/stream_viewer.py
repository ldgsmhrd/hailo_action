"""
NPU 검증용 실시간 스트림 뷰어 — Multi-Process 최적화 버전.

아키텍처 (N100 4 core 활용):
  ┌──────────────────────────────────────────────────────────┐
  │ Main Process (Core 0)                                    │
  │  - RTSP reader thread                                    │
  │  - MJPEG HTTP server thread                              │
  │  - shared_memory[frame] / shared_memory[annotated]       │
  └──────────────────────────────────────────────────────────┘
            │ shared mem (frame, latest-only)
            ↓
  ┌──────────────────────────────────────────────────────────┐
  │ Pose Worker Process (Core 1)                             │
  │  - Pose NPU 0 (yolov8m_pose)                             │
  │  - postprocess_pose_multi (9-output DFL decoding)        │
  │  - ByteTracker                                           │
  └──────────────────────────────────────────────────────────┘
            │ Queue: tracked detections + frame copy
            ↓
  ┌──────────────────────────────────────────────────────────┐
  │ Action+Draw Worker Process (Core 2)                      │
  │  - keypoint 버퍼링 (60 frames per track)                  │
  │  - Action NPU 1 (action_resnet18_5cls)                   │
  │  - Skeleton/bbox/label drawing                            │
  │  - shared_memory[annotated] 갱신                           │
  └──────────────────────────────────────────────────────────┘

성능 개선:
  - Python GIL 우회 (process 별 독립 인터프리터)
  - 3 코어 풀 활용 (이전: 1 코어만)
  - MJPEG / Reader / Inference 가 서로 블록 안 함

실행:
    python3 /app/npu/stream_viewer.py --rtsp <URL> --port 9999
"""
import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import shared_memory
from queue import Empty
from socketserver import ThreadingMixIn

import cv2
import numpy as np

# 실행 환경에 따라 경로 자동: docker (/app) 또는 native (스크립트 기준)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in ('/app/npu', '/app/src', '/app', _HERE, os.path.join(_REPO, 'src'), _REPO):
    if os.path.isdir(p):
        sys.path.insert(0, p)
# smtrack / smutils 는 보통 home 또는 repo 부모에 있음
for p in (os.path.expanduser('~'), os.path.dirname(_REPO)):
    if os.path.isdir(p):
        sys.path.insert(0, p)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s [%(processName)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 시각화 상수 (bus-eva 와 동일)
# ============================================================
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
SKELETON_COLORS = [
    (0,255,255),(0,255,255),(0,255,255),(0,255,255),
    (0,255,0),
    (255,128,0),(255,128,0),
    (0,128,255),(0,128,255),
    (255,255,255),(255,255,255),
    (255,255,255),
    (255,0,128),(255,0,128),
    (128,0,255),(128,0,255),
]
TRACK_COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),
    (255,0,255),(0,255,255),(128,255,0),(255,128,0),
    (0,128,255),(128,0,255),(255,0,128),(0,255,128),
]

# ============================================================
# Multi-task 모델 클래스 정의 (5 head / 31 outputs)
# 영어 idx ↔ (한글, 영어) 매핑
# ============================================================
MT_CLASSES = {
    'action_upper': ['none', 'punch', 'wave', 'clap', 'raise', 'put-down'],
    'action_lower': ['none', 'pacing', 'walk', 'run', 'jump-still', 'fall',
                     'kick', 'jump-2feet', 'jump-1leg', 'jump-1leg-still'],
    'pose':         ['sit', 'sit-chair', 'kneel-down', 'knee-standing',
                     'standing', 'standing-bending', 'lying', 'crawl', 'other'],
    'hand':         ['none', 'cross-arms', 'raise-both'],
    'foot':         ['none', 'leg-cross', 'one-leg-raise'],
}

# 한글 라벨 (인덱스가 동일하게 대응)
MT_CLASSES_KR = {
    'action_upper': ['없음', '펀치', '손흔들기', '손뼉치기', '손올리기', '손내리기'],
    'action_lower': ['없음', '서성이기', '걷기', '달리기', '점프-제자리', '넘어짐',
                     '킥', '점프-두발', '외발점프', '외발점프-제자리'],
    'pose':         ['바닥앉기', '의자앉기', '무릎꿇기', '무릎서기',
                     '서있기', '허리구부리기', '누워있기', '무릎기기', '기타'],
    'hand':         ['없음', '팔짱끼기', '양팔들기'],
    'foot':         ['없음', '다리꼬기', '한쪽다리들기'],
}

# 헤드명 한글 (짧게)
HEAD_LABEL_KR = {
    'action_upper': '상체',
    'action_lower': '하체',
    'pose':         '자세',
    'hand':         '손',
    'foot':         '발',
}

# HEF 출력 텐서 이름 → head 이름 매핑 (output_infos 순서대로)
# ONNX 의 output_names 와 동일 순서로 컴파일됨
MT_HEAD_ORDER = ['action_upper', 'action_lower', 'pose', 'hand', 'foot']

# fall / none 색상 (메인 라벨에 사용)
FALL_COLOR = (0, 0, 255)         # 빨강 - alarm
NORMAL_COLOR = (0, 255, 0)        # 초록 - 정상
DIM_COLOR = (180, 180, 180)       # 회색 - none/idle

NUM_FRAMES = 60
SLIDING_STRIDE = 8
ACTION_DISPLAY_TTL = 2.0
POSE_CONF_THR = float(os.environ.get('POSE_CONF_THR', '0.1'))
DEBUG_POSE = os.environ.get('DEBUG_POSE', '1') == '1'

# 한글 폰트 (PIL 로 렌더링)
# 환경별 기본: docker arn-npu → /share/NotoSansCJK / Pi → NanumGothic / fallback
_FONT_CANDIDATES = [
    os.environ.get('KOREAN_FONT_PATH'),
    '/share/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf',
]
KOREAN_FONT_PATH = next((p for p in _FONT_CANDIDATES if p and os.path.exists(p)), None)
_KR_FONT_CACHE = {}

def _get_kr_font(size=14):
    if size in _KR_FONT_CACHE:
        return _KR_FONT_CACHE[size]
    try:
        from PIL import ImageFont
        f = ImageFont.truetype(KOREAN_FONT_PATH, size)
    except Exception:
        f = None
    _KR_FONT_CACHE[size] = f
    return f


def _draw_text_kr(frame, lines_with_color, x, y, font_size=14, line_gap=4, bg=(0,0,0)):
    """frame (BGR ndarray) 에 한글 multi-line 텍스트 그리기. PIL 경유.

    lines_with_color: [(text, (B,G,R)), ...]
    return: (drawn_w, drawn_h) — 그려진 박스 크기
    """
    from PIL import Image, ImageDraw
    font = _get_kr_font(font_size)
    if font is None:
        # fallback to OpenCV (한글 깨짐)
        for i, (t, c) in enumerate(lines_with_color):
            cv2.putText(frame, t, (x, y + (i+1)*font_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        return 200, len(lines_with_color)*font_size

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 박스 너비 계산
    line_h = font_size + line_gap
    max_w = 0
    for t, _ in lines_with_color:
        bbox = draw.textbbox((0, 0), t, font=font)
        max_w = max(max_w, bbox[2] - bbox[0])
    box_w = max_w + 8
    box_h = len(lines_with_color) * line_h + 4
    # 배경
    draw.rectangle([x, y, x + box_w, y + box_h], fill=tuple(reversed(bg)))
    # 텍스트 (PIL 은 RGB)
    for i, (t, c) in enumerate(lines_with_color):
        rgb = (c[2], c[1], c[0])
        draw.text((x + 4, y + i*line_h + 2), t, font=font, fill=rgb)
    # 다시 BGR ndarray 로
    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # in-place 복사
    frame[:] = out
    return box_w, box_h

# ============================================================
# 공유 메모리 (최대 1920x1080 BGR)
# ============================================================
SHM_MAX_W, SHM_MAX_H = 1920, 1080
SHM_BYTES = SHM_MAX_H * SHM_MAX_W * 3        # 6.2 MB
SHM_FRAME_NAME = 'snvr_frame_raw'
SHM_ANNO_NAME = 'snvr_frame_anno'
SHM_META_NAME = 'snvr_meta'                  # [width, height, frame_id, ann_frame_id]


def _alloc_shm(name, size):
    """기존 동명 SHM 있으면 정리 후 새로 생성."""
    try:
        old = shared_memory.SharedMemory(name=name)
        old.close(); old.unlink()
    except FileNotFoundError:
        pass
    return shared_memory.SharedMemory(create=True, size=size, name=name)


# ============================================================
# Reader Thread (main process) — RTSP → shared frame
# ============================================================
def reader_thread(rtsp_url, shm_frame, shm_meta, stop_event):
    os.environ.setdefault(
        'OPENCV_FFMPEG_CAPTURE_OPTIONS',
        'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000'
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Reader cannot open: {rtsp_url}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Reader started.")

    frame_buf = np.ndarray((SHM_MAX_H, SHM_MAX_W, 3), dtype=np.uint8, buffer=shm_frame.buf)
    meta_buf = np.ndarray((4,), dtype=np.int64, buffer=shm_meta.buf)
    frame_id = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Reader reconnect...")
            cap.release(); time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        H, W = frame.shape[:2]
        if H > SHM_MAX_H or W > SHM_MAX_W:
            scale = min(SHM_MAX_H / H, SHM_MAX_W / W)
            frame = cv2.resize(frame, (int(W * scale), int(H * scale)))
            H, W = frame.shape[:2]
        frame_buf[:H, :W] = frame
        frame_id += 1
        meta_buf[0] = W; meta_buf[1] = H; meta_buf[2] = frame_id
    cap.release()
    logger.info("Reader exit.")


# ============================================================
# Pose Worker Process — 추론 + tracker
# ============================================================
def pose_worker(pose_hef, shm_frame_name, shm_meta_name, det_queue, stop_event,
                npu_stats):
    proc_name = mp.current_process().name
    logger.info(f"{proc_name} starting...")
    from hailo_platform import Device, InferVStreams
    from init_hailo import init_pose_npu
    from pose_extractor import postprocess_pose_multi
    from smtrack.builder.runner_builder import ByteTrackerRunner
    from mmengine.config import Config
    import torch

    devs = Device.scan()
    if not devs:
        logger.error("Pose: no NPU"); return
    pose_h = init_pose_npu(device_id=devs[0], npu_index=0, pose_hef_path=pose_hef)

    PERSON_TRACKER_CONFIG = Config(dict(
        type="ByteTracker",
        obj_score_thrs=dict(high=0.4, low=0.1),     # bus-eva BotSORT 와 동일 (low FPS 대응)
        init_track_thr=0.5,                          # 새 track 시작 임계값 (0.7 → 0.5)
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        use_cate_match=True,
        use_second_match_case=True,
        num_frames_retain=30,
    ))
    MOTION_CONFIG = Config(dict(type="KalmanFilter"))
    tracker = ByteTrackerRunner(tracker=PERSON_TRACKER_CONFIG, motion=MOTION_CONFIG)

    shm_f = shared_memory.SharedMemory(name=shm_frame_name)
    shm_m = shared_memory.SharedMemory(name=shm_meta_name)
    frame_buf = np.ndarray((SHM_MAX_H, SHM_MAX_W, 3), dtype=np.uint8, buffer=shm_f.buf)
    meta_buf = np.ndarray((4,), dtype=np.int64, buffer=shm_m.buf)

    in_name = pose_h['input_info'].name
    in_h, in_w = _model_in_hw(pose_h['input_info'])
    last_frame_id = -1
    frame_id_counter = 0

    with pose_h['network_group'].activate(pose_h['network_group_params']), \
         InferVStreams(pose_h['network_group'],
                       pose_h['input_params'],
                       pose_h['output_params']) as pose_pipe:
        logger.info(f"{proc_name} NPU ready.")
        while not stop_event.is_set():
            fid = int(meta_buf[2])
            if fid == last_frame_id:
                time.sleep(0.005); continue
            last_frame_id = fid
            W, H = int(meta_buf[0]), int(meta_buf[1])
            if W == 0 or H == 0:
                time.sleep(0.01); continue
            frame = frame_buf[:H, :W].copy()
            frame_id_counter += 1

            # pose 추론 (NPU 활용률 측정)
            img = cv2.resize(frame, (in_w, in_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            _t0 = time.time()
            raw = pose_pipe.infer({in_name: img[None, ...]})
            _dt = time.time() - _t0
            npu_stats['pose_t_sum'] = npu_stats.get('pose_t_sum', 0.0) + _dt
            npu_stats['pose_count'] = npu_stats.get('pose_count', 0) + 1
            if npu_stats['pose_count'] % 30 == 0:
                npu_stats['pose_last_avg_ms'] = (npu_stats['pose_t_sum'] / npu_stats['pose_count']) * 1000
            detections = postprocess_pose_multi(
                raw, orig_h=H, orig_w=W,
                conf_thr=POSE_CONF_THR, model_in_h=in_h, model_in_w=in_w,
            )

            # 디버그: 매 30프레임마다 cls head + detection 통계
            if DEBUG_POSE and frame_id_counter % 30 == 0:
                stats = []
                for name, arr in raw.items():
                    a = np.asarray(arr)
                    if a.ndim == 4: a = a[0]
                    if a.shape[-1] == 1:
                        stats.append(f"{name.split('/')[-1]}:max={float(a.max()):.3f}")
                det_info = []
                for d in detections[:3]:
                    b = d['box']
                    det_info.append(f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])},s={b[4]:.2f}]")
                logger.info(f"  pose f={frame_id_counter} dets={len(detections)} {' '.join(det_info)}  cls[{' '.join(stats)}]")

            # tracker
            if detections:
                boxes = torch.tensor(
                    np.array([d['box'] for d in detections], dtype=np.float32))
                labels = torch.zeros(len(detections), dtype=torch.long)
                track_out = tracker.run_tracker(
                    det_bboxes=boxes, det_labels=labels,
                    frame_id=frame_id_counter, num_classes=1)
                tracked = _match_tracks_to_detections(track_out, detections)
                if DEBUG_POSE and frame_id_counter % 30 == 0:
                    tb = track_out.get('track_bboxes', [])
                    tb_info = "empty"
                    if tb:
                        first = tb[0] if isinstance(tb, list) else tb
                        if first is not None and len(first) > 0:
                            tb_info = f"shape={np.asarray(first).shape} first={first[0] if len(first)>0 else None}"
                    logger.info(f"  tracker_out keys={list(track_out.keys())} track_bboxes:{tb_info}")
            else:
                tracked = []

            # 결과를 queue 로 (frame + tracked)
            try:
                det_queue.put_nowait({
                    'frame': frame, 'tracked': tracked, 'frame_id': frame_id_counter,
                })
            except Exception:
                if DEBUG_POSE and frame_id_counter % 30 == 0:
                    logger.info(f"  queue FULL — dropped {len(tracked)} tracks")
            if DEBUG_POSE and frame_id_counter % 30 == 0:
                logger.info(f"  tracks={len(tracked)} q={det_queue.qsize()}")

    shm_f.close(); shm_m.close()
    logger.info(f"{proc_name} exit.")


# ============================================================
# Action + Draw Worker Process
# ============================================================
def action_draw_worker(action_hef, det_queue, shm_anno_name, shm_meta_name,
                       stop_event, status_dict, npu_stats):
    proc_name = mp.current_process().name
    logger.info(f"{proc_name} starting...")
    from hailo_platform import Device, InferVStreams
    from init_hailo import init_action_npu
    from pseudo_image import keypoints_to_pseudo_image

    devs = Device.scan()
    if not devs:
        logger.error("Action: no NPU"); return
    # 단일 NPU 면 [0] 공유, 멀티 NPU 면 [1] 사용
    dev_id = devs[1] if len(devs) >= 2 else devs[0]
    npu_idx = 1 if len(devs) >= 2 else 0
    action_h = init_action_npu(device_id=dev_id, npu_index=npu_idx, action_hef_path=action_hef)

    shm_a = shared_memory.SharedMemory(name=shm_anno_name)
    shm_m = shared_memory.SharedMemory(name=shm_meta_name)
    anno_buf = np.ndarray((SHM_MAX_H, SHM_MAX_W, 3), dtype=np.uint8, buffer=shm_a.buf)
    meta_buf = np.ndarray((4,), dtype=np.int64, buffer=shm_m.buf)

    in_name = action_h['input_info'].name
    # 5 head 출력 — Hailo 컴파일러가 fc1~fc5 를 ModuleDict 순서와 무관하게 부여.
    # 출력 shape(=클래스 수) 가 unique 한 head 는 shape 기반 자동 매핑,
    # hand/foot 처럼 shape 가 같으면 fixed FC 번호로 구분 (test eval 결과 검증).
    # 실제 HEF: fc1=10 lower / fc2=6 upper / fc3=3 foot / fc4=3 hand / fc5=9 pose
    FC_TO_HEAD = {
        'fc1': 'action_lower',
        'fc2': 'action_upper',
        'fc3': 'foot',
        'fc4': 'hand',
        'fc5': 'pose',
    }
    head_to_vsname = {}
    for o in action_h['output_infos']:
        suffix = o.name.rsplit('/', 1)[-1]
        head = FC_TO_HEAD.get(suffix)
        if head:
            head_to_vsname[head] = o.name
    logger.info(f"Multi-task head→vstream: {head_to_vsname}")
    missing = [h for h in MT_HEAD_ORDER if h not in head_to_vsname]
    if missing:
        logger.warning(f"Missing head mapping: {missing}")

    kp_buffers = defaultdict(lambda: deque(maxlen=NUM_FRAMES))
    last_action_frame = defaultdict(int)
    last_action_result = {}

    fps_window = deque(maxlen=30)
    t_prev = time.time()
    t_prev_start = time.time()    # NPU util 계산용 시작 시각
    ann_id = 0

    with action_h['network_group'].activate(action_h['network_group_params']), \
         InferVStreams(action_h['network_group'],
                       action_h['input_params'],
                       action_h['output_params']) as action_pipe:
        logger.info(f"{proc_name} NPU ready.")
        while not stop_event.is_set():
            try:
                pkg = det_queue.get(timeout=0.3)
            except Empty:
                continue
            frame = pkg['frame']
            tracked = pkg['tracked']
            fid = pkg['frame_id']
            H, W = frame.shape[:2]

            # 사람별 keypoint 누적 + action 추론
            for tr in tracked:
                tid = tr['track_id']
                kp_buffers[tid].append(tr['keypoints'])
                if (len(kp_buffers[tid]) == NUM_FRAMES
                        and fid - last_action_frame[tid] >= SLIDING_STRIDE):
                    last_action_frame[tid] = fid
                    kp_seq = np.stack(list(kp_buffers[tid]), axis=0)
                    pseudo = keypoints_to_pseudo_image(kp_seq, frame_w=W, frame_h=H)
                    if pseudo.shape[1] == 7:
                        pseudo = np.transpose(pseudo, (0, 2, 3, 1)).astype(np.float32)
                    _t0 = time.time()
                    a_out = action_pipe.infer({in_name: pseudo})
                    _dt = time.time() - _t0
                    npu_stats['act_t_sum'] = npu_stats.get('act_t_sum', 0.0) + _dt
                    npu_stats['act_count'] = npu_stats.get('act_count', 0) + 1
                    if npu_stats['act_count'] % 30 == 0:
                        npu_stats['act_last_avg_ms'] = (npu_stats['act_t_sum'] / npu_stats['act_count']) * 1000
                    # 5-head 결과 모두 처리 (softmax + argmax)
                    mt_result = {}
                    for head_name in MT_HEAD_ORDER:
                        vs_name = head_to_vsname.get(head_name)
                        if vs_name is None:
                            continue
                        logits = np.asarray(a_out[vs_name]).reshape(-1)
                        # softmax (numerical stable)
                        e = np.exp(logits - logits.max())
                        probs = e / e.sum()
                        cidx = int(probs.argmax())
                        names = MT_CLASSES.get(head_name, [])
                        cname = names[cidx] if cidx < len(names) else f"cls{cidx}"
                        mt_result[head_name] = (cname, float(probs[cidx]))
                    last_action_result[tid] = (mt_result, time.time())

            # 그리기
            out = frame  # in-place ok (queue 에서 받은 거 우리 거)
            _draw_tracks(out, tracked, kp_buffers, last_action_result)

            # FPS / 상태 바
            now = time.time()
            dt = now - t_prev; t_prev = now
            if dt > 0:
                fps_window.append(1.0 / dt)
            fps = float(np.mean(fps_window)) if fps_window else 0.0
            status_dict['fps'] = fps
            status_dict['tracks'] = len(tracked)
            status_dict['frames'] = fid

            # NPU 활용률 계산: 평균 추론시간 × 처리량(FPS) / 1초
            pose_util = 0.0
            act_util = 0.0
            if npu_stats.get('pose_count', 0) > 0:
                pose_avg_ms = npu_stats.get('pose_last_avg_ms', 0)
                pose_util = pose_avg_ms * fps / 10.0   # ms * (inf/sec) / 1000 * 100 = util%
                status_dict['npu0_util'] = round(pose_util, 1)
            if npu_stats.get('act_count', 0) > 0 and fps > 0:
                act_avg_ms = npu_stats.get('act_last_avg_ms', 0)
                # action 은 매 sliding stride 8 마다 + track 수만큼 호출
                act_inferences_per_sec = (npu_stats['act_count'] / max(time.time() - t_prev_start, 1))
                act_util = act_avg_ms * act_inferences_per_sec / 10.0
                status_dict['npu1_util'] = round(act_util, 2)

            _draw_status_bar(out, fps, len(tracked), fid, pose_util, act_util)

            # shared annotated mem 으로 (W, H 도 갱신)
            anno_buf[:H, :W] = out
            meta_buf[0] = W
            meta_buf[1] = H
            ann_id += 1
            meta_buf[3] = ann_id   # annotated frame id

    shm_a.close(); shm_m.close()
    logger.info(f"{proc_name} exit.")


# ============================================================
# 헬퍼
# ============================================================
def _model_in_hw(input_info):
    s = tuple(input_info.shape)
    if len(s) == 3: return s[0], s[1]
    if len(s) == 4:
        if s[1] in (3, 1): return s[2], s[3]
        return s[1], s[2]
    return 640, 640


def _is_upright_pose(box, keypoints):
    """skeleton 기반 직립 여부 판별.

    fall 모델 출력을 후처리로 override 하기 위해 사용:
      - 진짜 넘어진 사람: bbox 가로 > 세로, 머리-발 거의 같은 높이
      - 서있거나 책상 작업 (상체만 숙임): bbox 세로 김, 다리는 수직 유지

    return: True → 서있음 (fall 오인). False → 진짜 누운/엎드림.
    """
    x1, y1, x2, y2 = box[:4]
    h = max(y2 - y1, 1.0)
    w = max(x2 - x1, 1.0)

    # 조건 1: bbox 세로 비율 (서있으면 세로가 가로보다 김)
    aspect_h_w = h / w
    if aspect_h_w >= 1.5:
        return True       # 분명히 세로형 → 서있음

    # 조건 2: 머리와 발 사이 수직 거리가 bbox 의 50% 이상
    #         (책상에서 몸을 깊이 숙여 bbox 가 정사각형 가까워도, 머리/발 분리되어 있으면 서있는 자세)
    nose_kp = keypoints[0]   if keypoints[0, 2]  > 0.3 else None
    lank_kp = keypoints[15]  if keypoints[15, 2] > 0.3 else None
    rank_kp = keypoints[16]  if keypoints[16, 2] > 0.3 else None
    if nose_kp is not None and (lank_kp is not None or rank_kp is not None):
        foot_ys = []
        if lank_kp is not None: foot_ys.append(float(lank_kp[1]))
        if rank_kp is not None: foot_ys.append(float(rank_kp[1]))
        foot_y = sum(foot_ys) / len(foot_ys)
        head_foot_vertical = abs(foot_y - float(nose_kp[1]))
        if head_foot_vertical >= h * 0.5:
            return True   # 머리-발 수직 분리 충분 → 서있음

    return False


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    xx1 = max(ax1, bx1); yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2); yy2 = min(ay2, by2)
    if xx2 <= xx1 or yy2 <= yy1: return 0.0
    inter = (xx2 - xx1) * (yy2 - yy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1e-6)


def _match_tracks_to_detections(track_out, detections):
    tb = track_out['track_bboxes']
    if not tb: return []
    tracks = tb[0] if isinstance(tb, list) else tb
    if tracks is None or len(tracks) == 0: return []
    tracks = np.asarray(tracks)
    if tracks.ndim == 1: tracks = tracks[None, :]
    results = []; used = set()
    for row in tracks:
        if len(row) < 6: continue
        # smtrack ByteTrackerRunner 출력: [track_id, x1, y1, x2, y2, score]
        tid, tx1, ty1, tx2, ty2, score = row[:6]
        best_iou, best_idx = 0.0, -1
        for di, det in enumerate(detections):
            if di in used: continue
            iou = _iou_xyxy([tx1, ty1, tx2, ty2], det['box'][:4])
            if iou > best_iou: best_iou, best_idx = iou, di
        if best_idx >= 0 and best_iou > 0.3:
            used.add(best_idx)
            results.append({
                'track_id': int(tid),
                'box': detections[best_idx]['box'],
                'keypoints': detections[best_idx]['keypoints'],
            })
    return results


def _draw_tracks(out, tracked, kp_buffers, last_action_result):
    now = time.time()
    for tr in tracked:
        tid = tr['track_id']
        box = tr['box']
        kp = tr['keypoints']
        x1, y1, x2, y2 = box.astype(int)[:4].tolist()
        conf = float(box[4])
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        for kx, ky, kc in kp:
            if kc > 0.3:
                cv2.circle(out, (int(kx), int(ky)), 4, (0, 255, 0), -1)
        for si, (a, b) in enumerate(SKELETON):
            if kp[a][2] > 0.3 and kp[b][2] > 0.3:
                cv2.line(out, (int(kp[a][0]), int(kp[a][1])),
                         (int(kp[b][0]), int(kp[b][1])), SKELETON_COLORS[si], 2)
        label = f"ID:{tid} | {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        res = last_action_result.get(tid)
        if res is None:
            buf_len = len(kp_buffers[tid])
            lines = [(f"버퍼 {buf_len}/{NUM_FRAMES}", (160, 160, 160))]
        else:
            mt, ts = res
            stale = now - ts > ACTION_DISPLAY_TTL
            # 후처리: action_lower 가 fall 인데 직립 자세면 standing 으로 override
            lower_name, lower_conf = mt.get('action_lower', ('?', 0.0))
            if lower_name == 'fall' and _is_upright_pose(box, kp):
                lower_name = 'standing(post)'
                mt['action_lower'] = (lower_name, lower_conf)

            # 5 head 라벨: "헤드한글 한글(영어) NN%" 형식
            lines = []
            for head in MT_HEAD_ORDER:
                cname_en, conf = mt.get(head, ('?', 0.0))
                # 영문 idx 찾아서 한글로 매핑
                en_list = MT_CLASSES.get(head, [])
                kr_list = MT_CLASSES_KR.get(head, [])
                try:
                    idx = en_list.index(cname_en)
                    cname_kr = kr_list[idx] if idx < len(kr_list) else cname_en
                except ValueError:
                    cname_kr = cname_en
                head_kr = HEAD_LABEL_KR.get(head, head)
                txt = f"{head_kr} {cname_kr}({cname_en}) {conf*100:.0f}%"
                # 색상: fall 이면 빨강, none 이면 회색, 그 외 정상색
                if cname_en in ('fall',):
                    color = FALL_COLOR
                elif cname_en == 'none':
                    color = DIM_COLOR
                else:
                    color = NORMAL_COLOR
                if stale:
                    color = tuple(c // 2 for c in color)
                lines.append((txt, color))

        # PIL 로 한글 multi-line 렌더링 (BGR 색상 그대로 전달)
        _draw_text_kr(out, lines, x1, y2, font_size=14, line_gap=3, bg=(0, 0, 0))


def _draw_status_bar(frame, fps, tracks, frames, pose_util=0.0, act_util=0.0):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    info = (f"FPS:{fps:.1f} | Tracks:{tracks} | Frames:{frames} | "
            f"NPU0(Pose):{pose_util:.1f}% | NPU1(Act):{act_util:.2f}%")
    cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)


# ============================================================
# MJPEG HTTP Server (main process)
# ============================================================
class MJPEGHandler(BaseHTTPRequestHandler):
    shm_anno = None
    shm_meta = None
    status_dict = None
    npu_stats = None

    def _get_jpeg(self):
        meta_buf = np.ndarray((4,), dtype=np.int64, buffer=self.shm_meta.buf)
        W, H = int(meta_buf[0]), int(meta_buf[1])
        ann_id = int(meta_buf[3])
        if W == 0 or H == 0 or ann_id == 0:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Waiting...", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, jpg = cv2.imencode('.jpg', ph)
            return jpg.tobytes()
        anno_buf = np.ndarray((SHM_MAX_H, SHM_MAX_W, 3), dtype=np.uint8, buffer=self.shm_anno.buf)
        frame = anno_buf[:H, :W].copy()
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpg.tobytes()

    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    jpg = self._get_jpeg()
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n'.encode())
                    self.wfile.write(b'\r\n'); self.wfile.write(jpg); self.wfile.write(b'\r\n')
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == '/snapshot':
            jpg = self._get_jpeg()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpg)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path == '/health':
            self.send_response(200); self.end_headers()
            self.wfile.write(b'ok')
        elif self.path == '/status':
            import json
            data = dict(self.status_dict) if self.status_dict else {}
            if self.npu_stats:
                ns = dict(self.npu_stats)
                # 평균 추론 시간 + 누적 횟수 → 활용률 계산
                #   utilization = (총 추론 시간 / wall-clock 누적 시간) × 100
                #   여기선 단순화: pose 평균 ms × 초당 inference 수 = ms/sec → / 10 = %
                if ns.get('pose_count', 0) > 0:
                    data['npu0_pose_avg_ms'] = round(ns.get('pose_last_avg_ms', 0), 2)
                    data['npu0_pose_total_inferences'] = ns['pose_count']
                if ns.get('act_count', 0) > 0:
                    data['npu1_action_avg_ms'] = round(ns.get('act_last_avg_ms', 0), 2)
                    data['npu1_action_total_inferences'] = ns['act_count']
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, fmt, *args):
        pass


class ThreadingMJPEGServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rtsp", required=True)
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--pose-hef", default=os.environ.get('POSE_HEF_PATH', '/app/models/yolov8m_pose.hef'))
    p.add_argument("--action-hef", default=os.environ.get('ACTION_HEF_PATH', '/app/models/action_resnet_mt.hef'))
    args = p.parse_args()

    mp.set_start_method('spawn', force=True)

    # 공유 메모리
    shm_frame = _alloc_shm(SHM_FRAME_NAME, SHM_BYTES)
    shm_anno  = _alloc_shm(SHM_ANNO_NAME, SHM_BYTES)
    shm_meta  = _alloc_shm(SHM_META_NAME, 4 * 8)   # 4 × int64

    # 메타 초기화 (zero)
    meta_init = np.ndarray((4,), dtype=np.int64, buffer=shm_meta.buf)
    meta_init[:] = 0

    stop_event = mp.Event()
    det_queue = mp.Queue(maxsize=4)
    manager = mp.Manager()
    status_dict = manager.dict()
    npu_stats = manager.dict()      # NPU 추론 시간/회수 통계

    # 워커 프로세스 시작
    pose_p = mp.Process(
        name='PoseWorker',
        target=pose_worker,
        args=(args.pose_hef, SHM_FRAME_NAME, SHM_META_NAME, det_queue, stop_event,
              npu_stats),
    )
    action_p = mp.Process(
        name='ActionDrawWorker',
        target=action_draw_worker,
        args=(args.action_hef, det_queue, SHM_ANNO_NAME, SHM_META_NAME,
              stop_event, status_dict, npu_stats),
    )

    pose_p.start()
    action_p.start()

    # RTSP reader thread (main process 내)
    reader_t = threading.Thread(
        target=reader_thread,
        args=(args.rtsp, shm_frame, shm_meta, stop_event),
        daemon=True,
    )
    reader_t.start()

    # MJPEG server
    MJPEGHandler.shm_anno = shm_anno
    MJPEGHandler.shm_meta = shm_meta
    MJPEGHandler.status_dict = status_dict
    MJPEGHandler.npu_stats = npu_stats
    server = ThreadingMJPEGServer(("0.0.0.0", args.port), MJPEGHandler)

    print()
    print("=" * 60)
    print("  NPU Stream Viewer (Multi-Process)")
    print(f"  RTSP     : {args.rtsp}")
    print(f"  Pose HEF : {args.pose_hef}")
    print(f"  Action HEF: {args.action_hef}")
    print(f"  MJPEG    : http://0.0.0.0:{args.port}/stream")
    print(f"  Snapshot : http://0.0.0.0:{args.port}/snapshot")
    print(f"  Status   : http://0.0.0.0:{args.port}/status")
    print(f"  Arch     : reader(main) + pose_worker + action_draw_worker")
    print("=" * 60)
    print()

    def shutdown(*_):
        logger.info("Shutting down...")
        stop_event.set()
        server.shutdown()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        server.serve_forever()
    finally:
        stop_event.set()
        pose_p.join(timeout=5)
        action_p.join(timeout=5)
        if pose_p.is_alive(): pose_p.terminate()
        if action_p.is_alive(): action_p.terminate()
        shm_frame.close(); shm_frame.unlink()
        shm_anno.close();  shm_anno.unlink()
        shm_meta.close();  shm_meta.unlink()
        print("Stopped.")


if __name__ == "__main__":
    main()
