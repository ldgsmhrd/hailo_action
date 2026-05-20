"""CPU-only stream viewer — NPU 없이 ultralytics YOLOv8n-pose + ONNX action 모델.

Pi5 CPU 만으로 동작 — NPU 미사용 테스트 용.

성능 목표:
  - yolov8n-pose @ 320×320 : 10~15 FPS (CPU)
  - action_resnet_mt        : 매 sliding_stride 8 frame 마다 (가벼움)
"""
import argparse
import logging
import os
import sys
import signal
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_REPO, 'src'), _REPO, os.path.expanduser('~')):
    if os.path.isdir(p):
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 클래스 정의 (멀티태스크와 동일)
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
MT_CLASSES_KR = {
    'action_upper': ['없음', '펀치', '손흔들기', '손뼉치기', '손올리기', '손내리기'],
    'action_lower': ['없음', '서성이기', '걷기', '달리기', '점프-제자리', '넘어짐',
                     '킥', '점프-두발', '외발점프', '외발점프-제자리'],
    'pose':         ['바닥앉기', '의자앉기', '무릎꿇기', '무릎서기',
                     '서있기', '허리구부리기', '누워있기', '무릎기기', '기타'],
    'hand':         ['없음', '팔짱끼기', '양팔들기'],
    'foot':         ['없음', '다리꼬기', '한쪽다리들기'],
}
HEAD_LABEL_KR = {'action_upper': '상체', 'action_lower': '하체',
                 'pose': '자세', 'hand': '손', 'foot': '발'}
MT_HEAD_ORDER = ['action_upper', 'action_lower', 'pose', 'hand', 'foot']

FALL_COLOR = (0, 0, 255)
NORMAL_COLOR = (0, 255, 0)
DIM_COLOR = (180, 180, 180)

NUM_FRAMES = 60
SLIDING_STRIDE = 8
ACTION_DISPLAY_TTL = 2.0

# 한글 폰트
_FONT_CANDIDATES = [
    os.environ.get('KOREAN_FONT_PATH'),
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf',
]
KOREAN_FONT_PATH = next((p for p in _FONT_CANDIDATES if p and os.path.exists(p)), None)
_KR_FONT_CACHE = {}


def _get_kr_font(size=14):
    if size in _KR_FONT_CACHE: return _KR_FONT_CACHE[size]
    try:
        from PIL import ImageFont
        f = ImageFont.truetype(KOREAN_FONT_PATH, size) if KOREAN_FONT_PATH else None
    except Exception:
        f = None
    _KR_FONT_CACHE[size] = f
    return f


def _draw_text_kr(frame, lines, x, y, font_size=14):
    from PIL import Image, ImageDraw
    font = _get_kr_font(font_size)
    if font is None:
        for i, (t, c) in enumerate(lines):
            cv2.putText(frame, t, (x, y + (i+1)*font_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        return
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    line_h = font_size + 3
    max_w = max(draw.textbbox((0, 0), t, font=font)[2] for t, _ in lines)
    draw.rectangle([x, y, x + max_w + 8, y + len(lines)*line_h + 4], fill=(0, 0, 0))
    for i, (t, c) in enumerate(lines):
        draw.text((x + 4, y + i*line_h + 2), t, font=font, fill=(c[2], c[1], c[0]))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
TRACK_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]


class FrameSlot:
    def __init__(self):
        self._lock = threading.Lock(); self._frame = None; self._fid = 0
    def put(self, frame):
        with self._lock:
            self._frame = frame; self._fid += 1
    def get(self):
        with self._lock:
            return (None, 0) if self._frame is None else (self._frame.copy(), self._fid)


def reader_thread(rtsp_url, slot, stop_event):
    os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS',
        'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000')
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Reader cannot open: {rtsp_url}"); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Reader started.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.release(); time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG); continue
        slot.put(frame)
    cap.release()


class ThreadingMJPEGServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True; allow_reuse_address = True


class MJPEGHandler(BaseHTTPRequestHandler):
    out_slot = None
    def log_message(self, *a, **kw): pass
    def _jpeg(self):
        frame, _ = self.out_slot.get()
        if frame is None:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Waiting...", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, jpg = cv2.imencode('.jpg', ph); return jpg.tobytes()
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
                    jpg = self._jpeg()
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n'.encode())
                    self.wfile.write(b'\r\n'); self.wfile.write(jpg); self.wfile.write(b'\r\n')
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError): pass
        elif self.path == '/snapshot':
            jpg = self._jpeg()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpg)))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers(); self.wfile.write(jpg)
        else:
            self.send_response(404); self.end_headers()


def http_thread(port, out_slot, stop_event):
    MJPEGHandler.out_slot = out_slot
    server = ThreadingMJPEGServer(("0.0.0.0", port), MJPEGHandler)
    logger.info(f"MJPEG: http://0.0.0.0:{port}/stream")
    server.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rtsp", required=True)
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--pose-imgsz", type=int, default=320,
                   help="YOLO 입력 해상도 (작을수록 빠름)")
    p.add_argument("--action-onnx", default=None,
                   help="action ONNX 경로 — 없으면 pose only 모드")
    args = p.parse_args()

    logger.info(f"CPU-only stream viewer 시작 (imgsz={args.pose_imgsz})")

    # YOLO pose — onnxruntime CPU (ultralytics 의존성 회피)
    import onnxruntime as ort
    pose_onnx_path = os.environ.get(
        'YOLO_POSE_ONNX',
        os.path.expanduser('~/action_recognition_npu/models/yolov8n-pose.onnx')
    )
    pose_sess = ort.InferenceSession(
        pose_onnx_path,
        providers=['CPUExecutionProvider'],
        sess_options=None,
    )
    pose_in_name = pose_sess.get_inputs()[0].name
    pose_out_name = pose_sess.get_outputs()[0].name
    pose_in_shape = pose_sess.get_inputs()[0].shape  # [1, 3, 320, 320]
    POSE_IN = pose_in_shape[2]
    logger.info(f"yolov8n-pose ONNX: in={pose_in_shape} out={pose_out_name}")

    # action ONNX (옵션)
    action_sess = None
    if args.action_onnx and os.path.exists(args.action_onnx):
        import onnxruntime as ort
        action_sess = ort.InferenceSession(args.action_onnx,
                                            providers=['CPUExecutionProvider'])
        action_in_name = action_sess.get_inputs()[0].name
        action_out_names = [o.name for o in action_sess.get_outputs()]
        logger.info(f"Action ONNX loaded: outs={action_out_names}")
        try:
            from pseudo_image import keypoints_to_pseudo_image
        except ImportError:
            keypoints_to_pseudo_image = None
            logger.warning("pseudo_image not available, action disabled")
            action_sess = None
    else:
        keypoints_to_pseudo_image = None
        logger.info("Action 비활성 (--action-onnx 미지정)")

    in_slot = FrameSlot(); out_slot = FrameSlot()
    stop = threading.Event()
    threading.Thread(target=reader_thread, args=(args.rtsp, in_slot, stop), daemon=True).start()
    threading.Thread(target=http_thread, args=(args.port, out_slot, stop), daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    kp_buffers = defaultdict(lambda: deque(maxlen=NUM_FRAMES))
    last_action_frame = defaultdict(int)
    last_action_result = {}
    fps_window = deque(maxlen=30); t_prev = time.time(); last_fid = 0
    pose_t_sum = 0.0; pose_count = 0; pose_last_ms = 0.0
    act_t_sum = 0.0; act_count = 0; act_last_ms = 0.0

    # simple track id assignment (no tracker — just persistent index by box overlap)
    next_tid = 1
    prev_tracks = []  # [(tid, bbox)]
    def assign_tid(box):
        nonlocal next_tid
        best_tid, best_iou = -1, 0.0
        for tid, pb in prev_tracks:
            ix1=max(box[0],pb[0]); iy1=max(box[1],pb[1])
            ix2=min(box[2],pb[2]); iy2=min(box[3],pb[3])
            if ix2 <= ix1 or iy2 <= iy1: continue
            inter = (ix2-ix1)*(iy2-iy1)
            ua = (box[2]-box[0])*(box[3]-box[1]) + (pb[2]-pb[0])*(pb[3]-pb[1]) - inter
            iou = inter / max(ua, 1e-6)
            if iou > best_iou: best_iou, best_tid = iou, tid
        if best_iou > 0.3: return best_tid
        next_tid += 1; return next_tid - 1

    while not stop.is_set():
        frame, fid = in_slot.get()
        if frame is None or fid == last_fid:
            time.sleep(0.01); continue
        last_fid = fid
        H, W = frame.shape[:2]

        # === YOLO pose 추론 (onnxruntime) ===
        # preprocess: letterbox → BGR→RGB → CHW → [0,1] normalize
        scale = POSE_IN / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((POSE_IN, POSE_IN, 3), dtype=np.uint8)
        pad_x = (POSE_IN - new_w) // 2; pad_y = (POSE_IN - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]  # [1, 3, H, W]

        _t0 = time.time()
        p_out = pose_sess.run([pose_out_name], {pose_in_name: inp})[0]  # [1, 56, N]
        pose_dt = time.time() - _t0
        pose_t_sum += pose_dt; pose_count += 1
        if pose_count % 10 == 0:
            pose_last_ms = (pose_t_sum / pose_count) * 1000

        # post-process YOLOv8 pose output
        preds = p_out[0].T  # [N, 56]: 4 bbox(cxcywh) + 1 conf + 17×3 kp
        confs = preds[:, 4]
        mask = confs > 0.3
        preds = preds[mask]
        tracked = []
        new_prev = []
        if len(preds) > 0:
            cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
            boxes = np.stack([x1, y1, x2, y2], axis=-1)
            scores = preds[:, 4]
            kps = preds[:, 5:].reshape(-1, 17, 3)
            # letterbox → original 좌표 복원
            boxes[:, [0,2]] -= pad_x; boxes[:, [1,3]] -= pad_y
            boxes /= scale
            kps[..., 0] -= pad_x; kps[..., 1] -= pad_y
            kps[..., :2] /= scale
            # 간단 NMS
            keep_idx = []
            order = scores.argsort()[::-1]
            while len(order) > 0:
                i = order[0]; keep_idx.append(i)
                if len(order) == 1: break
                rest = order[1:]
                xx1 = np.maximum(boxes[i,0], boxes[rest,0])
                yy1 = np.maximum(boxes[i,1], boxes[rest,1])
                xx2 = np.minimum(boxes[i,2], boxes[rest,2])
                yy2 = np.minimum(boxes[i,3], boxes[rest,3])
                iw = np.maximum(0, xx2-xx1); ih = np.maximum(0, yy2-yy1)
                inter = iw*ih
                area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
                area_r = (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])
                iou = inter / (area_i + area_r - inter + 1e-6)
                order = rest[iou < 0.45]
            for i in keep_idx:
                box5 = np.array([*boxes[i], float(scores[i])], dtype=np.float32)
                tid = assign_tid(boxes[i])
                tracked.append({'track_id': tid, 'box': box5, 'keypoints': kps[i]})
                new_prev.append((tid, boxes[i]))
        prev_tracks = new_prev

        # === Action 추론 (옵션) ===
        for tr in tracked:
            tid = tr['track_id']
            kp_buffers[tid].append(tr['keypoints'])
            if (action_sess is not None and keypoints_to_pseudo_image is not None
                    and len(kp_buffers[tid]) == NUM_FRAMES
                    and fid - last_action_frame[tid] >= SLIDING_STRIDE):
                last_action_frame[tid] = fid
                kp_seq = np.stack(list(kp_buffers[tid]), axis=0)
                pseudo = keypoints_to_pseudo_image(kp_seq, frame_w=W, frame_h=H)
                if pseudo.ndim == 4 and pseudo.shape[1] == 7:
                    pseudo_nchw = pseudo.astype(np.float32)
                else:
                    pseudo_nchw = pseudo.astype(np.float32)
                _ta0 = time.time()
                a_out = action_sess.run(action_out_names, {action_in_name: pseudo_nchw})
                act_dt = time.time() - _ta0
                act_t_sum += act_dt; act_count += 1
                if act_count % 5 == 0:
                    act_last_ms = (act_t_sum / act_count) * 1000
                # head index → name (assume ONNX 출력 순서: upper, lower, pose, hand, foot)
                mt = {}
                for hi, head in enumerate(MT_HEAD_ORDER):
                    if hi >= len(a_out): break
                    logits = np.asarray(a_out[hi]).reshape(-1)
                    e = np.exp(logits - logits.max()); probs = e/e.sum()
                    cidx = int(probs.argmax())
                    en = MT_CLASSES[head][cidx] if cidx < len(MT_CLASSES[head]) else f"cls{cidx}"
                    mt[head] = (en, float(probs[cidx]))
                last_action_result[tid] = (mt, time.time())

        # === Draw ===
        out = frame.copy()
        now = time.time()
        for tr in tracked:
            tid = tr['track_id']; box = tr['box']; kp = tr['keypoints']
            x1,y1,x2,y2 = box.astype(int)[:4].tolist()
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            for kx,ky,kc in kp:
                if kc > 0.3:
                    cv2.circle(out, (int(kx),int(ky)), 4, (0,255,0), -1)
            for a,b in SKELETON:
                if kp[a][2] > 0.3 and kp[b][2] > 0.3:
                    cv2.line(out, (int(kp[a][0]),int(kp[a][1])),
                             (int(kp[b][0]),int(kp[b][1])), (255,255,255), 2)
            lbl = f"ID:{tid} | {float(box[4]):.2f}"
            cv2.putText(out, lbl, (x1+2, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            res = last_action_result.get(tid)
            if res is None:
                if action_sess is None:
                    continue   # action 비활성 모드: 라벨 없음
                lines = [(f"버퍼 {len(kp_buffers[tid])}/{NUM_FRAMES}", (160,160,160))]
            else:
                mt, ts = res
                stale = now - ts > ACTION_DISPLAY_TTL
                lines = []
                for head in MT_HEAD_ORDER:
                    en, cf = mt.get(head, ('?', 0.0))
                    try:
                        idx = MT_CLASSES[head].index(en); kr = MT_CLASSES_KR[head][idx]
                    except ValueError:
                        kr = en
                    txt = f"{HEAD_LABEL_KR[head]} {kr}({en}) {cf*100:.0f}%"
                    col = FALL_COLOR if en == 'fall' else (DIM_COLOR if en == 'none' else NORMAL_COLOR)
                    if stale: col = tuple(c//2 for c in col)
                    lines.append((txt, col))
            _draw_text_kr(out, lines, x1, y2)

        # status bar
        dt = now - t_prev; t_prev = now
        if dt > 0: fps_window.append(1.0/dt)
        fps = float(np.mean(fps_window)) if fps_window else 0.0
        ov = out.copy()
        cv2.rectangle(ov, (0,0), (W, 40), (0,0,0), -1)
        cv2.addWeighted(ov, 0.7, out, 0.3, 0, out)
        info = (f"FPS:{fps:.1f} | Tracks:{len(tracked)} | "
                f"Pose:{pose_last_ms:.0f}ms | Act:{act_last_ms:.0f}ms | "
                f"CPU-ONLY (imgsz={args.pose_imgsz})")
        cv2.putText(out, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out_slot.put(out)

    stop.set()
    logger.info("exit.")


if __name__ == '__main__':
    main()
