"""Single-process stream viewer — 단일 NPU 보드 (Pi5 Hailo-8L) 용.

하나의 process 에서:
  Reader thread → main loop (pose 추론 → action 추론 → draw) → MJPEG server thread

VDevice 1개에 pose HEF + action HEF 모두 configure → NPU 충돌 없음.
"""
import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Empty
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
# Multi-task 클래스 정의 (5 head)
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
FC_TO_HEAD = {'fc1': 'action_lower', 'fc2': 'action_upper',
              'fc3': 'foot', 'fc4': 'hand', 'fc5': 'pose'}

FALL_COLOR = (0, 0, 255)
NORMAL_COLOR = (0, 255, 0)
DIM_COLOR = (180, 180, 180)

NUM_FRAMES = 60
SLIDING_STRIDE = 8
ACTION_DISPLAY_TTL = 2.0
POSE_CONF_THR = float(os.environ.get('POSE_CONF_THR', '0.1'))

# 한글 폰트
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
    box_w = max_w + 8
    box_h = len(lines) * line_h + 4
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0))
    for i, (t, c) in enumerate(lines):
        draw.text((x + 4, y + i*line_h + 2), t, font=font, fill=(c[2], c[1], c[0]))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
TRACK_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),
                (255,0,255),(0,255,255),(128,255,0),(255,128,0)]


# ============================================================
# Reader thread
# ============================================================
class FrameSlot:
    """latest-only frame slot — lock 으로 최신 1프레임만 보관."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._fid = 0

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._fid += 1

    def get(self):
        with self._lock:
            return (None, 0) if self._frame is None else (self._frame.copy(), self._fid)


def reader_thread(rtsp_url, slot: FrameSlot, stop_event):
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
    logger.info("Reader exit.")


# ============================================================
# MJPEG server
# ============================================================
class ThreadingMJPEGServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MJPEGHandler(BaseHTTPRequestHandler):
    out_slot = None
    status = None

    def log_message(self, *a, **kw): pass

    def _jpeg(self):
        frame, _ = self.out_slot.get()
        if frame is None:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Waiting...", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, jpg = cv2.imencode('.jpg', ph)
            return jpg.tobytes()
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
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == '/snapshot':
            jpg = self._jpeg()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpg)))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers(); self.wfile.write(jpg)
        elif self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        else:
            self.send_response(404); self.end_headers()


def http_thread(port, out_slot, status, stop_event):
    MJPEGHandler.out_slot = out_slot
    MJPEGHandler.status = status
    server = ThreadingMJPEGServer(("0.0.0.0", port), MJPEGHandler)
    logger.info(f"MJPEG: http://0.0.0.0:{port}/stream")
    server.serve_forever()


# ============================================================
# 추론 헬퍼
# ============================================================
def _model_in_hw(input_info):
    s = tuple(input_info.shape)
    if len(s) == 3: return s[0], s[1]
    if len(s) == 4:
        if s[1] in (3, 1): return s[2], s[3]
        return s[1], s[2]
    return 640, 640


def _iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xx1=max(ax1,bx1); yy1=max(ay1,by1); xx2=min(ax2,bx2); yy2=min(ay2,by2)
    if xx2<=xx1 or yy2<=yy1: return 0.0
    inter = (xx2-xx1)*(yy2-yy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1e-6)


def _is_upright(box, kp):
    x1,y1,x2,y2 = box[:4]
    h = max(y2-y1, 1.0); w = max(x2-x1, 1.0)
    if h/w >= 1.5: return True
    nose = kp[0] if kp[0,2]>0.3 else None
    la = kp[15] if kp[15,2]>0.3 else None
    ra = kp[16] if kp[16,2]>0.3 else None
    if nose is not None and (la is not None or ra is not None):
        ys = []
        if la is not None: ys.append(float(la[1]))
        if ra is not None: ys.append(float(ra[1]))
        if abs(sum(ys)/len(ys) - float(nose[1])) >= h*0.5: return True
    return False


# ============================================================
# Main loop
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rtsp", required=True)
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--pose-hef", default='/home/safemotion/action_recognition_npu/models/yolov8s_pose_h8l.hef')
    p.add_argument("--action-hef", default='/home/safemotion/action_recognition_npu/models/action_resnet_mt_h8l.hef')
    args = p.parse_args()

    from hailo_platform import (VDevice, HEF, ConfigureParams,
                                 HailoStreamInterface, InputVStreamParams,
                                 OutputVStreamParams, FormatType, InferVStreams,
                                 HailoSchedulingAlgorithm)
    from pose_extractor import postprocess_pose_multi
    from pseudo_image import keypoints_to_pseudo_image
    from smtrack.builder.runner_builder import ByteTrackerRunner

    # mmengine.Config 대용 — addict.Dict 가 동일 attribute 접근 지원
    from addict import Dict as Config

    # === 단일 VDevice + scheduler — 두 HEF 자동 시분할 ===
    vparams = VDevice.create_params()
    vparams.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdevice = VDevice(params=vparams)
    pose_hef = HEF(args.pose_hef)
    action_hef = HEF(args.action_hef)
    pose_cfg = ConfigureParams.create_from_hef(hef=pose_hef, interface=HailoStreamInterface.PCIe)
    action_cfg = ConfigureParams.create_from_hef(hef=action_hef, interface=HailoStreamInterface.PCIe)
    pose_ng = vdevice.configure(pose_hef, pose_cfg)[0]
    action_ng = vdevice.configure(action_hef, action_cfg)[0]

    pose_in_info = list(pose_hef.get_input_vstream_infos())[0]
    pose_out_infos = list(pose_hef.get_output_vstream_infos())
    action_in_info = list(action_hef.get_input_vstream_infos())[0]
    action_out_infos = list(action_hef.get_output_vstream_infos())

    pose_in_params = InputVStreamParams.make(pose_ng, format_type=FormatType.FLOAT32)
    pose_out_params = OutputVStreamParams.make(pose_ng, format_type=FormatType.FLOAT32)
    action_in_params = InputVStreamParams.make(action_ng, format_type=FormatType.FLOAT32)
    action_out_params = OutputVStreamParams.make(action_ng, format_type=FormatType.FLOAT32)
    pose_ng_params = pose_ng.create_params()
    action_ng_params = action_ng.create_params()

    pose_in_h, pose_in_w = _model_in_hw(pose_in_info)
    action_in_name = action_in_info.name
    pose_in_name = pose_in_info.name

    # head → vstream
    head_to_vs = {}
    for o in action_out_infos:
        head = FC_TO_HEAD.get(o.name.rsplit('/', 1)[-1])
        if head: head_to_vs[head] = o.name
    logger.info(f"action head→vstream: {head_to_vs}")
    logger.info(f"pose inputs ({pose_in_h}x{pose_in_w}) outs={len(pose_out_infos)}")

    # tracker
    tracker_cfg = Config(dict(
        type="ByteTracker",
        obj_score_thrs=dict(high=0.4, low=0.1),
        init_track_thr=0.5,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        use_cate_match=True, use_second_match_case=True, num_frames_retain=30,
    ))
    motion_cfg = Config(dict(type="KalmanFilter"))
    tracker = ByteTrackerRunner(tracker=tracker_cfg, motion=motion_cfg)

    # === Reader + HTTP ===
    in_slot = FrameSlot(); out_slot = FrameSlot()
    stop = threading.Event()
    status = {}
    threading.Thread(target=reader_thread,
                     args=(args.rtsp, in_slot, stop), daemon=True).start()
    threading.Thread(target=http_thread,
                     args=(args.port, out_slot, status, stop), daemon=True).start()

    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    # === Main loop ===
    kp_buffers = defaultdict(lambda: deque(maxlen=NUM_FRAMES))
    last_action_frame = defaultdict(int)
    last_action_result = {}
    fps_window = deque(maxlen=30); t_prev = time.time()
    last_fid = 0
    # NPU 사용량 추적
    pose_t_sum = 0.0; pose_count = 0; pose_last_ms = 0.0
    act_t_sum = 0.0; act_count = 0; act_last_ms = 0.0
    t_start = time.time()

    # scheduler 모드: activate() 호출 안 함 — InferVStreams 만으로 스케줄러가 자동 활성화
    with InferVStreams(pose_ng, pose_in_params, pose_out_params) as pose_pipe, \
         InferVStreams(action_ng, action_in_params, action_out_params) as action_pipe:
        logger.info("NPU ready (pose + action on single VDevice via scheduler)")
        while not stop.is_set():
            frame, fid = in_slot.get()
            if frame is None or fid == last_fid:
                time.sleep(0.01); continue
            last_fid = fid
            H, W = frame.shape[:2]

            # --- Pose 추론 (시간 측정) ---
            img = cv2.resize(frame, (pose_in_w, pose_in_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            _tp0 = time.time()
            p_out = pose_pipe.infer({pose_in_name: img[None]})
            pose_dt = time.time() - _tp0
            pose_t_sum += pose_dt; pose_count += 1
            if pose_count % 30 == 0:
                pose_last_ms = (pose_t_sum / pose_count) * 1000
            named = {n: p_out[n] for n in p_out}
            dets = postprocess_pose_multi(named, H, W, conf_thr=POSE_CONF_THR)

            # tracker (torch tensor 기대)
            if dets:
                import torch as _t
                bboxes = _t.from_numpy(np.array([d['box'][:5] for d in dets], dtype=np.float32))
                cates = _t.zeros((len(dets),), dtype=_t.int64)
                track_out = tracker.run_tracker(det_bboxes=bboxes, det_labels=cates, frame_id=fid)
                tb = track_out.get('track_bboxes', [])
                tracks_arr = tb[0] if isinstance(tb, list) and tb else (tb if not isinstance(tb, list) else None)
                tracks_arr = np.asarray(tracks_arr) if tracks_arr is not None else None
                if tracks_arr is not None and tracks_arr.ndim == 1:
                    tracks_arr = tracks_arr[None, :]
                tracked = []
                used = set()
                if tracks_arr is not None:
                    for row in tracks_arr:
                        if len(row) < 6: continue
                        tid, tx1, ty1, tx2, ty2, score = row[:6]
                        best_i, best_iou = -1, 0.0
                        for di, d in enumerate(dets):
                            if di in used: continue
                            iv = _iou([tx1,ty1,tx2,ty2], d['box'][:4])
                            if iv > best_iou: best_iou, best_i = iv, di
                        if best_i >= 0 and best_iou > 0.3:
                            used.add(best_i)
                            tracked.append({'track_id': int(tid),
                                            'box': dets[best_i]['box'],
                                            'keypoints': dets[best_i]['keypoints']})
            else:
                tracked = []

            # --- Action 추론 per track ---
            for tr in tracked:
                tid = tr['track_id']
                kp_buffers[tid].append(tr['keypoints'])
                if len(kp_buffers[tid]) == NUM_FRAMES \
                        and fid - last_action_frame[tid] >= SLIDING_STRIDE:
                    last_action_frame[tid] = fid
                    kp_seq = np.stack(list(kp_buffers[tid]), axis=0)
                    pseudo = keypoints_to_pseudo_image(kp_seq, frame_w=W, frame_h=H)
                    if pseudo.shape[1] == 7:
                        pseudo = np.transpose(pseudo, (0, 2, 3, 1)).astype(np.float32)
                    pseudo = np.ascontiguousarray(pseudo)
                    _ta0 = time.time()
                    a_out = action_pipe.infer({action_in_name: pseudo})
                    act_dt = time.time() - _ta0
                    act_t_sum += act_dt; act_count += 1
                    if act_count % 30 == 0:
                        act_last_ms = (act_t_sum / act_count) * 1000
                    mt = {}
                    for head_name in MT_HEAD_ORDER:
                        vs = head_to_vs.get(head_name)
                        if vs is None: continue
                        logits = np.asarray(a_out[vs]).reshape(-1)
                        e = np.exp(logits - logits.max()); probs = e/e.sum()
                        cidx = int(probs.argmax())
                        en = MT_CLASSES[head_name][cidx] if cidx < len(MT_CLASSES[head_name]) else f"cls{cidx}"
                        mt[head_name] = (en, float(probs[cidx]))
                    last_action_result[tid] = (mt, time.time())

            # --- Draw ---
            out = frame.copy()
            now = time.time()
            for tr in tracked:
                tid = tr['track_id']
                box = tr['box']; kp = tr['keypoints']
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

                # 5 head 라벨
                res = last_action_result.get(tid)
                if res is None:
                    lines = [(f"버퍼 {len(kp_buffers[tid])}/{NUM_FRAMES}", (160,160,160))]
                else:
                    mt, ts = res
                    stale = now - ts > ACTION_DISPLAY_TTL
                    lname, lconf = mt.get('action_lower', ('?', 0.0))
                    if lname == 'fall' and _is_upright(box, kp):
                        mt['action_lower'] = ('standing(post)', lconf)
                    lines = []
                    for head in MT_HEAD_ORDER:
                        en, cf = mt.get(head, ('?', 0.0))
                        idx = MT_CLASSES[head].index(en) if en in MT_CLASSES[head] else -1
                        kr = MT_CLASSES_KR[head][idx] if 0 <= idx < len(MT_CLASSES_KR[head]) else en
                        head_kr = HEAD_LABEL_KR[head]
                        txt = f"{head_kr} {kr}({en}) {cf*100:.0f}%"
                        if en == 'fall': col = FALL_COLOR
                        elif en == 'none': col = DIM_COLOR
                        else: col = NORMAL_COLOR
                        if stale: col = tuple(c//2 for c in col)
                        lines.append((txt, col))
                _draw_text_kr(out, lines, x1, y2)

            # status bar
            now = time.time()
            dt = now - t_prev; t_prev = now
            if dt > 0: fps_window.append(1.0/dt)
            fps = float(np.mean(fps_window)) if fps_window else 0.0
            status['fps'] = fps; status['tracks'] = len(tracked); status['frame'] = fid
            # NPU 사용량 계산
            elapsed = max(time.time() - t_start, 1.0)
            pose_util = (pose_count / elapsed) * pose_last_ms / 10.0 if pose_last_ms else 0.0
            act_util = (act_count / elapsed) * act_last_ms / 10.0 if act_last_ms else 0.0
            status['npu_pose'] = round(pose_util, 1)
            status['npu_act'] = round(act_util, 2)

            ov = out.copy()
            cv2.rectangle(ov, (0,0), (W, 40), (0,0,0), -1)
            cv2.addWeighted(ov, 0.7, out, 0.3, 0, out)
            info = (f"FPS:{fps:.1f} | Tracks:{len(tracked)} | Frames:{fid} | "
                    f"NPU(Pose):{pose_util:.1f}% | NPU(Act):{act_util:.2f}%")
            cv2.putText(out, info, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            out_slot.put(out)

    stop.set()
    logger.info("main loop exit.")


if __name__ == '__main__':
    main()
