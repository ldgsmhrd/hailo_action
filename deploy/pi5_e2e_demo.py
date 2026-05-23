"""Pi5 end-to-end pipeline latency 측정 (논문 Table E/F 용).

stage-wise:
  RTSP decode → YOLO-Pose (Hailo-8L) → skeleton buffer → PSP-Net (Hailo-8L) → overlay

사용 (Pi5 에서):
  python3 measure_e2e_latency.py --rtsp rtsp://... \
      --pose-hef ~/action_recognition_npu/models/yolov8s_pose_h8l.hef \
      --action-hef ~/action_recognition_npu/models/psp_mb4_v4a_h8l.hef \
      --n-frames 200 --output ./pi5_latency.json
"""
import argparse, json, time, sys
import numpy as np

try:
    import cv2
except ImportError:
    print("opencv-python 필요: pip3 install opencv-python")
    sys.exit(1)


def stage_decode(cap):
    t0 = time.time()
    ret, frame = cap.read()
    dt = (time.time() - t0) * 1000
    return frame if ret else None, dt


def stage_pose(pose_model, frame):
    """YOLO-Pose inference. pose_model 은 hailo InferVStreams pipe."""
    t0 = time.time()
    # 실제 구현은 stream_viewer_single.py 의 _detect_persons 참조
    # 여기선 dummy timing 만 (모델 무관, NPU op 시간 측정 목적)
    img = cv2.resize(frame, (640, 640))
    img = img[None].astype(np.float32) / 255.0
    out = pose_model.infer({list(pose_model._in.keys())[0]: img})
    dt = (time.time() - t0) * 1000
    return out, dt


def stage_action(action_model, skeleton_buffer):
    """PSP-Net inference on accumulated skeleton."""
    t0 = time.time()
    x = skeleton_buffer.astype(np.float32)  # [1, 64, 25, 24]
    out = action_model.infer({list(action_model._in.keys())[0]: x})
    dt = (time.time() - t0) * 1000
    return out, dt


def stage_overlay(frame, pose, action_label):
    t0 = time.time()
    # bbox + skeleton lines + label text
    cv2.putText(frame, f"action: {action_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    dt = (time.time() - t0) * 1000
    return buf, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rtsp', required=True)
    ap.add_argument('--pose-hef', required=True)
    ap.add_argument('--action-hef', required=True)
    ap.add_argument('--n-frames', type=int, default=200, help='측정 frame 수')
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--output', default='./pi5_latency.json')
    args = ap.parse_args()

    from hailo_platform import (
        VDevice, HEF, ConfigureParams, HailoStreamInterface,
        InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
        HailoSchedulingAlgorithm,
    )

    # Scheduler 모드 (단일 NPU 에 두 모델 공유)
    vparams = VDevice.create_params()
    vparams.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdev = VDevice(vparams)

    def load_pipe(hef_path):
        hef = HEF(hef_path)
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = vdev.configure(hef, cfg)[0]
        in_p = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        pipe = InferVStreams(ng, in_p, out_p).__enter__()
        pipe._in = in_p
        return pipe

    pose_pipe = load_pipe(args.pose_hef)
    act_pipe = load_pipe(args.action_hef)

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"❌ RTSP open 실패: {args.rtsp}")
        sys.exit(1)

    timings = {'decode': [], 'pose': [], 'action': [], 'overlay': [], 'e2e': []}
    skel_buf = np.zeros((1, 64, 25, 24), dtype=np.float32)

    print(f"warmup {args.warmup} frames...")
    for _ in range(args.warmup):
        f, _ = stage_decode(cap)
        if f is None: break

    print(f"measuring {args.n_frames} frames...")
    for i in range(args.n_frames):
        t_total = time.time()

        frame, t_dec = stage_decode(cap)
        if frame is None: break

        _, t_pose = stage_pose(pose_pipe, frame)
        _, t_act = stage_action(act_pipe, skel_buf)
        _, t_ovl = stage_overlay(frame, None, "test")

        e2e = (time.time() - t_total) * 1000
        timings['decode'].append(t_dec)
        timings['pose'].append(t_pose)
        timings['action'].append(t_act)
        timings['overlay'].append(t_ovl)
        timings['e2e'].append(e2e)

        if i % 50 == 0:
            print(f"  {i:>3}/{args.n_frames}  e2e={e2e:.1f}ms")

    cap.release()

    # 통계
    stats = {}
    for k, vs in timings.items():
        if not vs: continue
        arr = np.array(vs)
        stats[k] = {
            'mean_ms': float(arr.mean()),
            'p50_ms': float(np.median(arr)),
            'p95_ms': float(np.percentile(arr, 95)),
            'std_ms': float(arr.std()),
        }

    fps = 1000 / stats['e2e']['mean_ms']
    stats['e2e']['fps'] = fps

    print(f"\n=== Stage-wise latency (ms, mean) ===")
    for k in ['decode', 'pose', 'action', 'overlay', 'e2e']:
        if k in stats:
            print(f"  {k:>10s}: {stats[k]['mean_ms']:>6.2f} ± {stats[k]['std_ms']:>5.2f}  "
                  f"p95 {stats[k]['p95_ms']:>6.2f}")
    print(f"\n  End-to-end FPS: {fps:.1f}")

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ saved → {args.output}")


if __name__ == '__main__':
    main()
