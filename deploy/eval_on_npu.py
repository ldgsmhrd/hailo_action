"""PSP-Net HEF 의 NPU 위에서의 정확도 측정.

대상 보드:
- Hailo-8 (26 TOPS)
- Hailo-8L (13 TOPS, Raspberry Pi 5 + M.2 모듈)

사용:
  python3 eval_on_npu.py \\
      --hef psp_mb4_h8l.hef \\
      --test-x test_x.npy \\
      --test-y test_y.npy \\
      --chunk 256
"""
import argparse
import sys
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hef', required=True, help='Hailo HEF file path')
    ap.add_argument('--test-x', required=True,
                    help='Test inputs [N, T, J, C] NHWC float32 npy')
    ap.add_argument('--test-y', required=True,
                    help='Test labels [N] int64 npy')
    ap.add_argument('--chunk', type=int, default=256,
                    help='Batch chunk size (memory-limited boards: 64-256)')
    args = ap.parse_args()

    try:
        from hailo_platform import (
            VDevice, HEF, ConfigureParams, HailoStreamInterface,
            InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
        )
    except ImportError:
        print("Error: hailo_platform module not found.")
        print("Install HailoRT Python bindings first (sudo apt install hailo-all on Pi5).")
        sys.exit(1)

    print(f"HEF: {args.hef}")
    hef = HEF(args.hef)
    vdev = VDevice()
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = vdev.configure(hef, cfg)[0]
    in_p = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    out_p = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    n_in = list(in_p.keys())[0]
    n_out = list(out_p.keys())[0]

    x = np.load(args.test_x, mmap_mode='r')
    y = np.load(args.test_y)
    print(f"Test: x={x.shape}  y={y.shape}  classes={len(np.unique(y))}")

    correct = 0
    N = len(x)
    t0 = time.time()
    with network_group.activate():
        with InferVStreams(network_group, in_p, out_p) as pipe:
            for i in range(0, N, args.chunk):
                end = min(i + args.chunk, N)
                xb = np.ascontiguousarray(x[i:end].astype(np.float32))
                out = pipe.infer({n_in: xb})
                pred = np.argmax(out[n_out], axis=1)
                correct += int(np.sum(pred == y[i:end]))
                if i % (args.chunk * 8) == 0:
                    elapsed = time.time() - t0
                    print(f"  {end:>5}/{N}  acc={100*correct/end:.2f}%  ({elapsed:.0f}s)")

    dt = time.time() - t0
    acc = 100 * correct / N
    print(f"\nResult: {args.hef}")
    print(f"  Accuracy: {acc:.2f}%  ({correct}/{N})")
    print(f"  Total time: {dt:.1f}s  ({N/dt:.0f} samples/s)")


if __name__ == '__main__':
    main()
