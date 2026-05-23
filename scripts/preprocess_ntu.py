"""NTU RGB+D 60 / 120 .skeleton 파일을 학습용 .npy 로 일괄 변환.

NTU60: 56,880 개 (S001-S017)
NTU120: 114,480 개 (S001-S032)

실행:
    python scripts/preprocess_ntu.py \\
        --raw-dir data/ntu60/nturgb+d_skeletons \\
        --out-dir data/ntu60/npy \\
        --frames 64

출력 구조:
    data/ntu60/npy/{class_idx:02d}/{sample_id}.npy
    각 npy: [M=2, T=64, J=25, 3]  float32 (normalized + resampled)
"""
import argparse
import os
import sys
import glob
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from psp_net.dataset.ntu_parser import (
    parse_skeleton_file, normalize_skeleton, resample_to_T
)


# NTU 공식 GitHub 의 alignment / missing skeleton 목록 (사용 시 별도 추가)
NTU_MISSING = set()


def _process_one(args):
    src, dst, T = args
    try:
        sk, valid = parse_skeleton_file(src)
        if valid.sum() < 4:
            return ('skip_short', src)
        sk = normalize_skeleton(sk)
        sk = resample_to_T(sk, T)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst, sk.astype(np.float32))
        return ('ok', src)
    except Exception as e:
        return (f'error:{e}', src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True,
                    help='Directory of .skeleton files (NTU60 / NTU120)')
    ap.add_argument('--out-dir', required=True,
                    help='Output root for .npy files (per-class subdir)')
    ap.add_argument('--frames', type=int, default=64,
                    help='Resampled temporal length')
    ap.add_argument('--workers', type=int, default=8,
                    help='Parallel worker processes')
    ap.add_argument('--dataset', choices=['ntu60', 'ntu120'], default='ntu60',
                    help='Class index range (60 vs 120)')
    args = ap.parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"ERROR: raw directory not found: {args.raw_dir}")
        sys.exit(1)

    max_classes = 60 if args.dataset == 'ntu60' else 120

    all_files = sorted(glob.glob(os.path.join(args.raw_dir, '*.skeleton')))
    print(f"Found .skeleton files: {len(all_files)}")

    tasks = []
    for f in all_files:
        base = os.path.basename(f).replace('.skeleton', '')
        if base in NTU_MISSING:
            continue
        action_idx = int(base[17:20])    # 1..60 or 1..120
        if action_idx > max_classes:
            continue
        cls = action_idx - 1
        dst = os.path.join(args.out_dir, f"{cls:02d}", base + '.npy')
        if os.path.exists(dst):
            continue
        tasks.append((f, dst, args.frames))

    print(f"To process: {len(tasks)} (excluding already converted)")
    if not tasks:
        print("All files already converted.")
        return

    t0 = time.time()
    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = [exe.submit(_process_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futures)):
            status, src = fu.result()
            if status == 'ok':
                ok += 1
            elif status.startswith('skip'):
                skip += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  ! {status}  {os.path.basename(src)}")
            if (i + 1) % 1000 == 0:
                dt = time.time() - t0
                print(f"  progress {i+1}/{len(tasks)}  ({dt:.1f}s)")

    dt = time.time() - t0
    print(f"\nDone: ok={ok}  skip={skip}  fail={fail}  ({dt/60:.1f} min)")
    print(f"  Output: {args.out_dir}")


if __name__ == '__main__':
    main()
