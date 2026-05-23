"""NTU RGB+D .skeleton 파일 파서.

NTU .skeleton 파일 포맷 (텍스트):
    line 1:   N (frame 수)
    각 frame:
        line:  M (body 수, 0~4)
        각 body:
            line:  bodyID clippedEdges handLeftConfidence handLeftState ...
                   (10 개 metadata 값)
            line:  J (joint 수, 보통 25)
            각 joint:
                line:  x y z depthX depthY colorX colorY oW oX oY oZ trackState
                       (12 개 값)

반환:
    skeleton: np.ndarray [M, T, J, 3]  (x, y, z) — 최대 2 body
    valid_frames: np.ndarray [T] bool  — 실제 body 가 있는 프레임만 True
"""

import os
import numpy as np

NUM_JOINTS = 25
MAX_BODIES = 2   # PSP-Net 실험에서는 최대 2 body


def parse_skeleton_file(path):
    """단일 .skeleton 파일 파싱 → [M, T, J, 3], [T] bool."""
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')

    cursor = 0
    n_frames = int(lines[cursor])
    cursor += 1

    # body_id → frame → 25x3 좌표 저장
    body_data = {}   # {body_id: {frame_idx: ndarray [25, 3]}}

    for frame_idx in range(n_frames):
        n_bodies = int(lines[cursor])
        cursor += 1
        for _ in range(n_bodies):
            meta = lines[cursor].split()
            body_id = int(meta[0])
            cursor += 1
            n_joints = int(lines[cursor])
            cursor += 1
            joints = np.zeros((n_joints, 3), dtype=np.float32)
            for j in range(n_joints):
                vals = lines[cursor].split()
                joints[j, 0] = float(vals[0])  # x
                joints[j, 1] = float(vals[1])  # y
                joints[j, 2] = float(vals[2])  # z
                cursor += 1
            if body_id not in body_data:
                body_data[body_id] = {}
            body_data[body_id][frame_idx] = joints

    # body 정렬: 등장 frame 수가 많은 순으로
    sorted_bodies = sorted(body_data.items(),
                           key=lambda x: -len(x[1]))[:MAX_BODIES]

    # [M, T, J, 3] 으로 모음 (없는 frame 은 0)
    M = len(sorted_bodies)
    skeleton = np.zeros((MAX_BODIES, n_frames, NUM_JOINTS, 3), dtype=np.float32)
    valid_frames = np.zeros(n_frames, dtype=bool)
    for b, (_, frames) in enumerate(sorted_bodies):
        for f, joints in frames.items():
            skeleton[b, f] = joints
            valid_frames[f] = True

    return skeleton, valid_frames


def normalize_skeleton(skeleton):
    """Spine-base (joint 0) 기준으로 평행이동, 어깨 거리로 스케일.

    skeleton: [M, T, J, 3]
    return:   [M, T, J, 3] 정규화됨
    """
    M, T, J, C = skeleton.shape
    out = skeleton.copy()
    # body 별로 valid 한 frame 만 사용해서 정규화
    for m in range(M):
        # spine base = joint 0
        # 첫 valid frame 의 spine 기준점으로 평행이동
        non_zero = np.any(out[m] != 0, axis=(1, 2))
        if not non_zero.any():
            continue
        first_f = np.where(non_zero)[0][0]
        center = out[m, first_f, 0, :].copy()       # [3]
        # 어깨간 거리 (joint 4=L-shoulder, 8=R-shoulder)
        l_sh = out[m, first_f, 4, :]
        r_sh = out[m, first_f, 8, :]
        scale = np.linalg.norm(l_sh - r_sh) + 1e-6

        out[m] = out[m] - center                     # 평행이동
        out[m] = out[m] / scale                       # 정규화
        # 원래 0 이던 곳은 다시 0 으로 (padding 보존)
        zero_mask = np.all(skeleton[m] == 0, axis=2)  # [T, J]
        out[m][zero_mask] = 0

    return out


def resample_to_T(skeleton, target_T):
    """[M, T_in, J, 3] → [M, target_T, J, 3] 시간 축 리샘플 (linear pad/sample)."""
    M, T_in, J, C = skeleton.shape
    if T_in == target_T:
        return skeleton
    if T_in < target_T:
        # zero-pad 뒤쪽
        out = np.zeros((M, target_T, J, C), dtype=skeleton.dtype)
        out[:, :T_in] = skeleton
        return out
    # T_in > target_T : 균등 샘플링
    idx = np.linspace(0, T_in - 1, target_T).astype(np.int64)
    return skeleton[:, idx]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ntu_parser.py <path/to/file.skeleton>")
        sys.exit(0)
    sk, valid = parse_skeleton_file(sys.argv[1])
    print(f"Skeleton shape: {sk.shape}  valid frames: {valid.sum()}/{len(valid)}")
    print(f"  body 0 first frame spine: {sk[0, np.where(valid)[0][0], 0]}")
    sk_norm = normalize_skeleton(sk)
    print(f"Normalized first joint mean: {sk_norm[0].mean():.4f}")
    sk_60 = resample_to_T(sk_norm, 60)
    print(f"After resample to T=60: {sk_60.shape}")
