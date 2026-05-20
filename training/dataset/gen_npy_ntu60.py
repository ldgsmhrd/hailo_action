"""NTU RGB+D 60 .skeleton 파일 → NPY 변환.

NTU .skeleton 파일 형식 (텍스트):
  Line 1: numFrames
  Per frame:
    Line: numBodies
    Per body:
      Line: bodyID + 9 meta values
      Line: numJoints (=25)
      Per joint: 12 float values
        x, y, z, depthX, depthY, colorX, colorY, orientW, orientX, orientY, orientZ, trackingState

본 변환은 (x, y, z) 25 joint 좌표만 추출하여
[T, 25, 3] 형태로 저장한다.
"""
import os
import sys
import glob
import re
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from configs.ntu60_config import CONFIG, PROTOCOLS


# NTU 파일명 패턴: SsssCcccPpppRrrrAaaa.skeleton
# S: setup (camera setup), C: camera (1-3), P: subject (1-40), R: replication, A: action
FILENAME_RE = re.compile(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})')


def parse_filename(fname):
    """파일명에서 메타 추출 → dict {setup, camera, subject, replication, action}."""
    m = FILENAME_RE.search(os.path.basename(fname))
    if not m:
        return None
    return {
        'setup': int(m.group(1)),
        'camera': int(m.group(2)),
        'subject': int(m.group(3)),
        'replication': int(m.group(4)),
        'action': int(m.group(5)),  # 1-60 (1-indexed)
    }


def parse_skeleton_file(fp, max_persons=1):
    """NTU .skeleton 파일 → [T, num_persons, 25, 3] 좌표 배열."""
    with open(fp) as f:
        lines = f.read().split('\n')

    idx = 0
    num_frames = int(lines[idx]); idx += 1
    out = np.zeros((num_frames, max_persons, 25, 3), dtype=np.float32)

    for t in range(num_frames):
        num_bodies = int(lines[idx]); idx += 1
        for b in range(num_bodies):
            body_meta = lines[idx].split(); idx += 1   # body meta
            num_joints = int(lines[idx]); idx += 1
            joints = []
            for j in range(num_joints):
                vals = lines[idx].split(); idx += 1
                x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                joints.append([x, y, z])
            if b < max_persons:
                out[t, b] = np.array(joints, dtype=np.float32)

    return out


def get_split(meta, protocol='cross_subject'):
    """meta → 'train' or 'test'."""
    if protocol == 'cross_subject':
        return 'train' if meta['subject'] in PROTOCOLS['cross_subject']['train_subjects'] else 'test'
    elif protocol == 'cross_view':
        return 'train' if meta['camera'] in PROTOCOLS['cross_view']['train_cameras'] else 'test'
    raise ValueError(f"unknown protocol: {protocol}")


def main():
    src_root = CONFIG['paths']['ntu_skeleton_root']
    npy_root = CONFIG['paths']['npy_root']
    split_root = CONFIG['paths']['split_root']
    protocol = CONFIG['protocol']
    T = CONFIG['frames_per_clip']
    num_persons = CONFIG['num_persons']

    skeleton_files = sorted(glob.glob(os.path.join(src_root, '*.skeleton')))
    print(f"NTU skeleton files: {len(skeleton_files)}")

    counts = {'train': 0, 'test': 0, 'failed': 0}
    for i, fp in enumerate(skeleton_files):
        if i % 500 == 0:
            print(f"  [{i}/{len(skeleton_files)}] train={counts['train']} test={counts['test']} fail={counts['failed']}",
                  flush=True)
        meta = parse_filename(fp)
        if meta is None:
            counts['failed'] += 1; continue
        action_idx = meta['action'] - 1   # 0-indexed
        if not (0 <= action_idx < 60):
            counts['failed'] += 1; continue

        try:
            kp_full = parse_skeleton_file(fp, max_persons=num_persons)
            if kp_full.shape[0] < 10:
                counts['failed'] += 1; continue

            # 시간 축 T=60 으로 정규화 (중간 잘라내기 또는 padding)
            if kp_full.shape[0] >= T:
                start = (kp_full.shape[0] - T) // 2
                kp = kp_full[start:start+T]
            else:
                pad = np.repeat(kp_full[-1:], T - kp_full.shape[0], axis=0)
                kp = np.concatenate([kp_full, pad], axis=0)

            # 단일 사람 사용 시 첫 번째 사람만
            if num_persons == 1:
                kp = kp[:, 0]   # [T, 25, 3]

            split = get_split(meta, protocol=protocol)
            out_dir = os.path.join(split_root, split, f"{action_idx:02d}")
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(fp))[0]
            np.save(os.path.join(out_dir, f"{base}.npy"), kp)
            counts[split] += 1
        except Exception as e:
            counts['failed'] += 1
            if i % 500 == 0:
                print(f"    [warn] {fp}: {e}")

    print(f"\n완료: train {counts['train']}, test {counts['test']}, failed {counts['failed']}")


if __name__ == '__main__':
    main()
