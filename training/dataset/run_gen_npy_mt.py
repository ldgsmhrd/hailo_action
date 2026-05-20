"""Multi-task NPY 생성 v3 — 5 카테고리 라벨 모두 추출, 7개 dataset dir.

raw → simple 매핑은 config 의 *_RAW_TO_NEW / *_RAW_TO_SIMPLE 가 단독 truth.
"""
import os, sys, glob, json, copy, random
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from configs.aica_multitask_config import (
    CONFIG, CATEGORIES,
    UPPER_RAW_TO_NEW, LOWER_RAW_TO_NEW,
    POSE_RAW_TO_SIMPLE, HAND_RAW_TO_SIMPLE, FOOT_RAW_TO_SIMPLE,
)


def _map_with_default0(raw, table):
    """table 에 raw 가 있으면 그 값 (-1 도 그대로), 없으면 None.
    -1 은 '해당 head 에선 의미없음' → 호출자가 0 (None) 으로 처리.
    """
    if raw in table:
        v = table[raw]
        return None if v < 0 else v
    return None


def map_upper(raw):  return _map_with_default0(raw, UPPER_RAW_TO_NEW)
def map_lower(raw):  return _map_with_default0(raw, LOWER_RAW_TO_NEW)
def map_pose(raw):   return POSE_RAW_TO_SIMPLE.get(raw, None)
def map_hand(raw):   return HAND_RAW_TO_SIMPLE.get(raw, None)
def map_foot(raw):   return FOOT_RAW_TO_SIMPLE.get(raw, None)


def make_dummy_interp(pre, aft, num):
    pre_kps = np.array(pre['keypoints'])
    aft_kps = np.array(aft['keypoints'])
    out = []
    for i in range(1, num + 1):
        kp = pre_kps.copy()
        kp[:, :2] = pre_kps[:, :2] + (aft_kps[:, :2] - pre_kps[:, :2]) * i / (num + 1)
        kp[:, 2] *= 0.5
        new_anno = copy.deepcopy(pre); new_anno['keypoints'] = kp.tolist()
        out.append(new_anno)
    return out


def get_majority_id(annos, key, mapper, margin_ratio=0.3, default=0):
    n = len(annos); start = int(n * margin_ratio); end = n - start
    labels = []
    for a in annos[start:end]:
        raw = a.get('action_id', {}).get(key)
        if raw is None: continue
        s = mapper(raw)
        if s is not None:
            labels.append(s)
    if not labels:
        return default
    values, counts = np.unique(np.array(labels), return_counts=True)
    return int(values[counts.argmax()])


def clip_json_to_npy(clip_json_path, T=60):
    try:
        d = json.load(open(clip_json_path))
    except Exception:
        return None
    annos = d.get('annotations', [])
    if len(annos) < 10:
        return None
    annos = sorted(annos, key=lambda a: a.get('image_id', 0))

    filled = [annos[0]]
    for i in range(1, len(annos)):
        gap = annos[i].get('image_id', 0) - annos[i-1].get('image_id', 0)
        if gap > 1:
            filled.extend(make_dummy_interp(annos[i-1], annos[i], gap - 1))
        filled.append(annos[i])

    labels = {
        'action_upper': get_majority_id(filled, 'action_upper', map_upper),
        'action_lower': get_majority_id(filled, 'action_lower', map_lower),
        'pose':         get_majority_id(filled, 'pose',         map_pose),
        'hand':         get_majority_id(filled, 'hand',         map_hand),
        'foot':         get_majority_id(filled, 'foot',         map_foot),
    }

    kps = []
    for a in filled:
        kp = a.get('keypoints')
        if kp is None: continue
        arr = np.array(kp, dtype=np.float32)
        if arr.shape != (17, 3): continue
        kps.append(arr)
    if len(kps) < T:
        return None

    start = (len(kps) - T) // 2
    return {'kp': np.stack(kps[start:start+T], axis=0), 'labels': labels}


def split_by_groups(items, ratios=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    by_group = defaultdict(list)
    for p, g in items: by_group[g].append(p)
    groups = list(by_group.keys()); random.shuffle(groups)
    n = len(groups); n_train = int(n * ratios[0]); n_val = int(n * ratios[1])
    splits = {'train': [], 'val': [], 'test': []}
    for i, g in enumerate(groups):
        tgt = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')
        splits[tgt].extend(by_group[g])
    return splits


def main():
    cfg_paths = CONFIG['paths']
    T = CONFIG['frames_per_clip']
    npy_root = cfg_paths['npy_root']
    split_root = cfg_paths['split_root']
    os.makedirs(npy_root, exist_ok=True)

    print("=== Stage 1: clip 수집 ===")
    seen = {}
    for root in cfg_paths['clip_root_list']:
        for cat in ['action_lower', 'action_upper']:
            for f in glob.glob(os.path.join(root, cat, '*', '*.json')):
                base = os.path.splitext(os.path.basename(f))[0]
                if base not in seen:
                    seen[base] = f
    clip_files = list(seen.values())
    print(f"  unique clips: {len(clip_files)}")

    print("=== Stage 2: NPY 변환 ===")
    items_for_split = []
    label_dists = {k: defaultdict(int) for k in CATEGORIES}
    failed = 0
    for i, fp in enumerate(clip_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(clip_files)} ok={len(items_for_split)} fail={failed}", flush=True)
        res = clip_json_to_npy(fp, T=T)
        if res is None:
            failed += 1; continue
        base = os.path.splitext(os.path.basename(fp))[0]
        lower_lbl = res['labels']['action_lower']
        out_dir = os.path.join(npy_root, f'{lower_lbl:02d}')
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'{base}.npy'), res['kp'])
        with open(os.path.join(out_dir, f'{base}.meta.json'), 'w') as f:
            json.dump(res['labels'], f)
        for k, v in res['labels'].items():
            label_dists[k][v] += 1
        gkey = base.split('T')[0] if 'T' in base else base
        items_for_split.append((os.path.join(out_dir, f'{base}.npy'), gkey))

    print(f"\n총 {len(items_for_split)} NPY, 실패 {failed}")
    for cat, dist in label_dists.items():
        print(f"\n=== {cat} ({CATEGORIES[cat]} classes) ===")
        for k in range(CATEGORIES[cat]):
            print(f"  cls {k}: {dist.get(k, 0)}")

    print("\n=== Stage 3: split ===")
    splits = split_by_groups(items_for_split, ratios=(0.7, 0.15, 0.15))
    for split_name, paths in splits.items():
        for src in paths:
            label_dir = os.path.basename(os.path.dirname(src))
            dst_dir = os.path.join(split_root, split_name, 'action_lower', label_dir)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            if not os.path.exists(dst):
                os.symlink(src, dst)
            meta_src = src.replace('.npy', '.meta.json')
            meta_dst = dst.replace('.npy', '.meta.json')
            if os.path.exists(meta_src) and not os.path.exists(meta_dst):
                os.symlink(meta_src, meta_dst)
    print(f"  train: {len(splits['train'])}  val: {len(splits['val'])}  test: {len(splits['test'])}")


if __name__ == '__main__':
    main()
