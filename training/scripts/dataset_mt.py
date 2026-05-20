"""Multi-task Dataset — NPY + meta.json 에서 5 라벨 모두 반환."""
import os, sys, glob, json
import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from dataset.encoders import PseudoImageEncoder, COCO_FLIP_PAIRS, NUM_KEYPOINTS


class MultiTaskActionDataset(Dataset):
    def __init__(self, split_root, split, T=60, encoder=None,
                 augment=False, augment_cfg=None):
        self.T = T
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.encoder = encoder or PseudoImageEncoder()

        # split_root/<split>/action_lower/<cls>/*.npy
        base_dir = os.path.join(split_root, split, 'action_lower')
        self.paths = []
        self.labels = []   # list of dict {action_upper, action_lower, pose, hand, foot}
        for d in sorted(glob.glob(os.path.join(base_dir, '*'))):
            for npy in glob.glob(os.path.join(d, '*.npy')):
                meta_path = npy.replace('.npy', '.meta.json')
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                self.paths.append(npy)
                self.labels.append(meta)
        if not self.paths:
            raise RuntimeError(f"No NPY found under {base_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        clip = np.load(self.paths[idx]).astype(np.float32)
        labels = self.labels[idx]
        clip = self._fix_temporal(clip)
        clip = self._normalize_coords(clip)
        if self.augment:
            clip = self._augment(clip)
        tensor_np = self.encoder.encode(clip)
        # 라벨 dict → tensor dict
        label_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()}
        return torch.from_numpy(tensor_np), label_tensors

    def _fix_temporal(self, clip):
        T = self.T
        if clip.shape[0] == T: return clip
        if clip.shape[0] > T:
            start = np.random.randint(0, clip.shape[0] - T + 1) if self.augment else (clip.shape[0] - T) // 2
            return clip[start:start + T]
        pad = np.repeat(clip[-1:], T - clip.shape[0], axis=0)
        return np.concatenate([clip, pad], axis=0)

    def _normalize_coords(self, clip):
        valid = clip[..., 2] > 0.1
        if valid.any():
            xs = clip[..., 0][valid]; ys = clip[..., 1][valid]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            w = max(x_max - x_min, 1e-3); h = max(y_max - y_min, 1e-3)
            clip[..., 0] = (clip[..., 0] - x_min) / w
            clip[..., 1] = (clip[..., 1] - y_min) / h
        return clip

    def _augment(self, clip):
        cfg = self.augment_cfg
        if np.random.random() < cfg.get('flip_prob', 0.0):
            clip[..., 0] = 1.0 - clip[..., 0]
            for a, b in COCO_FLIP_PAIRS:
                clip[:, [a, b]] = clip[:, [b, a]]
        std = cfg.get('coord_noise_std', 0.0)
        if std > 0:
            clip[..., :2] += np.random.normal(0.0, std, clip[..., :2].shape).astype(np.float32)
        drop_p = cfg.get('conf_drop_prob', 0.0)
        if drop_p > 0:
            mask = np.random.random(clip.shape[:2]) < drop_p
            clip[mask, 2] = 0.0
        return clip
