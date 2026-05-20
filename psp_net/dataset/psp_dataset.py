"""PSP-Net 학습용 Dataset — 기존 NPY 데이터를 body-part 인코딩으로 변환."""
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from .encoder import BodyPartEncoder


class PSPDataset(Dataset):
    """기존 multi-task NPY 디렉터리 구조를 그대로 사용.

    구조: split_root/<split>/action_lower/<class_idx>/*.npy + .meta.json

    NPY 는 [T, 17, 3] (raw keypoint).
    Body-part 재배열 후 7채널 pseudo-image [7, T, 25] 생성.
    Meta JSON 에서 5 head 라벨 동시 로드.
    """

    def __init__(self, split_root, split, T=60, augment=False, augment_cfg=None):
        self.T = T
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.encoder = BodyPartEncoder()

        base = os.path.join(split_root, split, 'action_lower')
        self.paths = []
        self.labels = []
        for d in sorted(glob.glob(os.path.join(base, '*'))):
            for npy in glob.glob(os.path.join(d, '*.npy')):
                meta_path = npy.replace('.npy', '.meta.json')
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                self.paths.append(npy)
                self.labels.append(meta)
        if not self.paths:
            raise RuntimeError(f"No NPY found under {base}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        clip = np.load(self.paths[idx]).astype(np.float32)
        labels = self.labels[idx]
        clip = self._fix_temporal(clip)
        clip = self._normalize_coords(clip)
        if self.augment:
            clip = self._augment(clip)
        feat = self.encoder.encode(clip)
        label_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()}
        return torch.from_numpy(feat), label_tensors

    def _fix_temporal(self, clip):
        T = self.T
        if clip.shape[0] == T:
            return clip
        if clip.shape[0] > T:
            start = np.random.randint(0, clip.shape[0] - T + 1) if self.augment else (clip.shape[0] - T) // 2
            return clip[start:start + T]
        pad = np.repeat(clip[-1:], T - clip.shape[0], axis=0)
        return np.concatenate([clip, pad], axis=0)

    def _normalize_coords(self, clip):
        valid = clip[..., 2] > 0.1
        if valid.any():
            xs = clip[..., 0][valid]
            ys = clip[..., 1][valid]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            w = max(x_max - x_min, 1e-3)
            h = max(y_max - y_min, 1e-3)
            clip[..., 0] = (clip[..., 0] - x_min) / w
            clip[..., 1] = (clip[..., 1] - y_min) / h
        return clip

    def _augment(self, clip):
        cfg = self.augment_cfg
        # horizontal flip
        if np.random.random() < cfg.get('flip_prob', 0.0):
            # body-part 인코딩 사용하므로 17 idx 기준 좌우 swap
            FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                          (11, 12), (13, 14), (15, 16)]
            clip[..., 0] = 1.0 - clip[..., 0]
            for a, b in FLIP_PAIRS:
                tmp = clip[:, a].copy()
                clip[:, a] = clip[:, b]
                clip[:, b] = tmp
        # coordinate noise
        std = cfg.get('coord_noise_std', 0.0)
        if std > 0:
            noise = np.random.randn(*clip[..., :2].shape).astype(np.float32) * std
            valid = clip[..., 2:3] > 0
            clip[..., :2] += noise * valid
        # confidence dropout
        drop_p = cfg.get('conf_drop_prob', 0.0)
        if drop_p > 0:
            mask = np.random.random(clip.shape[:2]) < drop_p
            clip[..., 2] = clip[..., 2] * (~mask)
        return clip
