"""NTU RGB+D 60 PyTorch Dataset.

전처리된 .npy 파일들 로드 (preprocess_ntu.py 로 미리 변환).
파일 규약:
    {data_root}/{split_name}/{class_idx:02d}/{sample_id}.npy
    sample_id: SsssCcccPpppRrrrAaaa (원본 파일명에서 .skeleton 제외)

각 .npy : [M=2, T=64, J=25, 3]  float32  (원본은 3D 보관)
→ 학습 시 z 채널은 drop, 2D 만 사용 (YOLO-Pose 배포 정합성)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .ntu_joint_grouping import get_reorder_indices


# NTU120 Cross-Subject 공식 (53 train subjects 중 106 train clip subset)
CS_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103}

# NTU120 Cross-Setup 공식 — 짝수 setup train / 홀수 setup test
CSET_TRAIN_SETUPS = set(range(2, 33, 2))


def parse_filename(fname):
    """SsssCcccPpppRrrrAaaa → (setup, camera, performer, replication, action_idx)."""
    base = os.path.basename(fname).replace('.skeleton', '').replace('.npy', '')
    # 길이 20 자
    return {
        'setup':       int(base[1:4]),
        'camera':      int(base[5:8]),
        'performer':   int(base[9:12]),
        'replication': int(base[13:16]),
        'action':      int(base[17:20]),  # 1..60
    }


def make_split(npy_root, benchmark='xsub'):
    """전처리된 npy 들을 train/test 로 split.

    Returns: (train_files, test_files) — 각 list of (path, class_idx)
    """
    all_files = []
    for cls_dir in sorted(os.listdir(npy_root)):
        if not cls_dir.isdigit():
            continue
        cls_idx = int(cls_dir)
        for f in glob.glob(os.path.join(npy_root, cls_dir, '*.npy')):
            all_files.append((f, cls_idx))

    train, test = [], []
    for path, cls in all_files:
        meta = parse_filename(path)
        if benchmark == 'xsub':
            in_train = meta['performer'] in CS_TRAIN_SUBJECTS
        elif benchmark == 'cset':
            in_train = meta['setup'] in CSET_TRAIN_SETUPS
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        (train if in_train else test).append((path, cls))

    return train, test


class NTUDataset(Dataset):
    """NTU60 PSP-Net 입력 생성.

    입력 npy: [M=2, T=64, J=25, 3] (x, y, z) — z 는 drop
    출력:
        x:   [6, T, 25]  (x, y, vx, vy, bx, by)  ← 2D only
        lbl: scalar (0..59)
    """

    def __init__(self, file_list, T=64, augment=False,
                 use_body=0, use_3d=False, use_bone_motion=False,
                 augment_cfg=None):
        """
        file_list: [(path, class_idx), ...]
        use_body : 0=main body only, 1=second only, -1=concat both
        use_3d   : False=2D (6ch/body), True=3D (9ch/body)
        use_bone_motion: True 면 bone motion 추가 (Multi-Branch 용)
          - 2D + BM: 8ch/body  (x,y, vx,vy, bx,by, bvx,bvy)
          - 3D + BM: 12ch/body (x,y,z, vx,vy,vz, bx,by,bz, bvx,bvy,bvz)
        """
        self.files = file_list
        self.T = T
        self.augment = augment
        self.use_body = use_body
        self.use_3d = use_3d
        self.use_bone_motion = use_bone_motion
        self.augment_cfg = augment_cfg or {}
        self.reorder_idx = get_reorder_indices()

        # NTU joint parent (bone) — for bone direction
        # bone[i] = parent of joint i (0 has no parent → use itself = bone 0)
        self.bone_parent = np.array([
            0, 0, 20, 2,            # 0=spine_base, 1=spine_mid, 2=neck, 3=head
            20, 4, 5, 6,            # 4..7  left arm (shoulder, elbow, wrist, hand)
            20, 8, 9, 10,           # 8..11 right arm
            0, 12, 13, 14,          # 12..15 left leg
            0, 16, 17, 18,          # 16..19 right leg
            1,                       # 20 spine (between shoulders)
            7, 7,                    # 21 tip of left hand, 22 left thumb (parent=hand 7)
            11, 11,                  # 23 tip of right hand, 24 right thumb (parent=hand 11)
        ], dtype=np.int64)

    def __len__(self):
        return len(self.files)

    def _augment(self, sk):
        """sk: [M, T, J, 3]  — 강화된 augmentation 7가지.

        config 의 각 key 가 0/None 이면 해당 augment skip.
        """
        cfg = self.augment_cfg
        sk = sk.astype(np.float32, copy=True)

        # ① 좌우 flip (x 부호 반전 + L/R 인덱스 swap)
        if cfg.get('flip_prob', 0) > 0 and np.random.rand() < cfg['flip_prob']:
            sk[..., 0] = -sk[..., 0]
            l_idx = np.array([4, 5, 6, 7, 21, 22, 12, 13, 14, 15])
            r_idx = np.array([8, 9, 10, 11, 23, 24, 16, 17, 18, 19])
            sk[:, :, l_idx], sk[:, :, r_idx] = sk[:, :, r_idx].copy(), sk[:, :, l_idx].copy()

        # ② Random rotation (Y축 중심 ±deg 도) — 카메라 시점 변화
        rot_deg = cfg.get('rotation_deg', 0)
        if rot_deg > 0:
            angle = np.random.uniform(-rot_deg, rot_deg) * np.pi / 180.0
            cos, sin = np.cos(angle), np.sin(angle)
            # 2D 회전 (x, z) — Y축 중심
            if self.use_3d:
                x, z = sk[..., 0].copy(), sk[..., 2].copy()
                sk[..., 0] = cos * x - sin * z
                sk[..., 2] = sin * x + cos * z
            else:
                # 2D 만 있으면 (x, y) 회전 (이미지 plane)
                x, y = sk[..., 0].copy(), sk[..., 1].copy()
                sk[..., 0] = cos * x - sin * y
                sk[..., 1] = sin * x + cos * y

        # ③ Random scale (±range)
        sc = cfg.get('scale_range', 0)
        if sc > 0:
            scale = 1.0 + np.random.uniform(-sc, sc)
            sk *= scale

        # ④ Random translation (±range)
        tr = cfg.get('translation_range', 0)
        if tr > 0:
            tx = np.random.uniform(-tr, tr)
            ty = np.random.uniform(-tr, tr)
            sk[..., 0] += tx
            sk[..., 1] += ty

        # ⑤ 좌표 노이즈 (작게)
        std = cfg.get('coord_noise_std', 0)
        if std > 0:
            sk = sk + np.random.randn(*sk.shape).astype(np.float32) * std

        # ⑥ Joint dropout (랜덤 관절 mask)
        jd = cfg.get('joint_dropout', 0)
        if jd > 0:
            J = sk.shape[2]
            mask = np.random.rand(J) > jd   # True = keep
            for m in range(sk.shape[0]):
                sk[m, :, ~mask, :] = 0

        # ⑦ Temporal shift (앞뒤 일부 자르고 zero-pad)
        ts = cfg.get('temporal_shift', 0)
        if ts > 0:
            shift = np.random.randint(-ts, ts + 1)
            if shift != 0:
                sk = np.roll(sk, shift, axis=1)
                if shift > 0:
                    sk[:, :shift] = 0
                else:
                    sk[:, shift:] = 0

        return sk

    def _make_psp_input(self, sk_body):
        """sk_body: [T, J=25, 3] → [C, T, J]
        2D (use_3d=False): C=6 또는 8 (+ bone_motion)
        3D (use_3d=True):  C=9 또는 12 (+ bone_motion)
        """
        if self.use_3d:
            sk_use = sk_body                                  # [T, J, 3]
        else:
            sk_use = sk_body[..., :2]                         # [T, J, 2]
        # velocity (이전 프레임 대비)
        vel = np.zeros_like(sk_use)
        vel[1:] = sk_use[1:] - sk_use[:-1]
        # bone (parent - self)
        bone = sk_use - sk_use[:, self.bone_parent, :]

        feats = [sk_use, vel, bone]
        # bone motion (bone 의 시간 차분)
        if self.use_bone_motion:
            bone_vel = np.zeros_like(bone)
            bone_vel[1:] = bone[1:] - bone[:-1]
            feats.append(bone_vel)

        x = np.concatenate(feats, axis=2)                    # [T, J, 6/8/9/12]
        x = np.transpose(x, (2, 0, 1))                       # [C, T, J]

        # body-part reorder (NTU 원순서 → PSP-Net 부위순서)
        x = x[:, :, self.reorder_idx]
        return x.astype(np.float32)

    def __getitem__(self, idx):
        path, cls = self.files[idx]
        sk = np.load(path)   # [M, T_in, J, 3]

        # T 보정
        T_in = sk.shape[1]
        if T_in != self.T:
            if T_in < self.T:
                pad = np.zeros((sk.shape[0], self.T - T_in, sk.shape[2], 3),
                               dtype=sk.dtype)
                sk = np.concatenate([sk, pad], axis=1)
            else:
                idx_t = np.linspace(0, T_in - 1, self.T).astype(np.int64)
                sk = sk[:, idx_t]

        if self.augment:
            sk = self._augment(sk)

        # body 선택
        if self.use_body == -1:
            # 두 body 모두 → channel 차원에 concat
            x0 = self._make_psp_input(sk[0])
            x1 = self._make_psp_input(sk[1])
            x = np.concatenate([x0, x1], axis=0)   # [12, T, 25]
        else:
            x = self._make_psp_input(sk[self.use_body])  # [6, T, 25]

        return torch.from_numpy(x), torch.tensor(cls, dtype=torch.long)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ntu_dataset.py <npy_root>")
        sys.exit(0)
    train_files, test_files = make_split(sys.argv[1], benchmark='xsub')
    print(f"Cross-Subject:  train={len(train_files)}  test={len(test_files)}")
    train_files_cv, test_files_cv = make_split(sys.argv[1], benchmark='xview')
    print(f"Cross-View:     train={len(train_files_cv)}  test={len(test_files_cv)}")

    ds = NTUDataset(train_files, T=64, augment=True,
                    augment_cfg={'flip_prob': 0.5, 'coord_noise_std': 0.01})
    x, y = ds[0]
    print(f"Sample 0: x={tuple(x.shape)}  y={y.item()}")
