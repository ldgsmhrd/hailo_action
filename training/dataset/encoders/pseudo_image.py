"""
Pseudo-image 인코더 — keypoints[T, 17, 3] → tensor[C, T, J] 변환.

설정 두 가지:
  order   : 'naive' (0~16 순) | 'tssi' (트리 DFS 순서)
  channels: 'pos' / 'velocity' / 'angle' 의 조합 (리스트)

채널별 의미:
  - pos      (3) : x, y, conf
  - velocity (2) : dx/dt, dy/dt    (시간축 미분)
  - angle    (2) : dx, dy from parent (뼈 방향 벡터)

조합 예:
  channels=['pos']                       → 3 channel
  channels=['pos', 'velocity']           → 5 channel
  channels=['pos', 'angle']              → 5 channel
  channels=['pos', 'velocity', 'angle']  → 7 channel  ← 권장
"""

import numpy as np
from .skeleton import (
    NUM_KEYPOINTS,
    COCO_PARENT,
    get_order,
)


VALID_CHANNELS = ('pos', 'velocity', 'angle')


class PseudoImageEncoder:
    """
    학습/추론 양쪽에서 동일하게 사용.
    입력: np.ndarray [T, 17, 3]  (x, y, conf)
    출력: np.ndarray [C, T, J]   (C와 J는 설정에 따라)
    """

    def __init__(self, order='tssi', channels=('pos', 'velocity', 'angle')):
        # 검증
        for c in channels:
            if c not in VALID_CHANNELS:
                raise ValueError(f"channel '{c}' not in {VALID_CHANNELS}")
        if order not in ('naive', 'tssi'):
            raise ValueError(f"order '{order}' not in ('naive', 'tssi')")

        self.order_name = order
        self.order_idx = get_order(order)     # [J]  관절 인덱스 매핑
        self.channels = list(channels)

        # 부모 인덱스를 정렬 순서 기준으로 미리 계산 (각도 채널용)
        #   self.order_idx[j] 의 부모가 self.order_idx 안의 어디 위치인지
        self._parent_pos = self._precompute_parent_positions()

    @property
    def num_channels(self):
        cmap = {'pos': 3, 'velocity': 2, 'angle': 2}
        return sum(cmap[c] for c in self.channels)

    @property
    def num_joints(self):
        return len(self.order_idx)

    # -----------------------------------------------------------------
    def _precompute_parent_positions(self):
        """각 위치 j 의 부모가 same sequence 의 어느 위치에 있는지.
        부모가 없거나 sequence 에 없으면 자기 자신 (dx=dy=0 됨)."""
        parent_pos = []
        for pos, joint_idx in enumerate(self.order_idx):
            parent_joint = COCO_PARENT.get(joint_idx, -1)
            if parent_joint < 0:
                parent_pos.append(pos)            # root → 자기 자신
                continue
            # 같은 sequence 안에서 부모 관절의 첫 등장 위치
            if parent_joint in self.order_idx:
                parent_pos.append(self.order_idx.index(parent_joint))
            else:
                parent_pos.append(pos)
        return np.array(parent_pos, dtype=np.int64)

    # -----------------------------------------------------------------
    def encode(self, keypoints):
        """
        keypoints: np.ndarray shape [T, 17, 3]  (x, y, conf)
        반환:      np.ndarray shape [C, T, J]   float32
        """
        if keypoints.shape[1] != NUM_KEYPOINTS or keypoints.shape[2] != 3:
            raise ValueError(f"Expected [T, 17, 3], got {keypoints.shape}")

        kp = keypoints.astype(np.float32, copy=False)

        # 관절 순서 재배열  [T, 17, 3] → [T, J, 3]
        reordered = kp[:, self.order_idx, :]    # J = num_joints

        T, J, _ = reordered.shape
        out_channels = []

        # 1) 위치
        if 'pos' in self.channels:
            pos_x = reordered[..., 0]            # [T, J]
            pos_y = reordered[..., 1]
            pos_c = reordered[..., 2]
            out_channels.extend([pos_x, pos_y, pos_c])

        # 2) 속도 (시간 미분)
        if 'velocity' in self.channels:
            vel = np.zeros((T, J, 2), dtype=np.float32)
            vel[1:] = reordered[1:, :, :2] - reordered[:-1, :, :2]
            out_channels.extend([vel[..., 0], vel[..., 1]])

        # 3) 뼈 방향 (부모로부터의 dx, dy)
        if 'angle' in self.channels:
            parent_kp = reordered[:, self._parent_pos, :2]    # [T, J, 2]
            bone = reordered[..., :2] - parent_kp              # [T, J, 2]
            out_channels.extend([bone[..., 0], bone[..., 1]])

        tensor = np.stack(out_channels, axis=0)   # [C, T, J]
        return tensor.astype(np.float32)


def build_encoder(config):
    """config dict 에서 인코더 생성. configs/resnet18_action_config.py 형식 사용."""
    enc_cfg = config['encoder'] if isinstance(config, dict) else config
    return PseudoImageEncoder(
        order=enc_cfg.get('order', 'tssi'),
        channels=tuple(enc_cfg.get('channels', ('pos', 'velocity', 'angle'))),
    )
