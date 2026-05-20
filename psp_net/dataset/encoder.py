"""PSP-Net 전용 pseudo-image encoder.

기존 PseudoImageEncoder 와 다른 점:
  - 관절 순서가 'body_part' (5 신체부위 × 5 슬롯)
  - 7채널 (pos 3 + velocity 2 + bone direction 2) 동일
  - Padding 슬롯은 confidence=0 으로 처리 → 모델이 자연히 무시

출력 shape: [7, T, 25]   (TSSI 25 와 같지만 순서가 다름)
"""
import numpy as np
from .joint_grouping import (
    BODY_PART_ORDER, reorder_to_body_part, get_parent_positions,
    NUM_JOINTS_PADDED,
)


class BodyPartEncoder:
    """입력 [T, 17, 3] → 출력 [7, T, 25] (body-part 순서)."""

    NUM_CHANNELS = 7    # pos(3) + velocity(2) + bone_dir(2)

    def __init__(self):
        self.parent_pos = get_parent_positions()
        self.num_joints = NUM_JOINTS_PADDED
        self.num_channels = self.NUM_CHANNELS

    def encode(self, keypoints):
        """
        keypoints: np.ndarray [T, 17, 3]  (x, y, conf)
        return:    np.ndarray [7, T, 25]  float32
        """
        kp = keypoints.astype(np.float32)
        # body-part 순서로 재배열 → [T, 25, 3]
        kp_bp = reorder_to_body_part(kp)
        T, J, _ = kp_bp.shape

        # 1. position (3): x, y, conf
        pos = kp_bp                                          # [T, 25, 3]

        # 2. velocity (2): dx, dy at time t = kp[t] - kp[t-1]
        vel = np.zeros((T, J, 2), dtype=np.float32)
        vel[1:] = kp_bp[1:, :, :2] - kp_bp[:-1, :, :2]
        # confidence 가 0 인 (padding) 슬롯에서는 velocity 도 0
        valid = (kp_bp[:, :, 2:3] > 0).astype(np.float32)
        vel = vel * valid

        # 3. bone direction (2): dx, dy from parent
        bone = np.zeros((T, J, 2), dtype=np.float32)
        parent_xy = kp_bp[:, self.parent_pos, :2]
        bone = kp_bp[:, :, :2] - parent_xy
        bone = bone * valid

        # 결합: [T, J, 3+2+2=7] → permute → [7, T, J]
        feat = np.concatenate([pos, vel, bone], axis=-1)
        return np.transpose(feat, (2, 0, 1)).astype(np.float32)
