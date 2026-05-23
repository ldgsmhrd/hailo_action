"""NTU RGB+D 25 joint → 5 신체부위 그룹 매핑.

NTU 25 관절 인덱스 (1-base in raw, here 0-base):
  0  base of the spine        13 left knee
  1  middle of the spine      14 left ankle
  2  neck                     15 left foot
  3  head                     16 right hip
  4  left shoulder            17 right knee
  5  left elbow               18 right ankle
  6  left wrist               19 right foot
  7  left hand                20 spine (between shoulders)
  8  right shoulder           21 tip of left hand
  9  right elbow              22 left thumb
 10  right wrist              23 tip of right hand
 11  right hand               24 right thumb
 12  left hip

PSP-Net 입력 규약: 5 부위 × 5 슬롯 = 25 (마침 NTU 도 25개!)
"""

import numpy as np

# 5 부위 × 5 슬롯
# -1 = padding (실제로는 안 씀, NTU 는 모든 슬롯 채워짐)
BODY_PART_GROUPS_NTU = {
    'head':   [3,  2, 20,  0,  1],   # 머리/목/척추 (5)
    'l_arm':  [4,  5,  6,  7, 21],   # 왼어깨-팔꿈치-손목-손-손끝 (5)
    'r_arm':  [8,  9, 10, 11, 23],   # 오른팔 (5)
    'l_leg':  [12, 13, 14, 15, 22],  # 왼다리 + 왼엄지(여분) (5)
    'r_leg':  [16, 17, 18, 19, 24],  # 오른다리 + 오른엄지 (5)
}

PART_ORDER = ['head', 'l_arm', 'r_arm', 'l_leg', 'r_leg']
NUM_PARTS = 5
JOINTS_PER_PART = 5
TOTAL_SLOTS = NUM_PARTS * JOINTS_PER_PART   # 25


def get_reorder_indices():
    """Flatten 한 25 슬롯 인덱스 리스트 반환."""
    idx = []
    for part in PART_ORDER:
        idx.extend(BODY_PART_GROUPS_NTU[part])
    return np.array(idx, dtype=np.int64)


def reorder_to_body_part(skeleton):
    """[..., 25, C] → [..., 25, C] 부위순서로 재배열.

    NTU 원본 순서 → PSP-Net body-part 순서.
    """
    idx = get_reorder_indices()
    return skeleton[..., idx, :]


if __name__ == '__main__':
    idx = get_reorder_indices()
    print(f"Reorder indices ({len(idx)}): {idx.tolist()}")
    assert len(idx) == 25
    assert len(set(idx.tolist())) == 25, "중복 없어야 함"
    print("✅ NTU 25 → 5부위×5슬롯 매핑 OK")
