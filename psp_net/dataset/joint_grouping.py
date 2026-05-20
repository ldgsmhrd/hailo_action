"""Body-part joint grouping for PSP-Net.

PSP-Net 은 17 COCO keypoint 를 5 신체부위 × 5 슬롯 = 25 형태로 재배열한다.
실제 관절 수가 5 미만인 부위는 zero padding 으로 채운다.

매핑:
    Head  (5 real)  : [0 코, 1 왼눈, 2 오른눈, 3 왼귀, 4 오른귀]
    L-arm (3 real)  : [5 왼어깨, 7 왼팔꿈치, 9 왼손목, PAD, PAD]
    R-arm (3 real)  : [6 오른어깨, 8 오른팔꿈치, 10 오른손목, PAD, PAD]
    L-leg (3 real)  : [11 왼골반, 13 왼무릎, 15 왼발목, PAD, PAD]
    R-leg (3 real)  : [12 오른골반, 14 오른무릎, 16 오른발목, PAD, PAD]
"""
import numpy as np


# 신체부위 그룹 정의 (-1 = padding slot)
BODY_PART_GROUPS = {
    'head':  [0, 1, 2, 3, 4],
    'l_arm': [5, 7, 9, -1, -1],
    'r_arm': [6, 8, 10, -1, -1],
    'l_leg': [11, 13, 15, -1, -1],
    'r_leg': [12, 14, 16, -1, -1],
}

PART_NAMES = ['head', 'l_arm', 'r_arm', 'l_leg', 'r_leg']
NUM_PARTS = 5
JOINTS_PER_PART = 5
NUM_JOINTS_PADDED = NUM_PARTS * JOINTS_PER_PART   # 25

# 평탄화 한 인덱스 시퀀스 (-1 = padding)
BODY_PART_ORDER = []
for part in PART_NAMES:
    BODY_PART_ORDER.extend(BODY_PART_GROUPS[part])
assert len(BODY_PART_ORDER) == NUM_JOINTS_PADDED


def reorder_to_body_part(keypoints):
    """[T, 17, C] → [T, 25, C] 신체부위 그룹 순서로 재배열.

    Padding 슬롯은 0 으로 채움.

    Args:
        keypoints: np.ndarray shape [T, 17, C] (C=3: x, y, conf)
    Returns:
        np.ndarray shape [T, 25, C]
    """
    T, J, C = keypoints.shape
    assert J == 17, f"expected 17 COCO joints, got {J}"
    out = np.zeros((T, NUM_JOINTS_PADDED, C), dtype=keypoints.dtype)
    for slot, src_idx in enumerate(BODY_PART_ORDER):
        if src_idx >= 0:
            out[:, slot] = keypoints[:, src_idx]
    return out


def get_part_slice(part_name):
    """특정 신체부위의 슬롯 범위 반환 (start, end)."""
    idx = PART_NAMES.index(part_name)
    return idx * JOINTS_PER_PART, (idx + 1) * JOINTS_PER_PART


def get_padding_mask():
    """[25] bool array — True = padding slot, False = real joint."""
    return np.array([idx < 0 for idx in BODY_PART_ORDER], dtype=bool)


# 부모 관절 매핑 (body-part 순서 기준)
# bone direction 채널 계산용 — 본인의 부모가 같은 시퀀스의 어디에 있는지
_RAW_PARENT = {
    0: -1, 1: 0, 2: 0, 3: 1, 4: 2,
    5: 0, 6: 0, 7: 5, 8: 6, 9: 7, 10: 8,
    11: 5, 12: 6, 13: 11, 14: 12, 15: 13, 16: 14,
}


def get_parent_positions():
    """[25] int array — 각 슬롯의 부모가 시퀀스의 어느 위치에 있는가.
    부모가 없거나 padding 이면 자기 자신 (= bone direction = 0)."""
    parent_pos = []
    raw_to_slot = {}
    for slot, raw_idx in enumerate(BODY_PART_ORDER):
        if raw_idx >= 0:
            raw_to_slot[raw_idx] = slot
    for slot, raw_idx in enumerate(BODY_PART_ORDER):
        if raw_idx < 0:
            parent_pos.append(slot)   # padding → 자기
            continue
        parent_raw = _RAW_PARENT.get(raw_idx, -1)
        if parent_raw < 0 or parent_raw not in raw_to_slot:
            parent_pos.append(slot)   # 루트 → 자기
        else:
            parent_pos.append(raw_to_slot[parent_raw])
    return np.array(parent_pos, dtype=np.int64)
