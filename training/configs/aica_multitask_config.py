"""Multi-task 5-head ResNet18 학습 config v3 — 공식 v22 라벨 + AICA 데이터 100% 활용.

사용자 공식 라벨 표 (행동 22 + 자세 14) + cross-overlap 규칙 (정적 = pose):

  action_upper (6) : None / punch / wave / clap / raise / put-down
                     overlap 제거: 허리구부리기 / 허리펴기 → pose 가 담당
                     양손들기(raw21) → hand 가 담당
  action_lower (10): None / pacing / walk / run / jump-still / fall / kick
                     / jump-2feet / jump-1leg / jump-1leg-still
                     overlap 제거: 앉기/일어서기/기어가기/눕기 → pose
                     서있기(raw31)/앉아있기(raw32)/한발들기(raw38) → pose 가 라벨 보유
  pose (9)         : sit / sit-chair / kneel-down / knee-standing / standing
                     / standing-bending / lying / crawl / other
  hand (3)         : None / cross-arms / raise-both
                     overlap: 가리키기 → 0개 (drop)
  foot (3)         : None / leg-cross / one-leg-raise

합계 31 outputs.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# action_upper : 6 (None + 5)
# raw 매핑 (cvt_labelmap_v22_action_upper, bus01 22-entry 라벨맵 기반)
# ---------------------------------------------------------------------------
ACTION_UPPER_CLASSES = {
    0: 'none',
    1: 'punch',           # 펀치
    2: 'wave',            # 손흔들기
    3: 'clap',            # 손뼉치기
    4: 'raise',           # 손올리기 / 한손들기
    5: 'put-down',        # 손내리기
}

# raw v22 idx → new upper idx (-1 = None 으로)
UPPER_RAW_TO_NEW = {
    0: 0,    # 없음
    1: -1,   # 허리 구부리기 → pose 담당
    2: -1,   # 허리 펴기 → pose 담당
    3: -1,   # 먹기
    4: 1,    # 펀치
    5: -1,   # 휘두르기
    6: 2,    # 손흔들기
    7: -1,   # 가리키기 → hand 담당
    8: 1,    # 밀치기 → punch (offensive arm motion)
    9: -1, 10: -1,
    11: 3,   # 손뼉치기
    12: 4,   # 손올리기
    13: 5,   # 손내리기
    14: -1, 15: -1, 16: -1, 17: -1, 18: -1, 19: -1,
    20: 4,   # 한손들기 → raise 와 합침
    21: -1,  # 양손들기 → hand 담당 (raise-both)
}

# ---------------------------------------------------------------------------
# action_lower : 10 (None + 9)
# ---------------------------------------------------------------------------
ACTION_LOWER_CLASSES = {
    0: 'none',
    1: 'pacing',           # 서성이기
    2: 'walk',             # 걷기
    3: 'run',              # 달리기
    4: 'jump-still',       # 점프-제자리
    5: 'fall',             # 넘어짐 (안전 핵심)
    6: 'kick',             # 킥 (데이터 5개, 형식 유지)
    7: 'jump-2feet',       # 점프-두발
    8: 'jump-1leg',        # 외발점프
    9: 'jump-1leg-still',  # 외발점프-제자리
}

LOWER_RAW_TO_NEW = {
    0: 0,    # 없음
    1: -1,   # 앉기 → pose
    2: -1,   # 일어서기 → pose
    3: 1,    # 서성이기
    4: 2,    # 걷기
    5: 3,    # 달리기
    6: -1,   # 기어가기 → pose (무릎기기)
    7: 4,    # 점프-제자리
    8: 5,    # 넘어짐
    9: -1,   # 떨어짐 (없음)
    10: 6,   # 킥
    11: -1,  # 턴 (0개)
    12: 7,   # 점프-두발
    13: -1,  # 기타
    14: 8,   # 외발점프
    15: 9,   # 외발점프-제자리
    16: -1, 17: -1, 18: -1, 19: -1, 20: -1,
    21: -1,  # 식별불가
    22: -1, 23: -1, 24: -1, 25: -1, 26: -1, 27: -1,
    28: -1,  # 눕기 → pose
    29: -1,  # 허리 구부리기 → pose
    30: -1,  # 허리펴기 → pose
    31: -1,  # 서있기 → pose 가 이미 라벨 보유
    32: -1,  # 앉아있기 → pose 가 이미 라벨 보유
    33: -1, 34: -1, 35: -1, 36: -1, 37: -1,
    38: -1,  # 한발들기 포즈 → pose 가 이미 라벨 보유
}

# ---------------------------------------------------------------------------
# pose : 9 (other 추가)
# ---------------------------------------------------------------------------
POSE_CLASSES = {
    0: 'sit',
    1: 'sit-chair',
    2: 'kneel-down',
    3: 'knee-standing',
    4: 'standing',
    5: 'standing-bending',
    6: 'lying',
    7: 'crawl-pose',
    8: 'other',           # raw 10~18 occlusion / unknown
}

POSE_RAW_TO_SIMPLE = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 17: 8, 18: 8,
}

# ---------------------------------------------------------------------------
# hand : 3 (가리키기 0개라 drop)
# ---------------------------------------------------------------------------
HAND_CLASSES = {
    0: 'none',
    1: 'cross-arms',    # 팔짱끼기
    2: 'raise-both',    # 양팔들기
}

HAND_RAW_TO_SIMPLE = {
    0: 0,
    1: 1,
    4: 2,
}

# ---------------------------------------------------------------------------
# foot : 3
# ---------------------------------------------------------------------------
FOOT_CLASSES = {
    0: 'none',
    1: 'leg-cross',     # 다리꼬기
    2: 'one-leg-raise', # 한쪽다리들기
}

FOOT_RAW_TO_SIMPLE = {
    0: 0,
    1: 1,
    5: 2,
}

# ---------------------------------------------------------------------------
CATEGORIES = {
    'action_upper': len(ACTION_UPPER_CLASSES),  # 6
    'action_lower': len(ACTION_LOWER_CLASSES),  # 10
    'pose':         len(POSE_CLASSES),          # 9
    'hand':         len(HAND_CLASSES),          # 3
    'foot':         len(FOOT_CLASSES),          # 3
}
# 총 head 출력: 31

CONFIG = {
    'paths': {
        'clip_root_list': [
            '/home/ubuntu/safemotion/action_dataset/action_dataset/bus/20250521/clip',
            '/home/ubuntu/safemotion/action_dataset/action_dataset/bus/20250523/clip',
            '/home/ubuntu/safemotion/action_dataset/action_dataset/bus/20250526/clip',
            '/home/ubuntu/safemotion/action_dataset/action_dataset/kids_cafe/20250512/clip',
            '/home/ubuntu/safemotion/action_dataset/action_dataset/kids_cafe/20250513/clip',
            '/home/ubuntu/safemotion/action_dataset/action_dataset/kids_cafe/20250515/clip',
            '/home/ubuntu/safemotion/action_dataset/action_dataset/kids_cafe/20250516/clip',
        ],
        'npy_root':   os.path.join(ROOT, 'data', 'npy_mt_v3'),
        'split_root': os.path.join(ROOT, 'data', 'split_mt_v3'),
        'model_dir':  os.path.join(ROOT, 'models'),
        'result_dir': os.path.join(ROOT, 'result'),
    },
    'categories': CATEGORIES,
    'frames_per_clip': 60,
    'split_ratio': {'train': 0.7, 'val': 0.15, 'test': 0.15},
    'random_seed': 42,
    'encoder': {'order': 'tssi', 'channels': ['pos', 'velocity', 'angle']},
    'model': {
        'backbone': 'resnet18',
        'pretrained': True,
        'first_conv_stride': (2, 1),
        'multi_task': True,
        'heads': CATEGORIES,
    },
    'training': {
        'optimizer': 'SGD', 'lr': 0.05, 'momentum': 0.9, 'nesterov': True,
        'weight_decay': 0.0001, 'lr_schedule': 'cosine',
        'batch_size': 32, 'num_workers': 8, 'epochs': 100,
        'amp': False, 'device': 'cuda',
        'loss_weights': {
            'action_lower': 1.0,
            'action_upper': 1.0,
            'pose':         1.0,
            'hand':         0.5,
            'foot':         0.5,
        },
        # imbalance 완화: 각 head 별 class weight (none 이 너무 많아서)
        'use_class_weights': True,
    },
    'augment': {
        'enabled': True,
        'flip_prob': 0.5, 'coord_noise_std': 0.01,
        'conf_drop_prob': 0.05, 'temporal_shift': 5,
    },
    'export': {'opset_version': 11, 'dynamic_axes': False},
}
