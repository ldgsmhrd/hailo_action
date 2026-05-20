"""NTU RGB+D 60 학습 config — 표준 benchmark 비교용 단일 헤드.

NTU RGB+D 60 데이터셋:
  - 60 액션 클래스 (drink water, brushing teeth, ..., walking towards each other 등)
  - 56,880 비디오 샘플
  - 25 joint skeleton (NTU 표준 — 우리 TSSI 25 joint 와 동일 구조)
  - Cross-Subject (CS), Cross-View (CV) 표준 프로토콜

다운로드:
  https://rose1.ntu.edu.sg/dataset/actionRecognition/ (registration 필요)

전처리:
  - .skeleton 파일 → .npy 변환 필요 (training/dataset/gen_npy_ntu60.py 참고)
  - 17 COCO 변환 없이 NTU 25 joint 직접 사용

학습 목표:
  - CS 정확도 88-91% (PoseC3D 94.1%, CTR-GCN 92.4% 와 비교)
  - 핵심 contribution: NPU 호환 + INT8 양자화 거의 무손실
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NTU 60 액션 클래스 (1-indexed → 0-indexed 변환)
NTU60_CLASSES = [
    "drink_water", "eat_meal_snack", "brushing_teeth", "brushing_hair", "drop",
    "pickup", "throw", "sitting_down", "standing_up", "clapping",
    "reading", "writing", "tear_up_paper", "wear_jacket", "take_off_jacket",
    "wear_a_shoe", "take_off_a_shoe", "wear_on_glasses", "take_off_glasses", "put_on_a_hat_cap",
    "take_off_a_hat_cap", "cheer_up", "hand_waving", "kicking_something", "reach_into_pocket",
    "hopping_one_foot_jumping", "jump_up", "make_a_phone_call", "playing_with_phone", "typing_on_a_keyboard",
    "pointing_to_something", "taking_a_selfie", "check_time_from_watch", "rub_two_hands", "nod_head_bow",
    "shake_head", "wipe_face", "salute", "put_palms_together", "cross_hands_in_front",
    "sneeze_cough", "staggering", "falling_down", "touch_head", "touch_chest",
    "touch_back", "touch_neck", "nausea_vomiting", "fan_self", "punching_slapping_other_person",
    "kicking_other_person", "pushing_other_person", "pat_on_back_of_other_person", "point_finger_at_other_person", "hugging_other_person",
    "giving_object_to_other_person", "touch_other_persons_pocket", "handshaking", "walking_towards_each_other", "walking_apart_from_each_other",
]

# 분류 프로토콜
PROTOCOLS = {
    'cross_subject': {
        # 표준 X-Sub: subject 1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38 = train
        # 나머지 = test
        'train_subjects': [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38],
    },
    'cross_view': {
        # 표준 X-View: camera 2,3 = train, camera 1 = test
        'train_cameras': [2, 3],
    },
}

CONFIG = {
    'paths': {
        # 사용자가 다운로드한 NTU60 데이터 위치
        'ntu_skeleton_root': '/data/ntu60/nturgb+d_skeletons',  # .skeleton 파일들
        'npy_root':   os.path.join(ROOT, 'data', 'ntu60_npy'),
        'split_root': os.path.join(ROOT, 'data', 'ntu60_split'),
        'model_dir':  os.path.join(ROOT, 'models'),
        'result_dir': os.path.join(ROOT, 'result'),
    },
    'protocol': 'cross_subject',   # 'cross_subject' 또는 'cross_view'
    'num_classes': 60,
    'frames_per_clip': 60,
    'num_joints': 25,              # NTU 25 joint 직접 사용
    'num_persons': 2,              # NTU 는 최대 2명 상호작용 라벨 있음. 단일 사람만 사용 시 1
    'encoder': {
        'order': 'ntu_tree',       # NTU 의 spine-centric tree traversal
        'channels': ['pos', 'velocity', 'angle'],  # 7채널 동일
    },
    'model': {
        'backbone': 'resnet18',
        'pretrained': True,
        'first_conv_stride': (2, 1),
        'multi_task': False,       # 단일 헤드
        'num_classes': 60,
    },
    'training': {
        'optimizer': 'SGD', 'lr': 0.05, 'momentum': 0.9, 'nesterov': True,
        'weight_decay': 0.0001, 'lr_schedule': 'cosine',
        'batch_size': 64, 'num_workers': 8, 'epochs': 100,
        'amp': True, 'device': 'cuda',
        'use_class_weights': False,  # NTU 는 balance 데이터셋이라 불필요
    },
    'augment': {
        'enabled': True,
        'flip_prob': 0.5, 'coord_noise_std': 0.01,
        'temporal_shift': 5,
    },
    'export': {'opset_version': 11, 'dynamic_axes': False},
}
