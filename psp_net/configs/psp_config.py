"""PSP-Net 학습 config — 자체 데이터셋 (13,878 clip)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 5 head 클래스 수 (기존 ResNet18 baseline 과 동일)
CATEGORIES = {
    'action_upper': 6,
    'action_lower': 10,
    'pose':         9,
    'hand':         3,
    'foot':         3,
}

CONFIG = {
    'paths': {
        # 기존 NPY 그대로 재사용 (AICA 서버 기준)
        'split_root': '/home/ubuntu/safemotion/ResNet-ActionRecognition/data/split_mt_v3',
        'model_dir':  os.path.join(ROOT, 'psp_net', 'models', 'checkpoints'),
        'result_dir': os.path.join(ROOT, 'psp_net', 'results'),
    },
    'categories': CATEGORIES,
    'frames_per_clip': 60,
    'num_parts': 5,
    'joints_per_part': 5,
    'in_channels': 7,
    'model': {
        'num_parts': 5,
        'joints_per_part': 5,
    },
    'training': {
        'optimizer': 'SGD',
        'lr': 0.05,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
        'lr_schedule': 'cosine',
        'batch_size': 64,    # 모델이 가벼워서 batch 크게 가능
        'num_workers': 8,
        'epochs': 100,
        'amp': False,
        'device': 'cuda',
        'loss_weights': {
            'action_lower': 1.0,
            'action_upper': 1.0,
            'pose':         1.0,
            'hand':         0.5,
            'foot':         0.5,
        },
        'use_class_weights': True,
    },
    'augment': {
        'enabled': True,
        'flip_prob': 0.5,
        'coord_noise_std': 0.01,
        'conf_drop_prob': 0.05,
        'temporal_shift': 5,
    },
    'export': {
        'opset_version': 11,
        'dynamic_axes': False,
    },
}
