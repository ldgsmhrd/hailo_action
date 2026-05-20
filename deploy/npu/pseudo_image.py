"""
키포인트 [T, 17, 3] → pseudo-image 변환.

학습 시 사용한 PseudoImageEncoder 를 그대로 사용해서 분포 일치 보장.
학습 인코더는 src/encoders/ 에 복사되어 있음.

기본 설정 (학습과 동일):
  order    = 'tssi'           (트리 DFS 순서, J=25)
  channels = ['pos', 'velocity', 'angle']  (7채널)
"""
import os
import sys

import numpy as np

# encoders 모듈 경로
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from encoders import PseudoImageEncoder


# 학습 시 사용한 설정 (best_aica5_action_lower.pth 메타데이터와 일치)
DEFAULT_ENCODER_ORDER = os.environ.get('ENCODER_ORDER', 'tssi')
DEFAULT_ENCODER_CHANNELS = os.environ.get(
    'ENCODER_CHANNELS', 'pos,velocity,angle'
).split(',')

_singleton_encoder = None


def _get_encoder():
    """싱글톤 — 매번 생성하지 않도록."""
    global _singleton_encoder
    if _singleton_encoder is None:
        _singleton_encoder = PseudoImageEncoder(
            order=DEFAULT_ENCODER_ORDER,
            channels=tuple(DEFAULT_ENCODER_CHANNELS),
        )
    return _singleton_encoder


def keypoints_to_pseudo_image(kp, frame_w=None, frame_h=None):
    """
    kp: np.ndarray [T, 17, 3]  (x, y, conf)  픽셀 좌표
    return: np.ndarray [1, C, T, J]  float32
            C = encoder.num_channels (기본 7)
            J = encoder.num_joints   (기본 25)
    """
    if kp.shape[1] != 17 or kp.shape[2] != 3:
        raise ValueError(f"Expected [T, 17, 3], got {kp.shape}")

    # 1) 좌표 정규화 — 사람 박스 기준 (학습 dataset.py 와 동일)
    out = kp.astype(np.float32).copy()
    valid = out[..., 2] > 0.1
    if valid.any():
        xs, ys = out[..., 0][valid], out[..., 1][valid]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = max(x_max - x_min, 1e-3)
        h = max(y_max - y_min, 1e-3)
        out[..., 0] = (out[..., 0] - x_min) / w
        out[..., 1] = (out[..., 1] - y_min) / h
    elif frame_w is not None and frame_h is not None:
        out[..., 0] /= max(frame_w, 1)
        out[..., 1] /= max(frame_h, 1)

    # 2) 인코더 적용 → [C, T, J]
    tensor = _get_encoder().encode(out)
    return tensor[None, ...].astype(np.float32)


# 호환성 — 옛 이름
keypoints_to_pseudo_image_with_encoder = keypoints_to_pseudo_image


def get_input_shape(T=60):
    """모델 입력 shape 반환 — main 에서 모델 로드 시 사용."""
    enc = _get_encoder()
    return (1, enc.num_channels, T, enc.num_joints)
