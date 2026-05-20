"""
ActionResNet HEF 출력 후처리.
출력: [1, num_classes] logits (또는 softmax 확률).
"""
import numpy as np


def postprocess_action(raw_output):
    """
    return: (cls_idx, score, prob_vector)
    """
    arr = np.asarray(raw_output).squeeze()
    if arr.ndim == 0:
        return 0, 0.0, np.array([1.0], dtype=np.float32)

    # softmax 변환 (이미 확률이면 정규화만)
    if arr.min() < 0 or arr.max() > 1.0:
        # logits
        e = np.exp(arr - arr.max())
        probs = e / e.sum()
    else:
        # 이미 0~1, 합 1 가정
        s = arr.sum()
        probs = arr / s if s > 0 else arr

    cls_idx = int(np.argmax(probs))
    score = float(probs[cls_idx])
    return cls_idx, score, probs
