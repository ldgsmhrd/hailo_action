"""
YOLOv8-pose HEF (Hailo) raw 출력 post-processing.

HEF 출력 구조 (Hailo Model Zoo yolov8s_pose 표준):
  3 스케일 (stride 32 / 16 / 8) × 3 head (bbox DFL / objectness / keypoint)
  = 총 9 텐서.

  스케일 1 (20x20, stride 32):
    bbox : (20, 20, 64)   ← DFL 4×16
    obj  : (20, 20, 1)    ← logit
    kpt  : (20, 20, 51)   ← (17, 3)
  스케일 2 (40x40, stride 16): 64/1/51
  스케일 3 (80x80, stride 8) : 64/1/51

디코딩:
  bbox = DFL(softmax × [0..15] 합) → 4 거리(l, t, r, b)
       → (cx - l, cy - t, cx + r, cy + b) × stride
  conf = sigmoid(obj)
  kpt  : 채널 51을 (17, 3) 로 reshape
         x = (cell_x + sigmoid(dx) * 2 - 0.5 + ??) * stride  ← Hailo YOLOv8-pose는 raw offset 직접 출력
         실제로 Ultralytics YOLOv8-pose 의 kpts_decode:
            kpts[:, 0::3] = kpts[:, 0::3] * 2.0 + (anchors[:, 0] - 0.5)
            kpts[:, 1::3] = kpts[:, 1::3] * 2.0 + (anchors[:, 1] - 0.5)
            kpts[:, 2::3] = sigmoid(kpts[:, 2::3])
            그 후 anchor 단위로 stride 곱
"""
import numpy as np


REG_MAX = 16   # YOLOv8 DFL: 4 sides × 16 bins
NUM_KP = 17    # COCO


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _dfl_decode(bbox_tensor):
    """bbox_tensor: (H, W, 64) → distances (H*W, 4)."""
    h, w, _ = bbox_tensor.shape
    arr = bbox_tensor.reshape(h * w, 4, REG_MAX)
    prob = _softmax(arr, axis=-1)
    bins = np.arange(REG_MAX, dtype=np.float32)
    dist = (prob * bins).sum(axis=-1)        # (H*W, 4)
    return dist


def _grid_anchors(h, w):
    """grid cell 중심 좌표 (H*W, 2)  — 단위는 'cell'."""
    cy, cx = np.mgrid[0:h, 0:w]
    grid = np.stack([cx, cy], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5
    return grid


def _decode_scale(bbox_t, obj_t, kpt_t, stride):
    """단일 스케일 디코딩.
    return:
      boxes : (N, 4)  xyxy  픽셀 좌표
      scores: (N,)
      kpts  : (N, 17, 3)  픽셀 좌표 + sigmoid conf
    """
    h, w, _ = bbox_t.shape
    anchors = _grid_anchors(h, w)

    dist = _dfl_decode(bbox_t)               # (H*W, 4) cell 단위 (l, t, r, b)
    x1 = (anchors[:, 0] - dist[:, 0]) * stride
    y1 = (anchors[:, 1] - dist[:, 1]) * stride
    x2 = (anchors[:, 0] + dist[:, 2]) * stride
    y2 = (anchors[:, 1] + dist[:, 3]) * stride
    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    # keypoints: (H*W, 17, 3) — x, y, raw_vis
    kpt = kpt_t.reshape(h * w, NUM_KP, 3).astype(np.float32)
    kpt[:, :, 0] = (kpt[:, :, 0] * 2.0 + (anchors[:, 0:1] - 0.5)) * stride
    kpt[:, :, 1] = (kpt[:, :, 1] * 2.0 + (anchors[:, 1:2] - 0.5)) * stride
    kpt[:, :, 2] = _sigmoid(kpt[:, :, 2])     # 0~1

    # cls head 점수 사용 (Hailo Model Zoo v2.18.0+ HEF 는 cls head 가 sigmoid 까지
    # 모델에 baked-in 되어 있어 0~1 범위 값이 바로 나옴. raw logit 일 경우 sigmoid 적용.)
    obj_arr = obj_t.reshape(-1).astype(np.float32)
    if obj_arr.max() > 1.0 or obj_arr.min() < 0.0:
        scores = _sigmoid(obj_arr)
    else:
        scores = obj_arr   # 이미 [0,1] 범위 → sigmoid 중복 적용 안 함

    return boxes, scores, kpt


def _nms_xyxy(boxes, scores, iou_thr=0.45, top_k=300):
    """간단한 NMS (numpy). 사람 단일 클래스 가정."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    keep = []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)

    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        if i == len(boxes) - 1:
            break
        xx1 = np.maximum(x1[i], x1[i+1:])
        yy1 = np.maximum(y1[i], y1[i+1:])
        xx2 = np.minimum(x2[i], x2[i+1:])
        yy2 = np.minimum(y2[i], y2[i+1:])
        iw = (xx2 - xx1).clip(0)
        ih = (yy2 - yy1).clip(0)
        inter = iw * ih
        union = areas[i] + areas[i+1:] - inter + 1e-9
        iou = inter / union
        suppressed[i+1:] |= (iou > iou_thr)
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# stride 매핑: HEF 출력 grid 크기 → stride
# 모델 입력 640 기준: 80→8, 40→16, 20→32
STRIDE_BY_GRID = {80: 8, 40: 16, 20: 32}

# bbox / obj / kpt 채널 수로 head 구분
def _classify_head(shape):
    """shape: (H, W, C) → 'bbox' / 'obj' / 'kpt'"""
    c = shape[-1]
    if c == 4 * REG_MAX:   # 64
        return 'bbox'
    if c == 1:
        return 'obj'
    if c == NUM_KP * 3:    # 51
        return 'kpt'
    return None


def _group_outputs(named_outputs):
    """{name: ndarray} → [(stride, bbox, obj, kpt), ...] 3 스케일."""
    by_grid = {}
    for name, arr in named_outputs.items():
        arr = np.asarray(arr)
        if arr.ndim == 4:    # batch 차원 제거
            arr = arr[0]
        if arr.ndim != 3:
            continue
        h, w, c = arr.shape
        if h != w or h not in STRIDE_BY_GRID:
            continue
        head = _classify_head(arr.shape)
        if head is None:
            continue
        by_grid.setdefault(h, {})[head] = arr

    groups = []
    for h, heads in by_grid.items():
        if {'bbox', 'obj', 'kpt'}.issubset(heads):
            groups.append((STRIDE_BY_GRID[h], heads['bbox'], heads['obj'], heads['kpt']))
    return groups


def postprocess_pose_multi(named_outputs, orig_h, orig_w,
                           conf_thr=0.3, iou_thr=0.45,
                           model_in_h=640, model_in_w=640):
    """
    YOLOv8-pose HEF 의 9개 출력 → 사람 N명 검출 결과.

    named_outputs: dict {tensor_name: ndarray}
                   InferVStreams.infer() 반환값을 그대로 넣으면 됨.
    orig_h, orig_w: 원본 프레임 크기 (HEF 입력 크기와 다를 수 있으니 스케일 복원)
    return: list[{'box': np.array([x1,y1,x2,y2,score]),
                  'keypoints': np.array([17, 3])}]
            좌표는 원본 프레임 픽셀.
    """
    groups = _group_outputs(named_outputs)
    if not groups:
        return []

    all_boxes, all_scores, all_kpts = [], [], []
    for stride, bbox_t, obj_t, kpt_t in groups:
        boxes, scores, kpt = _decode_scale(bbox_t, obj_t, kpt_t, stride)
        mask = scores >= conf_thr
        if not mask.any():
            continue
        all_boxes.append(boxes[mask])
        all_scores.append(scores[mask])
        all_kpts.append(kpt[mask])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    kpts = np.concatenate(all_kpts, axis=0)

    keep = _nms_xyxy(boxes, scores, iou_thr=iou_thr)
    boxes = boxes[keep]
    scores = scores[keep]
    kpts = kpts[keep]

    # 모델 입력 (640x640) → 원본 (orig_h x orig_w) 스케일 복원
    sx = orig_w / float(model_in_w)
    sy = orig_h / float(model_in_h)
    boxes[:, 0] *= sx
    boxes[:, 2] *= sx
    boxes[:, 1] *= sy
    boxes[:, 3] *= sy
    kpts[:, :, 0] *= sx
    kpts[:, :, 1] *= sy

    # 결과 패키징
    results = []
    for b, s, k in zip(boxes, scores, kpts):
        # bbox 가 frame 안으로 들어오는 것만
        b[0] = max(0.0, b[0]); b[1] = max(0.0, b[1])
        b[2] = min(float(orig_w - 1), b[2]); b[3] = min(float(orig_h - 1), b[3])
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        results.append({
            'box': np.array([b[0], b[1], b[2], b[3], float(s)], dtype=np.float32),
            'keypoints': k.astype(np.float32),
        })
    return results


def postprocess_pose(named_outputs, orig_h, orig_w):
    """가장 신뢰도 높은 1명 — test_pipeline 호환용."""
    dets = postprocess_pose_multi(named_outputs, orig_h, orig_w)
    if not dets:
        return np.zeros((NUM_KP, 3), dtype=np.float32)
    return dets[0]['keypoints']
