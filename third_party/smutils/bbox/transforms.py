import copy
import torch

#numpy 타입, list 타입, 배치단위 처리 추가 필요
def xyxy_to_xywh(xyxy):    
    xywh = copy.deepcopy(xyxy)
    xywh[2] = xyxy[2]-xyxy[0]
    xywh[3] = xyxy[3]-xyxy[1]
    return xywh

def xywh_to_xyxy(xywh):    
    xyxy = copy.deepcopy(xywh)
    xyxy[2] = xywh[2]+xywh[0]
    xyxy[3] = xywh[3]+xywh[1]
    return xyxy

def xyxy_to_cwh(xyxy):
    cwh = copy.deepcopy(xyxy)
    cwh[2] = xyxy[2]-xyxy[0]
    cwh[3] = xyxy[3]-xyxy[1]
    cwh[0] = xyxy[0]+cwh[2]*0.5
    cwh[1] = xyxy[1]+cwh[3]*0.5
    return cwh

def cwh_to_xyxy(cwh):
    xyxy = copy.deepcopy(cwh)
    xyxy[0] = cwh[0]-cwh[2]*0.5
    xyxy[1] = cwh[1]-cwh[3]*0.5
    xyxy[2] = cwh[2]+xyxy[0]
    xyxy[3] = cwh[3]+xyxy[1]
    return xyxy

def bbox_xyxy_to_cxcyah(bboxes):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, ratio, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
    cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    xyah = torch.stack([cx, cy, w / h, h], -1)
    return xyah


def bbox_cxcyah_to_xyxy(bboxes):
    """Convert bbox coordinates from (cx, cy, ratio, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, ratio, h = bboxes.split((1, 1, 1, 1), dim=-1)
    w = ratio * h
    x1y1x2y2 = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    return torch.cat(x1y1x2y2, dim=-1)

