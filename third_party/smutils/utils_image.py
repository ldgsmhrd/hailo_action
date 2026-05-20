from smutils.bbox.transforms import *

#detector로 찾은 이미지 크롭
def crop_image(image, bboxes, box_type='xyxy'):
    cropped_img = []

    if bboxes.ndim != 1:
        for bbox in bboxes:
            if box_type == 'xywh':
                bbox = xywh_to_xyxy(bbox)
            
            x1,y1,x2,y2 = bbox[:4]
            cropped_img.append(image[int(y1): int(y2), int(x1): int(x2)])
    else:
        bbox = bboxes[:4]
        if box_type == 'xywh':
            bbox = xywh_to_xyxy(bbox)

        x1,y1,x2,y2 = bbox[:4]
        cropped_img.append(image[int(y1): int(y2), int(x1): int(x2)])

    return cropped_img

def crop_box_xyxy(image, crop_bbox):
    h, w, _ = image.shape
    x1, y1, x2, y2 = crop_bbox
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), w)
    y2 = min(int(y2), h)
    return image[y1:y2, x1:x2]
