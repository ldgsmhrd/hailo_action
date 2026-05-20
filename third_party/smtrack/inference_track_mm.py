import numpy as np
from mmtrack.apis import inference_sot

def cvt_track_format(track_result):
    # track_bboxes = track_result['track_bboxes']

    # np.array()
    # for class_label, bboxes in enumerate(track_bboxes):
    #     np.concatenate((ids[labels == i, None], bboxes[labels == i, :]), axis=1)
    pass

def inference_track_mot_yolov5(model, det_bboxes, det_labels, frame_id, num_classes=1, return_format=None):
    """
    Args:
        model (): 
        det_bboxes (type): shape, 설명
        det_labels ():
        frame_id (int): frame id가 0 이면 트래커 리셋함
        num_classes : 사용하는 클래스 숫자, 사용하는 클래스 인덱스들의 최대 인덱스+1를 기입
    Return:
        track_result (list(np.array)) : list의 인덱스는 클래스 번호를 의미함
    """

    data = {}
    data['img'] = [None]
    data['img_metas'] = [[dict(frame_id=frame_id)]]
    data['det_bboxes'] = det_bboxes
    data['det_labels'] = det_labels
    data['num_classes'] = num_classes

    track_result = model(return_loss=False, rescale=True, **data)

    if return_format == 'cvt':
        return cvt_track_format(track_result)

    return track_result

def inference_track_sot(model, image, init_bbox, frame_id):
    '''
    Args:
        model : torch model
        image (numpy.array) : 
        init_bbox (list) : [x, y, x, y]
        frame_id (int)
    Return:
        result (dict) : 
            key : 'track_bboxes'
            value (numpy.array) : [xyxy, score]
    '''
    return inference_sot(model, image, init_bbox, frame_id=frame_id)

__mm_inference_func__ = {
    "mm_mot_yolo" : inference_track_mot_yolov5,
    "mm_sot" : inference_track_sot,
}

def inference_track_mm(type, **args):
    assert type in __mm_inference_func__, \
        f"not found track model type : {type}"
    
    model_inference_func = __mm_inference_func__[type]
    return model_inference_func(**args)



