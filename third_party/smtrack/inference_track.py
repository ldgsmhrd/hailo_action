from smtrack.inference_track_mm import inference_track_mm

__inference_func__ = {
    "mm_mot_yolo" : inference_track_mm,
    "mm_sot" : inference_track_mm,
}

def inference_track(type, **args):
    assert type in __inference_func__, \
        f"not found pose model type : {type}"

    model_inference_func = __inference_func__[type]
    return model_inference_func(type, **args)
