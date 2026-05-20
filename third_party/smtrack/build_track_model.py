from smtrack.build_track_model_mm import build_track_model_mm

__track_model_builders__ = {
    "mm_mot_yolo" : build_track_model_mm,
    "mm_sot" : build_track_model_mm,
}

def build_track_model(args):
    assert args.type in __track_model_builders__, \
        f'not found pose model type : {args.type}'

    model_builder = __track_model_builders__[args.type]
    return model_builder(args)
