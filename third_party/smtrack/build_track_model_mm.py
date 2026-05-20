from mmtrack.apis import init_model

def build_track_model(args):
    return init_model(args.config, args.checkpoint, device=args.device)

__mm_build_func__ = {
    "mm_mot_yolo" : build_track_model,
    "mm_sot" : build_track_model,
}

def build_track_model_mm(args):
    assert args.type in __mm_build_func__, \
        f"not found track model type : {args.type}"
    
    model_builder = __mm_build_func__[args.type]
    return model_builder(args)
