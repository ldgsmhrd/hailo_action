from smtrack.models.motion.kalman_filter import KalmanFilter

__motion_builders__ = {
    "KalmanFilter" : KalmanFilter,
}

def build_motion(args):
    assert args.type in __motion_builders__, \
            f'not found motion model type : {args.type}'
    
    motion_builder = __motion_builders__[args.type]
    motion_args = args.copy()
    motion_args.pop('type')
    return motion_builder(**motion_args)
