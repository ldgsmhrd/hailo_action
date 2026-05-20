from smtrack.runner.byte_tracker_runner import ByteTrackerRunner

__runner_builders__ = {
    "ByteTrackerRunner" : ByteTrackerRunner,
}

def build_track_runner(args):
    assert args.type in __runner_builders__, \
            f'not found track model type : {args.type}'
    
    runner_builder = __runner_builders__[args.type]
    runner_args = args.copy()
    runner_args.pop('type')
    return runner_builder(**runner_args)