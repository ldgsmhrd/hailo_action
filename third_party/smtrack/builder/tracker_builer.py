from smtrack.models.trackers.byte_tracker import ByteTracker

__track_builders__ = {
    "ByteTracker" : ByteTracker,
}

def build_tracker(args):
    assert args.type in __track_builders__, \
            f'not found track model type : {args.type}'
    
    track_builder = __track_builders__[args.type]
    track_args = args.copy()
    track_args.pop('type')
    return track_builder(**track_args)