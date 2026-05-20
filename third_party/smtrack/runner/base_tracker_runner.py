from abc import abstractmethod

from smtrack.builder.tracker_builer import build_tracker
from smtrack.builder.motion_builder import build_motion

class BaseTrackerRunner:

    def __init__(self, tracker=None, motion=None):

        self.tracker = None
        self.motion = None

        if tracker != None:
            self.tracker = build_tracker(tracker)

        if motion != None:
            self.motion = build_motion(motion)

    def set_tracker(self, tracker):
        if isinstance(tracker, str):
            tracker = build_motion(tracker)
        self.tracker = tracker

    def set_motion(self, motion):
        if isinstance(motion, str):
            motion = build_motion(motion)
        self.motion = motion

    @abstractmethod
    def run_tracker(self, *args, **kwargs):
        pass
