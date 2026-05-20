from abc import abstractmethod
from addict import Dict
import torch

class BaseTracker:
    """Base tracker model.

    Args:
        momentums (dict[str:float], optional): Momentums to update the buffers.
            The `str` indicates the name of the buffer while the `float`
            indicates the momentum. Default to None.
        num_frames_retain (int, optional). If a track is disappeared more than
            `num_frames_retain` frames, it will be deleted in the memo.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, momentums=None, num_frames_retain=30):
        # super().__init__(init_cfg)
        if momentums is not None:
            assert isinstance(momentums, dict), 'momentums must be a dict'
        self.momentums = momentums
        self.num_frames_retain = num_frames_retain
        self.fp16_enabled = False

        self.reset()

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()

    @property
    def empty(self):
        """Whether the buffer is empty or not."""
        return False if self.tracks else True

    @property
    def ids(self):
        """All ids in the tracker."""
        return list(self.tracks.keys())

    @property
    def with_reid(self):
        """bool: whether the framework has a reid model"""
        return hasattr(self, 'reid') and self.reid is not None

    def update(self, **kwargs):
        """Update the tracker.

        Args:
            kwargs (dict[str: Tensor | int]): The `str` indicates the
                name of the input variable. `ids` and `frame_ids` are
                obligatory in the keys.
        """

        memo_items = [k for k, v in kwargs.items() if v is not None]
        rm_items = [k for k in kwargs.keys() if k not in memo_items]
        for item in rm_items:
            kwargs.pop(item)
        if not hasattr(self, 'memo_items'):
            self.memo_items = memo_items
        else:
            assert memo_items == self.memo_items

        assert 'ids' in memo_items
        num_objs = len(kwargs['ids'])
        id_indice = memo_items.index('ids')
        assert 'frame_ids' in memo_items
        frame_id = int(kwargs['frame_ids'])
        if isinstance(kwargs['frame_ids'], int):
            kwargs['frame_ids'] = torch.tensor([kwargs['frame_ids']] * num_objs)
        # cur_frame_id = int(kwargs['frame_ids'][0])
        for k, v in kwargs.items():
            if len(v) != num_objs:
                raise ValueError()

        for obj in zip(*kwargs.values()):
            id = int(obj[id_indice])
            if id in self.tracks:
                self.update_track(id, obj)
            else:
                self.init_track(id, obj)

        self.pop_invalid_tracks(frame_id)

    def pop_invalid_tracks(self, frame_id):
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['frame_ids'][-1] >= self.num_frames_retain:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def update_track(self, id, obj):
        """Update a track."""
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                m = self.momentums[k]
                self.tracks[id][k] = (1 - m) * self.tracks[id][k] + m * v
            else:
                self.tracks[id][k].append(v)
                

    def init_track(self, id, obj):
        """Initialize a track."""
        self.tracks[id] = Dict()
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                self.tracks[id][k] = v
            else:
                self.tracks[id][k] = [v]

    @abstractmethod
    def track(self, *args, **kwargs):
        """Tracking forward function."""
        pass