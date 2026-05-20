from smutils.bbox.result_transforms import outs2results
from smutils.bbox.transforms import bbox_cxcyah_to_xyxy
from .base_tracker_runner import BaseTrackerRunner

import numpy as np
import torch

class ByteTrackerRunner(BaseTrackerRunner):
    def __init__(self, tracker=None, motion=None):
        super().__init__(tracker=tracker, motion=motion)
        pass

    def run_tracker(self, det_bboxes, det_labels, frame_id, num_classes=1):
        track_bboxes, track_labels, track_ids = self.tracker.track(
            motion=self.motion,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id)
        
        track_results = outs2results(
                bboxes=track_bboxes,
                labels=track_labels,
                ids=track_ids,
                num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'],
            )

    def get_tracks(self):
        track_bboxes = np.zeros((0, 4))
        for track_id in self.tracker.tracks.keys():
            track_bboxes = np.concatenate(
                    (track_bboxes, self.tracker.tracks[track_id].mean[:4][None]), axis=0)
            
        track_bboxes = torch.from_numpy(track_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        return track_bboxes
    
    def get_track_ids(self):
        return list(self.tracker.tracks.keys())
