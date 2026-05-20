

class SMByterTracker:
    def __init__(self, track_cfg, motion_cfg):
        self.tracker = ByteTracker(track_cfg)
        self.motion = KalmanFilter(motion_cfg)
    
    def inference(self, det_bboxes, det_labels, frame_id, num_classes=1):
        if frame_id == 0:
            self.tracker.reset()

        track_bboxes, track_labels, track_ids = self.tracker.track(
                img = None,
                img_metas = [dict(frame_id=frame_id)],
                model = self,
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
            track_bboxes=track_results['bbox_results'])