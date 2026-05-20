model = dict(
    type='ByteTrackerRunner',
    tracker=dict(
        type='ByteTracker',                                                                                                                   
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=False,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        use_cate_match=False,
        use_second_match_case=False,
        num_frames_retain=30),
    motion=dict(type='KalmanFilter'))
    