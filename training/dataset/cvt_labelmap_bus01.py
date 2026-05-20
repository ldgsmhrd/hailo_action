import copy

## v22 labelmap
action_clip_margin = 30
action_clip_interval = 15
cvt_labelmap_v22_action_upper = [
    [0, "없음", 0],
    [1, "허리 구부리기", -1],
    [2, "허리 펴기", -1],
    [3, "먹기", -1],
    [4, "펀치", 1],
    [5, "휘두르기", -1],
    [6, "손흔들기", 0],
    [7, "가리키기", -1],
    [8, "밀치기", 1],
    [9, "끼적이기", -1],
    [10, "기타", -1],
    [11, "손뼉치기", 0],
    [12, "손올리기", 0],
    [13, "손내리기", 0],
    [14, "상체-구부리기", -1],
    [15, "상체-펴기", -1],
    [16, "식별불가", -1],
    [17, "물건 짚기", -1],
    [18, "물건 놓기", -1],
    [19, "계산 하기", -1],
    [20, "한손들기", 0],
    [21, "양손들기", 0],
]

simple_labelmap_v22_action_upper = [
    # idx, name, split_interval, min_clip_len, use_margin, check_train_frame
    [0, "없음", 1, 0, -5],
    [1, "펀치", action_clip_interval, 8, action_clip_margin],
]

cvt_labelmap_v22_action_lower = [
    [0, "없음", 0],
    [1, "앉기", 1],
    [2, "일어서기", 2],
    [3, "서성이기", -1],
    [4, "걷기", 3],
    [5, "달리기", 4],
    [6, "기어가기", -1],
    [7, "점프-제자리", 0],
    [8, "넘어짐", 5],
    [9, "떨어짐", -1],
    [10, "킥", -1],
    [11, "턴", -1],
    [12, "점프-두발", 0],
    [13, "기타", -1],
    [14, "외발점프", 0],
    [15, "외발점프-제자리", 0],
    [16, "한발내딛기", -1],
    [17, "90도턴", -1],
    [18, "180도턴", -1],
    [19, "360도턴", -1],
    [20, "밀쳐짐+당겨짐", -1],
    [21, "식별불가", -1],
    [22, "걷기-무릎", -1],
    [23, "경보", -1],
    [24, "뛰어내리기", -1],
    [25, "짚고일어서기", -1],
    [26, "기어가기-앉기", -1],
    [27, "넘어짐-엉덩방아", -1],
    [28, "눕기", 6],
    [29, "허리 구부리기", 0],
    [30, "허리펴기", -1],
    [31, "서있기", 7],
    [32, "앉아있기", 8],
    [33, "기타", -1],
    [34, "기타", -1],
    [35, "기타", -1],
    [36, "기타", -1],
    [37, "기타", -1],
    [38, "한발들기 포즈", -1],
]

simple_labelmap_v22_action_lower = [
    [0, "없음", 1, 0, -5],
    [1, "앉기", action_clip_interval, 15, action_clip_margin],
    [2, "일어서기", action_clip_interval, 20, action_clip_margin],
    [3, "걷기", action_clip_interval, 60, action_clip_margin],
    [4, "달리기", action_clip_interval, 20, action_clip_margin],
    [5, "넘어짐", action_clip_interval, 20, action_clip_margin],
    [6, "눕기", action_clip_interval, 60, action_clip_margin],
    [7, "서있기 포즈", action_clip_interval, 55, action_clip_margin],
    [8, "앉아있기 포즈", action_clip_interval, 55, action_clip_margin],
]


cvt_list = {
    "action_upper": simple_labelmap_v22_action_upper,
    "action_lower": simple_labelmap_v22_action_lower,
}


def cvt_labelmap_v22(annotations):
    ret_annotations = copy.deepcopy(annotations)
    for i, anno in enumerate(annotations):
        action_id = anno["action_id"]
        upper = action_id["action_upper"]
        lower = action_id["action_lower"]

        new_upper = cvt_labelmap_v22_action_upper[upper][2]
        new_lower = cvt_labelmap_v22_action_lower[lower][2]

        ret_annotations[i]["action_id"]["action_upper"] = new_upper
        ret_annotations[i]["action_id"]["action_lower"] = new_lower

    return ret_annotations
