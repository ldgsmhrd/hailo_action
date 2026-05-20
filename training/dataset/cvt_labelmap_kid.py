import copy

## v22 labelmap
action_clip_margin = 30
action_clip_interval = 15
cvt_labelmap_v22_action_upper=[
    [0, "없음", 0],
    [1, "허리 구부리기", 1],
    [2, "허리 펴기", 2],
    [3, "먹기", -1],
    [4, "펀치", 3],
    [5, "휘두르기", -1],
    [6, "손흔들기", 4],
    [7, "가리키기", 5],
    [8, "밀치기", -1],
    [9, "끼적이기", -1],
    [10, "기타", -1],
    [11, "손뼉치기", 6],
    [12, "손올리기", 7],
    [13, "손내리기", 8],
    [14, "상체-구부리기", -1],
    [15, "상체-펴기", -1],
    [16, "식별불가", -1],
    [17, "물건 짚기", 5],
    [18, "물건 놓기", 6],
    [19, "계산 하기", 7],

]

simple_labelmap_v22_action_upper = [
    #idx, name, split_interval, min_clip_len, use_margin, check_train_frame
    [0, "없음", 1, 0, -5],
    [1, "허리 구부리기", action_clip_interval, 15, action_clip_margin],
    [2, "허리 펴기", action_clip_interval, 15, action_clip_margin],    
    [3, "펀치", action_clip_interval, 5, action_clip_margin],
    [4, "손흔들기", action_clip_interval, 60, action_clip_margin],
    [5, "가리키기", action_clip_interval, 100, action_clip_margin],
    [6, "손뼉치기", action_clip_interval, 45, action_clip_margin],
    [7, "손올리기", action_clip_interval, 7, action_clip_margin],
    [8, "손내리기", action_clip_interval, 7, action_clip_margin],
    
]

cvt_labelmap_v22_action_lower=[
    [0, "없음", 0],
    [1, "앉기", 1],
    [2, "일어서기", 2],
    [3, "서성이기", 3],
    [4, "걷기", 4],
    [5, "달리기", 5],     
    [6, "기어가기", 6],   
    [7, "점프-제자리", 7], 
    [8, "넘어짐", 8],
    [9, "떨어짐", -1],
    [10, "킥", 9],      
    [11, "턴", 10],
    [12, "점프-두발", 11], 
    [13, "기타", -1],
    [14, "외발점프", 12],
    [15, "외발점프-제자리", 13],
    [16, "한발내딛기", -1],
    [17, "90도턴", -1],
    [18, "180도턴", -1],
    [19, "360도턴", -1],
    [20, "밀쳐짐+당겨짐", -1],
    [21, "식별불가", -1],
    [22, "걷기-무릎", -1],
    [23, "경보", -1],
    [23, "뛰어내리기", -1],
    [23, "짚고일어서기", -1],
    [23, "기어가기-앉기", -1],
    [23, "넘어짐-엉덩방아", -1],
    [28, "눕기", 14],
]

simple_labelmap_v22_action_lower = [
    [0, "없음", 1, 0, -5],
    [1, "앉기", action_clip_interval, 15, action_clip_margin],
    [2, "일어서기", action_clip_interval, 25, action_clip_margin],
    [3, "서성이기", action_clip_interval, 100, action_clip_margin],
    [4, "걷기", action_clip_interval, 60, action_clip_margin],
    [5, "달리기", action_clip_interval, 20, action_clip_margin],
    [6, "기어가기", action_clip_interval, 60, action_clip_margin],
    [7, "점프-제자리", action_clip_interval, 7, action_clip_margin],  
    [8, "넘어짐", action_clip_interval, 20, action_clip_margin],
    [9, "킥", action_clip_interval, 15, action_clip_margin],
    [10, "턴", action_clip_interval, 120, action_clip_margin],
    [11, "점프-두발", action_clip_interval, 7, action_clip_margin],
    [12, "외발점프", action_clip_interval, 7, action_clip_margin],
    [13, "외발점프-제자리", action_clip_interval, 7, action_clip_margin],
    [14, "눕기", action_clip_interval, 70, action_clip_margin],


]



cvt_list = {
    
    'action_upper' : simple_labelmap_v22_action_upper,
    'action_lower': simple_labelmap_v22_action_lower,

}

def cvt_labelmap_v22(annotations):
    ret_annotations = copy.deepcopy(annotations)
    for i, anno in enumerate(annotations):
        action_id = anno['action_id']
        upper = action_id['action_upper']
        lower = action_id['action_lower']        
        
        new_upper = cvt_labelmap_v22_action_upper[upper][2]
        new_lower = cvt_labelmap_v22_action_lower[lower][2]        
        
        ret_annotations[i]['action_id']['action_upper'] = new_upper
        ret_annotations[i]['action_id']['action_lower'] = new_lower
        
    
    return ret_annotations



