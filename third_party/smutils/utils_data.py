import random
import copy
import pickle
import shutil
import os
from smutils.utils_os import search_file, create_directory

def load_pkl_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_datas = pickle.load(f)
    return pkl_datas

def save_pkl_data(pkl_data, save_path):  
    with open(save_path, 'wb') as f:  
        pickle.dump(pkl_data, f)

def dataset_class_filtering(pkl_datas, active_class):
    if isinstance(pkl_datas, str):
        with open(pkl_datas, 'rb') as f:
            pkl_datas = pickle.load(f)

    filtered_datas = []
    for data in pkl_datas:
        if data['label'] in active_class:
            filtered_datas.append(data)
    return filtered_datas

def print_data_num_per_class(pkl_datas, labelmap):
    if isinstance(pkl_datas, str):
        with open(pkl_datas, 'rb') as f:
            pkl_datas = pickle.load(f)
            
    bins_data = dict()
    for i in range(len(labelmap.keys())):
        bins_data[i] = 0
        
    for data in pkl_datas:    
        bins_data[data['label']] += 1
    
    if isinstance(labelmap, dict):
        for label, cnt in bins_data.items():
            print(f"{labelmap[label]} : {cnt}")
    else:
        for label, cnt in bins_data.items():
            print(f"{labelmap[label][1]} : {cnt}")

def split_train_and_val(datas, class_num, max_train_data_num, max_val_data_num):
            
    data_dict = dict()
    train_list = []
    val_list = []
    for i in range(class_num):
        data_dict[i] = []
        
    for data in datas:
        data_dict[data['label']].append(data)    
    
    for key, data_list in data_dict.items():
        data_num = len(data_list)
        train_data_num = max_train_data_num
        if data_num < max_train_data_num:
            train_data_num = int(data_num * 0.95)
        
        val_data_idx = min(data_num, train_data_num+max_val_data_num)
        
        random.shuffle(data_list)
        train_list.extend(copy.deepcopy(data_list[:train_data_num]))
        val_list.extend(copy.deepcopy(data_list[train_data_num:val_data_idx]))
        
    random.shuffle(train_list)
    random.shuffle(val_list)
    
    return train_list, val_list

def remove_items(original_list, remove_items):
    return [item for item in original_list if item not in remove_items]
    
def make_pkl_info(pkl_name_list, pkl_path_list, key='label'):    
    label_info_dict = dict()
    file_info_dict = dict()
    for name, path in zip(pkl_name_list, pkl_path_list):
        pkl_list = load_pkl_data(path)
        file_info_dict[name] = []
        for pkl in pkl_list:
            label = pkl[key]
            if label not in label_info_dict:
                label_info_dict[label] = []
                
            if label not in file_info_dict[name]:
                file_info_dict[name].append(label)
            
            if name not in label_info_dict[label]:
                label_info_dict[label].append(name)
                
    label_info_dict = dict(sorted(label_info_dict.items()))
    #label_info_dict : 라벨이 포함된 클립 리스트
    #file_info_dict : 클립에 포함되는 라벨 리스트
    return label_info_dict, file_info_dict


# def select_items(pkl_folder, ratio=0.2):
#     #하나에 클립에 여러개의 라벨이 있을 수 있음
#     #최소한의 클립으로 여러 라벨이 골고루 커버가능하도록 함
#     pkl_name_list, pkl_path_list = search_file(pkl_folder, '.pkl')
#     label_info, file_info = make_pkl_info(pkl_name_list, pkl_path_list)

#     # 클립 수로 오름차순으로 정렬
#     sorted_data = copy.deepcopy(dict(sorted(label_info.items(), key=lambda item: len(item[1]))))
#     item_dict = {}
#     select_item_list = []
#     for key in label_info.keys():
#         item_dict[key] = []
    
#     while True:
#         #클립 수가 가장 적은 라벨 선택
#         label = list(sorted_data.keys())[0]
#         clip_list = sorted_data[label]

#         #선택할 클립의 수 결정
#         sample_size = max(1, int(len(clip_list)*ratio)) # 선택해야하는 클립 수
#         sample_size = max(0, sample_size - len(item_dict[label])) # 선택해야하는 클립 수에 이미 선택된 클립수를 뺌
        
#         #클립 선택
#         selected_clip = random.sample(clip_list, sample_size)
#         select_item_list.extend(selected_clip)
#         for clip in selected_clip:
#             l_list = file_info[clip]
#             for l in l_list:
#                 item_dict[l].append(clip)
        
#         #선택된 클립 제거
#         for key, value_list in sorted_data.items():
#             sorted_data[key] = [val for val in value_list if val not in selected_clip]
        
#         del sorted_data[label]
        
#         if not sorted_data:
#             break
        
#         # 클립 수로 오름차순으로 정렬
#         sorted_data = dict(sorted(sorted_data.items(), key=lambda item: len(item[1])))
    
#     remain_item_list = remove_items(pkl_name_list, select_item_list)

#     return select_item_list, remain_item_list


def split_train_and_val_v22(data_folder, save_folder, category_info, total_data_num=100, ratio=0.8):
    #변수 초기화 및 저장 폴더 생성
    split_datas = dict()
    for category, num in category_info.items():
        split_datas[category] = dict()
        for label in range(num):
            split_datas[category][label] = dict()
            split_datas[category][label]['train'] = []
            split_datas[category][label]['val'] = []
            folder_train = os.path.join(save_folder, 'train', category, f'{label:02d}')
            folder_val = os.path.join(save_folder, 'val', category, f'{label:02d}')
            create_directory(folder_train)
            create_directory(folder_val)


    for category, num in category_info.items():
        for label in range(num):

            folder = os.path.join(data_folder, category, f'{label:02d}')
            name_list, path_list = search_file(folder, '.pkl')

            data_num = min(len(path_list), total_data_num)
            train_num = int( data_num * ratio )
            
            random.shuffle(path_list)
            split_datas[category][label]['train'] = path_list[:train_num]
            split_datas[category][label]['val'] = path_list[train_num:data_num]

            if train_num < total_data_num*ratio*0.5:
                split_datas[category][label]['train'] = split_datas[category][label]['train']*2
            
    for category, num in category_info.items():
        for label in range(num):
            for mode, path_list in split_datas[category][label].items():
                folder = os.path.join(save_folder, mode, category, f'{label:02d}')
                for path in path_list:
                    name = path.split('/')[-1]
                    save_path = os.path.join(folder, name)
                    shutil.copy(path, save_path)

def load_labelmap(path):
    label_map = [x.strip() for x in open(path).readlines()]
    return label_map
