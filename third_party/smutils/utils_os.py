import os
import json
import numpy as np
import shutil

def search_file(folder, fileEx):
    
    file_path = []
    for path, dirs, files in os.walk(folder):
        if path.split('/')[-1].startswith('.'):
            continue
        file_path.extend([ os.path.join(path, file) for file in files if file.endswith(fileEx)])

    file_path.sort()
    
    file_list = []
    for path in file_path:            
        file_list.append(os.path.basename(path))
    
    return file_list, file_path

def search_folder(folder):
    sub_folder_path_list = []
    sub_folder_name_list = []
    for item in os.listdir(folder): # 해당 폴더 내 모든 파일 및 폴더 추출
        sub_folder = os.path.join(folder, item)

        if os.path.isdir(sub_folder): # 폴더 여부 확인
            sub_folder_path_list.append(sub_folder)
            sub_folder_name_list.append(item)
            
    return sub_folder_name_list, sub_folder_path_list


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            pass
    except OSError:
        print("Error: Failed to create the directory.")



### json 저장할 때 필요한 클래스
class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)  

def save_json(data, path):
    try:
        with open(path, "w") as json_file:
            json.dump(data, json_file)
    except Exception  as e:
        print(e)

def load_json(path):
    try:
        with open(path, "r") as json_file:
            data = json.load(json_file)
    except Exception  as e:
        data = None

    return data

def copy_file(src_path, dest_path):
    try:
        shutil.copy(src_path, dest_path)
    except Exception as e:
        print(f"파일 복사 중 오류 발생: {src_path} -> {dest_path}")
