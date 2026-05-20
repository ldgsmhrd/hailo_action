import numpy as np
# import mmcv
import copy
import cv2
from tqdm import tqdm

import smutils.utils_os as utils

# from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

def get_palette(n):
    state = np.random.get_state()
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(n, 3))
    np.random.set_state(state)
    colors = [tuple(c) for c in palette]
    return colors

def vis_instance_segmentation(image, masks=None, bboxes=None, colors=None):
    if colors is None:
        if masks is not None:
            n = masks.shape[0]
        else:
            n = bboxes.shape[0]
        colors = get_palette(n)

    vis_img = image.copy()

    if masks is not None:
        mask_img = np.zeros_like(image)
        for mask, color in zip(reversed(masks), colors):
            mask_img[mask] = color
        
        mask = mask_img > 0    
        vis_img[mask] = vis_img[mask]*0.3 + mask_img[mask]*0.7

    if bboxes != None:
        for bbox, color in zip(reversed(bboxes), colors):
            vis_img = draw_single_bbox_and_label(vis_img, bbox, f'{bbox[-1]:.3f}', color, 2, color, 2, 0.7, box_type='xyxy')

    return vis_img

def vis_pose_coco_skeleton(image, pose_results,):
    """
        pose_results (dict):
            keypoints : ndarray
            bbox : ndarray
    """
    if not isinstance(pose_results, list):
        pose_results = [pose_results]

    for pose in pose_results:
        kps = pose['keypoints'] if pose['keypoints'].ndim == 2 else pose['keypoints'][0]    
        # kps[:, :2] += pose['bbox'][:2]
        image = vis_pose_coco_skeleton_one_person(image, kps)

    return image

def vis_pose_coco_skeleton_one_person(image, keypoints):
    # COCO ordered keypoints:
    # 0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear  5:left_shoulder
    # 6:right_shoulder  7:left_elbow  8:right_elbow  9:left_wrist  10:right_wrist
    # 11:left_hip  12:right_hip  13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

    # define connections between keypoints for COCO dataset
    coco_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10),  # right arm
        (5, 6), (5, 11), (6, 12), (11, 12),  # body
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16)  # right leg
    ]

    # define colors for different parts of the body
    colors = {
        'face': (255, 210, 127),  # blue
        'left_body': (0, 165, 255),  # orange
        'right_body': (0, 255, 0),  # green
        'body': (255, 210, 127)  # blue
    }

    # map keypoints to colors
    keypoint_colors = [colors['face']] * 5 + [colors['left_body'], colors['right_body']] * 6

    # map pairs to colors
    pair_colors = [colors['face']] * 4 + [colors['left_body']] * 2 + [colors['right_body']] * 2 + \
                [colors['body']] * 4 + [colors['left_body']] * 2 + [colors['right_body']] * 2


    # draw lines (skeleton) on the image
    for i, pair in enumerate(coco_pairs):
        kp1, kp2 = pair
        if keypoints[kp1, 2] > 0.5 and keypoints[kp2, 2] > 0.5:
            cv2.line(image, (int(keypoints[kp1, 0]), int(keypoints[kp1, 1])),
                    (int(keypoints[kp2, 0]), int(keypoints[kp2, 1])), pair_colors[i], 2)

    # draw keypoints on the image
    for i, keypoint in enumerate(keypoints):
        x, y, score = keypoint
        if score > 0.5:  # if score is less than 0.5, the keypoint is not considered as detected
            cv2.circle(image, (int(x), int(y)), 5, keypoint_colors[i], thickness=-1, lineType=cv2.FILLED)

    return image

def draw_single_bbox_and_label(img, bbox, label, box_color, box_thk, txt_color, txt_thk, txt_scale, box_type='xywh'):
    txt_xy = (int(bbox[0]), int(bbox[1]-20))
    box_color = tuple(map(int, box_color))
    txt_color = tuple(map(int, txt_color))
    if int(bbox[1]-20) < 20:
        txt_xy = (int(bbox[0]), int(bbox[1]+bbox[3]))
    if box_type == 'xywh':
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), box_color, box_thk)
    elif box_type == 'xyxy':
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, box_thk)

    cv2.putText(img, label, txt_xy, cv2.FONT_HERSHEY_DUPLEX, txt_scale, txt_color, txt_thk, 1)
    return img

def draw_bboxes_list(image, bboxes, box_color, box_thk, box_type, labels=None):
    if labels == None:
        labels = ['']*len(bboxes)

    for bbox, label in zip(bboxes, labels):
        if label == '' and bbox.shape[0] == 6:
            label = str(int(bbox[0]))
            bbox = bbox[1:]
        
        image = draw_single_bbox_and_label(image, bbox, label, box_color, box_thk, (255, 255, 255), 2, 1.5, box_type=box_type)
    return image

def draw_bboxes_dict(image, bboxes, box_color, box_thk, box_type):
    for track_id, bbox in bboxes.items():
        label = f'{track_id}'
        image = draw_single_bbox_and_label(image, bbox, label, box_color, box_thk, (255, 255, 255), 2, 1.5, box_type=box_type)
    return image

def draw_and_save_bboxes(image, save_path, bboxes, box_color, box_thk, box_type='xywh'):
    
    if isinstance(image, str):
        image = cv2.imread(image)
    
    if isinstance(bboxes, dict):
        image = draw_bboxes_dict(image, bboxes, box_color, box_thk, box_type)
    else:
        image = draw_bboxes_list(image, bboxes, box_color, box_thk, box_type)

    cv2.imwrite(save_path, image)


def draw_and_save_single_bbox(image, save_path, bbox, label, box_color, box_thk, box_type='xywh'):
    
    if isinstance(image, str):
        image = cv2.imread(image)
    
    if bbox.shape[0] == 6:
        bbox = bbox[1:]
        
        image = draw_single_bbox_and_label(image, bbox, label, box_color, box_thk, (255, 255, 255), 2, 1.5, box_type=box_type)
    cv2.imwrite(save_path, image)        



# def draw_single_instance(img_path, pose_model, pose, label, box_color, box_thk, txt_color, txt_thk, txt_scale):
#     img = vis_pose_result(pose_model, img_path, pose)
#     cv2.putText(img, label, (int(bbox[0]), int(bbox[1]-20)), cv2.FONT_HERSHEY_DUPLEX, txt_scale, txt_color, txt_thk, 1)
#     return img


def make_video(folder_or_paths, save_path, fps=30, i_size='half'):

    if isinstance(folder_or_paths, str):
        _, path_list = utils.search_file(folder_or_paths, '.jpg')
    else:
        path_list = folder_or_paths

    img = cv2.imread(path_list[0])
    h, w, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if i_size == 'half':
        h = int(h/2)
        w = int(w/2)
    
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    # prog_bar = mmcv.ProgressBar(len(path_list))
    N = len(path_list)
    for i in tqdm(range(N)):
        # prog_bar.update()
        path = path_list[i]
        img = cv2.imread(path)

        if i_size == 'half':
            img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_AREA)

        video_writer.write(img)

    print()

    video_writer.release()


