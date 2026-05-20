from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

from smutils.utils_os import search_file

def crop_clip(input_video_path, output_video_path, crop_area, time_interval=[0, -1], interver_type='time'):
    '''
        crop_area : 동영상에서 잘라낼 영역의 좌상단과 우하단 좌표 (x1, y1, x2, y2)
        time_interval : 동영상에서 잘라낼 시작 구간과 종료 구간 (시작 구간, 종료 구간)
        interver_type : time_interval 단위 
            'time' : 초단위, 소수점 입력 가능
            'frame' : 프레임 단위, 정수로 입력
    '''

    # 동영상 읽기
    clip = VideoFileClip(input_video_path)

    # 구간 설정
    start_time, end_time = time_interval
    if interver_type == 'frame':
        fps = clip.fps
        start_time = start_time / fps
        end_time = end_time / fps

    # 지정한 시간 구간을 잘라내기
    subclip = clip.subclip(start_time, end_time)

    # 지정한 영역을 잘라내기
    cropped_clip = subclip.crop(y1=crop_area[1], y2=crop_area[3], x1=crop_area[0], x2=crop_area[2])

    # 잘라낸 영역을 새 동영상 파일로 저장
    cropped_clip.write_videofile(output_video_path, codec="libx264")

    # 자원 해제
    clip.close()


def make_video(folder_or_paths, save_path, fps=30, half=True):
    
    if isinstance(folder_or_paths, str):
        _, path_list = search_file(folder_or_paths, '.jpg')
    else:
        path_list = folder_or_paths

    # 이미지 시퀀스로부터 동영상 클립 생성
    clip = ImageSequenceClip(path_list, fps=fps)

    # 크기를 절반으로 줄이는 옵션이 활성화된 경우, 크기를 조절합니다.
    if half:
        clip = clip.resize(height=int(clip.h/2), width=int(clip.w/2))

    # 동영상 파일로 저장
    clip.write_videofile(save_path, codec='libx264', bitrate='4000k')

    # 자원 해제
    clip.close()
    

def save_clip_images():

    # 동영상 파일 경로 설정
    input_video_path = "input_video.mp4"

    # 구간 설정
    time_interval = (start_time, end_time)
    intervar_type = 'frame'

    clip = VideoFileClip(input_video_path)

    # 구간 설정
    start_time, end_time = time_interval
    if interver_type == 'frame':
        fps = clip.fps
        start_time = start_time / fps
        end_time = end_time / fps

    # 지정한 시간 구간을 잘라내기
    subclip = clip.subclip(start_time, end_time)

    # 프레임 추출 및 이미지로 저장
    frame_count = 0
    for frame in subclip.iter_frames(fps=fps):  # 10은 프레임 추출 속도를 나타냅니다. 필요에 따라 조정 가능합니다.
        frame_count += 1
        frame_image = Image.fromarray(frame)  # 이미지로 변환
        frame_image.save(f"frame_{frame_count}.jpg")

    # 저장이 끝나면 클립을 닫습니다
    clip.close()
