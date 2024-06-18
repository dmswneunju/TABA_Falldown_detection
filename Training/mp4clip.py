import json
from moviepy.editor import VideoFileClip
import os

# video 폴더에 똑같은 폴더의 이름 만들어두기
'''path = "./동영상/Abnormal_Behavior_Falldown"
file_list = os.listdir(path)
newpath = "./data/video"

for folder in file_list:
    if '[원천]' in folder:
        new_folder_path = os.path.join(newpath, folder)
        try:
            os.makedirs(new_folder_path, exist_ok=True)  # 폴더 생성
            print(f"폴더 생성: {new_folder_path}")
        except OSError as e:
            print(f"폴더 생성 실패: {new_folder_path}, 에러: {e}")'''
            

def extract_event_clips(input_path, output_path, json_path, new_json_path, padding_factor=2):
    # JSON 파일을 불러옵니다
    with open(json_path, 'r') as f:
        data = json.load(f)

    # JSON 파일에서 'annotations' 키를 가져옵니다. 
    # 'annotations' 키가 없으면 빈 딕셔너리를 반환합니다.
    annotations = data.get('annotations', {})
    fps = annotations.get('fps', 29.97)
    objects = annotations.get('object', [])

    first_object = objects[0]
    start_frame = first_object.get('startFrame', 0)
    end_frame = first_object.get('endFrame', 0)
    
    # 시작 시간과 종료 시간을 계산합니다
    start_time = start_frame / fps
    end_time = end_frame / fps
    event_duration = end_time - start_time
    
    # 비디오를 불러옵니다
    video = VideoFileClip(input_path)
    
    # 이벤트 시작 전과 종료 후 길이를 계산합니다
    pre_event_start = max(0, start_time - event_duration * padding_factor)
    post_event_end = min(video.duration, end_time + event_duration)
    
    # 필요한 부분의 클립을 만듭니다
    event_clip = video.subclip(pre_event_start, post_event_end)
    
    # 새로운 비디오 파일로 저장합니다
    event_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    # 새로운 JSON 파일을 만듭니다
    new_start_frame = int((start_time - pre_event_start) * fps)
    new_end_frame = int((end_time - pre_event_start) * fps)

    new_data = {
        "annotations": {
            "duration": f"{int(event_clip.duration // 60):02}:{int(event_clip.duration % 60):02}",
            "resourceId": annotations.get('resourceId'),
            "resource": os.path.basename(output_path),
            "resourcePath": os.path.dirname(output_path),
            "fps": fps,
            "totFrame": int(event_clip.duration * fps),
            "resourceSize": os.path.getsize(output_path),
            "object": [
                {
                    "startPosition": first_object.get('startPosition'),
                    "endPosition": first_object.get('endPosition'),
                    "startFrame": new_start_frame,
                    "endFrame": new_end_frame,
                    "actionType": first_object.get('actionType'),
                    "actionName": first_object.get('actionName')
                }
            ]
        }
    }

    # 새로운 JSON 파일을 저장합니다
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)

# MP4 파일과 JSON 파일 찾기
def find_files(base_path):
    mp4_files = []
    json_files = []
    
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if '[원천]' in dir_name:
                full_dir_path = os.path.join(root, dir_name)
                for sub_dir_name in os.listdir(full_dir_path):
                    sub_dir_path = os.path.join(full_dir_path, sub_dir_name)
                    if os.path.isdir(sub_dir_path):
                        for file_name in os.listdir(sub_dir_path):
                            if file_name.endswith('.mp4'):
                                mp4_files.append(os.path.join(sub_dir_path, file_name))
            elif '[라벨]' in dir_name:
                full_dir_path = os.path.join(root, dir_name)
                for file_name in os.listdir(full_dir_path):
                    if file_name.endswith('.json'):
                        json_files.append(os.path.join(full_dir_path, file_name))
    
    return mp4_files, json_files

def match_files(mp4_files, json_files):
    matched_files = []

    for mp4_file in mp4_files:
        # os.path.basename(mp4_file): mp4_file 경로에서 파일명 부분만 추출합니다.
        # -> FD_In_H11H21H31_0001_20210112_09.mp4
        mp4_base_name = os.path.basename(mp4_file).replace('.mp4', '')
        for json_file in json_files:
            json_base_name = os.path.basename(json_file).replace('.json', '')
            if mp4_base_name == json_base_name:
                matched_files.append((mp4_file, json_file))
                break

    return matched_files


video = "./동영상/Abnormal_Behavior_Falldown"
output_base_path = "./data/video"

# MP4 파일과 JSON 파일 찾기
mp4_files, json_files = find_files(video)
# mp4_files : ./동영상/Abnormal_Behavior_Falldown/[원천]inside_H11H21H31/H11H21H31/FD_In_H11H21H31_0001_20210112_09.mp4
matched_files = match_files(mp4_files, json_files)
# matched_files : 
# ('./동영상/Abnormal_Behavior_Falldown/[원천]inside_H12H22H33/H12H22H33/FD_In_H12H22H33_0008_20201016_20.mp4', 
#'./동영상/Abnormal_Behavior_Falldown/[라벨]inside_H12H22H33/FD_In_H12H22H33_0008_20201016_20.json')

for mp4_file, json_file in matched_files:
    # os.path.relpath(경로, 기준경로): 주어진 경로를 기준경로로부터의 상대 경로로 변환합니다.
    # os.path.relpath('./동영상/Abnormal_Behavior_Falldown/[원천]inside_H11H21H31/H11H21H31', './동영상/Abnormal_Behavior_Falldown')는 
    # [원천]inside_H11H21H31/H11H21H31로 변환됩니다. : mp4_relative_path
    mp4_relative_path = os.path.relpath(mp4_file, video)
    output_file_path = os.path.join(output_base_path, mp4_relative_path)
    # output_dir : ./data/video/[원천]inside_H12H22H33/H12H22H33
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    json_relative_path = os.path.relpath(json_file, video)
    new_json_file_path = os.path.join(output_base_path, json_relative_path)
    new_json_dir = os.path.dirname(new_json_file_path)
    os.makedirs(new_json_dir, exist_ok=True)


    extract_event_clips(mp4_file, output_file_path, json_file,new_json_file_path)
    

# 함수 호출 예시
'''extract_event_clips("./동영상/Abnormal_Behavior_Falldown/[원천]inside_H11H21H32/H11H21H32/FD_In_H11H21H32_0001_20210110_09.mp4", 
                    "./data/video/[원천]inside_H11H21H31/H11H21H31/FD_In_H11H21H31_0001_20210112_09.mp4", 
                    "./동영상/Abnormal_Behavior_Falldown/[라벨]inside_H11H21H32/FD_In_H11H21H32_0001_20210110_09.json4")'''
