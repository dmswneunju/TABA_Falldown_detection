import os
import cv2
import pandas as pd
import json
import torch

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 사용 가능 시 GPU로 이동

# NMS 시간 제한 늘리기
model.iou = 0.5  # IOU threshold for NMS
model.conf = 0.4  # Confidence threshold for NMS
model.nms = {'time_limit': 2.0}  # Time limit for NMS

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

# MP4 파일과 JSON 파일 매칭
def match_files(mp4_files, json_files):
    matched_files = []

    for mp4_file in mp4_files:
        mp4_base_name = os.path.basename(mp4_file).replace('.mp4', '')
        for json_file in json_files:
            json_base_name = os.path.basename(json_file).replace('.json', '')
            if mp4_base_name == json_base_name:
                matched_files.append((mp4_file, json_file))
                break

    return matched_files

# 프레임 추출 함수
def extract_frames(video_path, frame_rate=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (30 // frame_rate) == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

# 바운딩 박스 검출 함수
def draw_bounding_boxes(frames, model):
    bboxes = []
    for frame in frames:
        results = model(frame)
        frame_bboxes = []

        for bbox in results.xyxy[0].cpu().numpy():
            xmin, ymin, xmax, ymax, conf, cls = bbox
            if int(cls) == 0:  # 사람이 검출된 경우
                frame_bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        bboxes.append(frame_bboxes)
    return bboxes



# 비디오와 JSON 파일 매칭 경로 설정
base_path = "/media/leeej/My Passport/시니어 이상행동 영상/Training/data/video"

# MP4 파일과 JSON 파일 찾기
mp4_files, json_files = find_files(base_path)
matched_files = match_files(mp4_files, json_files)


all_data = []

for video_id, (video_path, json_path) in enumerate(matched_files, start=1):
    # JSON 파일 로드
    with open(json_path, 'r') as f:
        annotations = json.load(f)['annotations']

    # 프레임 추출
    frames = extract_frames(video_path)
    # 바운딩 박스 검출
    bboxes = draw_bounding_boxes(frames, model)

    if not frames or not bboxes:
        continue  # 비디오에서 프레임이나 바운딩 박스를 추출하지 못한 경우 다음 비디오로 넘어감

    data = []

    for frame_idx, frame_bboxes in enumerate(bboxes):
        actual_frame_idx = frame_idx * 3  # 10fps에서의 인덱스를 30fps 기준으로 변환
        if frame_bboxes:
            bbox = frame_bboxes[0]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_ratio = bbox_width / bbox_height if bbox_height != 0 else 0
        else:
            bbox_ratio = 0

        label = 0
        for obj in annotations['object']:
            if obj['startFrame'] <= actual_frame_idx <= obj['endFrame']:
                label = 1
                break

        data.append([video_id, frame_idx, bbox_ratio, label])

    if data:
        video_df = pd.DataFrame(data, columns=['VideoID', 'FrameNumber', 'BBoxRatio', 'Label'])
        all_data.append(video_df)

# 모든 데이터프레임을 하나로 합치기
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
else:
    final_df = pd.DataFrame(columns=['VideoID', 'FrameNumber', 'BBoxRatio', 'Label'])

# 데이터프레임 출력
print(final_df)

