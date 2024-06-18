import os
import json
import cv2
import pandas as pd
import torch
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8m-pose.pt')

#프레임 추출
def extract_frames(video_path, frame_rate=10): #10fps
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        return frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (30 // frame_rate) == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

#bbox&포인트 추출 
def infer_pose(frames, model):
    results = []
    for frame_idx, frame in enumerate(frames):
        pose_results = model(frame)
        for result in pose_results:
            boxes = result.boxes  # 바운딩 박스
            keypoints = result.keypoints  # 키포인트

            for i in range(len(boxes)):
                if boxes.cls[i] == 0:  # '0'은 사람 클래스
                    bbox = boxes.xyxy[i].tolist()  # 바운딩 박스 좌표
                    keypoint = keypoints.xyn[i].tolist()  # 정규화 키포인트 좌표

                    results.append({
                        "frame_idx": frame_idx,
                        "bbox": bbox,
                        "keypoints": keypoint
                    })
    return results


#결과 값 데이터프레임으로 저장
def results_to_dataframe(results, annotations, video_id):
    data = []
    for result in results:
        row = {}
        row['VideoID'] = video_id
        row['frame_idx'] = result['frame_idx']
        bbox = result['bbox']
        row['x'], row['y'], row['w'], row['h'] = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        
        # 너비와 높이의 비율을 추가
        row['aspect_ratio'] = row['w'] / row['h'] if row['h'] != 0 else 0
        
        keypoints = result['keypoints']
        for i, (x, y) in enumerate(keypoints):
            row[f'point{i+1}x'] = x
            row[f'point{i+1}y'] = y
        
        actual_frame_idx = result['frame_idx'] * 3  # 10fps에서의 인덱스를 30fps 기준으로 변환
        label = 0
        for obj in annotations['object']:
            if obj['startFrame'] <= actual_frame_idx <= obj['endFrame']:
                label = 1
                break
        row['Label'] = label
        data.append(row)
    df = pd.DataFrame(data)
    return df

#video_json 쌍 찾기
def find_video_json_pairs(root_dir):
    video_json_pairs = []
    for root, dirs, files in os.walk(root_dir):
        video_files = [f for f in files if f.endswith('.mp4')]
        json_files = [f for f in files if f.endswith('.json')]

        for video_file in video_files:
            json_file = video_file.replace('.mp4', '.json')
            json_file_path = None
            for dirpath, _, filenames in os.walk(root_dir):
                if json_file in filenames:
                    json_file_path = os.path.join(dirpath, json_file)
                    break

            if json_file_path:
                video_path = os.path.join(root, video_file)
                video_json_pairs.append((video_path, json_file_path))

    print(f"Found {len(video_json_pairs)} video-JSON pairs")
    return video_json_pairs

#전체 데이터에 대해 처리
def process_video_and_json(video_path, json_path, model, video_id):
    frames = extract_frames(video_path)
    pose_results = infer_pose(frames, model)

    with open(json_path, 'r') as f:
        annotations = json.load(f)['annotations']

    df = results_to_dataframe(pose_results, annotations, video_id)
    return df



# 루트 디렉토리 설정
root_dir = '/media/leeej/My Passport/시니어 이상행동 영상/Training/data/video'

# 비디오-JSON 쌍 찾기
video_json_pairs = find_video_json_pairs(root_dir)

# 모든 비디오와 JSON 파일 처리
all_dataframes = []
for video_path, json_path in video_json_pairs:
    video_id = os.path.basename(video_path).split('.')[0]
    df = process_video_and_json(video_path, json_path, model, video_id)
    all_dataframes.append(df)

# 모든 결과를 하나의 DataFrame으로 병합
final_df = pd.concat(all_dataframes, ignore_index=True)

# 최종 DataFrame을 CSV 파일로 저장
output_csv_path = '/media/leeej/My Passport/시니어 이상행동 영상/Training/final_dataframe.csv'
output_dir = os.path.dirname(output_csv_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

final_df.to_csv(output_csv_path, index=False)

print(f"DataFrame saved to {output_csv_path}")



# 최종 DataFrame 출력
print(final_df)




