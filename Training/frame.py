import os
import cv2
import numpy as np
import pickle
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8m-pose.pt')

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

def draw_bounding_boxes(frames):
    bboxes = []
    for i, frame in enumerate(frames):
        results = model(frame)
        frame_bboxes = []

        for result in results:
            boxes = result.boxes  # 바운딩 박스 정보
            if boxes is not None:
                for box in boxes:
                    # 바운딩 박스 좌표와 클래스 정보 추출
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = box.cls[0]
                    if int(cls) == 0:  # 사람이 검출된 경우 (YOLO의 클래스 0은 'person')
                        frame_bboxes.append((xmin, ymin, xmax, ymax))

        print(f"Frame {i} has {len(frame_bboxes)} bounding boxes.")  # 각 프레임의 바운딩 박스 수 출력
        bboxes.append(frame_bboxes)
    return bboxes

def process_videos_in_directory(directory_path, frame_rate=10):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(f"Processing {video_path}...")

                # 비디오에서 프레임 추출
                frames = extract_frames(video_path)
                print(f"Extracted {len(frames)} frames from {video_path}.")
                if not frames:
                    print(f"No frames extracted from {video_path}. Skipping.")
                    continue

                # 바운딩 박스 검출
                bboxes = draw_bounding_boxes(frames)
                if not bboxes:
                    print(f"No bounding boxes detected in frames from {video_path}. Skipping.")
                    continue

                # 저장할 디렉토리 경로 설정
                output_dir = os.path.join('./output_images', os.path.splitext(file)[0])
                os.makedirs(output_dir, exist_ok=True)

                # 바운딩 박스가 그려진 프레임을 저장
                for i, frame in enumerate(frames):
                    for bbox in bboxes[i]:
                        xmin, ymin, xmax, ymax = bbox
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    output_path = os.path.join(output_dir, f'output_frame_{i}.jpg')
                    cv2.imwrite(output_path, frame)
                    print(f"Saved {output_path}")

                # 프레임과 바운딩 박스를 pickle 파일로 저장
                with open(os.path.join(output_dir, 'frames.pkl'), 'wb') as f:
                    pickle.dump(frames, f)
                with open(os.path.join(output_dir, 'bboxes.pkl'), 'wb') as f:
                    pickle.dump(bboxes, f)

                print(f"Frames and bounding boxes have been saved for {video_path}.")

# 주어진 디렉토리에서 모든 비디오 파일 처리
directory_path = r'/media/leeej/My Passport/시니어 이상행동 영상/Training/data/video'
process_videos_in_directory(directory_path)
