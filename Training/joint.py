import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pickle

# MediaPipe 포즈 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def crop_and_estimate_pose(frames, bboxes, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pose_landmarks_list = []
    landmarks_columns = ['frame', 'person', 'landmark', 'x', 'y']

    all_landmarks = []

    for i, frame in enumerate(frames):
        frame_bboxes = bboxes[i]
        frame_landmarks = []

        for j, bbox in enumerate(frame_bboxes):
            xmin, ymin, xmax, ymax = bbox
            person_crop = frame[ymin:ymax, xmin:xmax]

            # MediaPipe로 포즈 추정
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_crop_rgb)

            if results_pose.pose_landmarks:
                landmarks = []
                head_coords = []
                # 포즈 랜드마크 추출
                for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                    cx = int(landmark.x * person_crop.shape[1])
                    cy = int(landmark.y * person_crop.shape[0])
                    if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # 머리 부분 랜드마크
                        head_coords.append((cx, cy))
                    else:
                        landmarks.append((cx, cy))
                        all_landmarks.append([i, j, idx, xmin + cx, ymin + cy])

                if head_coords:
                    # 머리 좌표의 평균을 계산하여 하나의 좌표로 사용
                    head_x = sum([coord[0] for coord in head_coords]) // len(head_coords)
                    head_y = sum([coord[1] for coord in head_coords]) // len(head_coords)
                    landmarks.append((head_x, head_y))
                    all_landmarks.append([i, j, 'head', xmin + head_x, ymin + head_y])

                frame_landmarks.append(landmarks)

        pose_landmarks_list.append(frame_landmarks)

    # 데이터 프레임 생성
    landmarks_df = pd.DataFrame(all_landmarks, columns=landmarks_columns)
    return pose_landmarks_list, landmarks_df

# 주어진 디렉토리에서 모든 프레임과 바운딩 박스 파일 처리
base_directory_path = r'/media/leeej/My Passport/시니어 이상행동 영상/Training/output_images3'
for root, _, files in os.walk(base_directory_path):
    if 'frames.pkl' in files and 'bboxes.pkl' in files:
        frames_path = os.path.join(root, 'frames.pkl')
        bboxes_path = os.path.join(root, 'bboxes.pkl')
        output_dir = os.path.join(root, 'pose_output')
        
        # 프레임과 바운딩 박스 로드
        with open(frames_path, 'rb') as f:
            frames = pickle.load(f)
        with open(bboxes_path, 'rb') as f:
            bboxes = pickle.load(f)

        # 포즈 추정 및 결과 저장
        pose_landmarks, landmarks_df = crop_and_estimate_pose(frames, bboxes, output_dir)

        # 랜드마크 데이터를 CSV 파일로 저장
        landmarks_df.to_csv(os.path.join(output_dir, 'landmarks.csv'), index=False)
        print(f"Landmarks saved to {os.path.join(output_dir, 'landmarks.csv')}")
