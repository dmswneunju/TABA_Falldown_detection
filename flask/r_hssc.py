from ultralytics import YOLO
import numpy as np
import math

# YOLOv8 모델 로드
yolo_model = YOLO('yolov8m-pose.pt')

# bbox & 키포인트 추출 함수
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

# HSSC 특징·키포인트 추출 알고리즘
def calculate_center(points):
    center_x = 0.0
    center_y = 0.0
    area = 0.0

    # 유효한 좌표만 필터링
    valid_points = [point for point in points if point != (0.0, 0.0)]
    
    n = len(valid_points)
    if n == 0:
        return (0.0, 0.0)  # 유효한 좌표가 없는 경우 (0.0, 0.0) 반환
    
    for i in range(n):
        next_i = (i + 1) % n
        first_point = valid_points[i]
        second_point = valid_points[next_i]

        factor = (first_point[0] * second_point[1]) - (second_point[0] * first_point[1])
        area += factor
        center_x += (first_point[0] + second_point[0]) * factor
        center_y += (first_point[1] + second_point[1]) * factor

    try:
        area /= 2.0
        centroidFactor = 1.0 / (6.0 * area)
        center_x *= centroidFactor
        center_y *= centroidFactor
    except ZeroDivisionError:
        print("ZeroDivision")

    return (round(center_x, 5), round(center_y, 5))

# 두 프레임 사이의 거리 계산
def calculate_distances(centers):
    distances = [0.0]
    for i in range(1, len(centers)):
        x_i, y_i = centers[i]
        x_prev, y_prev = centers[i - 1]
        distance = math.sqrt((x_i - x_prev)**2 + (y_i - y_prev)**2)
        distances.append(round(distance, 5))
    return distances

# 두 프레임 사이의 속도 계산
def calculate_velocities(distances, frame_time):
    velocities = [0.0]  # 첫 번째 프레임의 속도는 0으로 설정
    for distance in distances[1:]:
        velocity = distance / frame_time
        velocities.append(round(velocity, 5))
    return velocities

# 결과 값을 numpy 배열로 저장
def results_to_numpy(results):
    data = []
    points_frames = []
    for result in results:
        row = [result['frame_idx']]
        bbox = result['bbox']
        row.extend([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])  # x, y, w, h

        # 너비와 높이의 비율을 추가
        aspect_ratio = (bbox[2]-bbox[0]) / (bbox[3]-bbox[1]) if (bbox[3]-bbox[1]) != 0 else 0
        row.append(aspect_ratio)
        
        keypoints = result['keypoints']
        frame_points = []
        for i, (x, y) in enumerate(keypoints[:7]):  # 포인트 1~7만 사용
            frame_points.append((x, y))
        
        points_frames.append(frame_points)
        data.append(row)
    
    np_data = np.array(data, dtype=np.float32)  # 데이터를 float32 타입으로 변환
    
    # 각 프레임의 중심 좌표 계산
    centers = [calculate_center(points) for points in points_frames]

    # 두 프레임 사이의 거리 계산
    distances = calculate_distances(centers)

    # 각 프레임 사이의 시간 간격 (초 단위)
    fps = 10 # 예: 10fps
    frame_time = 1.0 / fps  # 한 프레임의 시간 간격이 0.1초라는 뜻

    # 속도 계산
    velocities = calculate_velocities(distances, frame_time)
    
    # 속도를 numpy 배열로 추가
    velocities = np.array(velocities).reshape(-1, 1)
    np_data = np.hstack((np_data, velocities))
    
    '''
    frame_idx - 프레임 인덱스
    x - 바운딩 박스의 x 좌표
    y - 바운딩 박스의 y 좌표
    w - 바운딩 박스의 너비
    h - 바운딩 박스의 높이
    aspect_ratio - 바운딩 박스의 너비와 높이의 비율
    velocity - 두 프레임 사이의 속도 (HSSC 값)
    '''
    return np_data