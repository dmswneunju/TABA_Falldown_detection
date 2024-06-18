import os
import pandas as pd
import math
import numpy as np

# pandas 옵션 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# CSV 파일을 읽어 데이터프레임으로 변환
csv_path = '/media/leeej/My Passport/시니어 이상행동 영상/Training/final_dataframe.csv'
df = pd.read_csv(csv_path)


def dataframe_to_points_frames(df):
    points_frames = []
    frame_indices = df['frame_idx'].unique()  # 프레임 인덱스 추출

    for frame in frame_indices:
        frame_data = df[df['frame_idx'] == frame]
        for index, row in frame_data.iterrows():
            # (point{i}x, point{i}y) 형식으로
            frame_points = [(round(row[f'point{i}x'], 5), round(row[f'point{i}y'], 5)) for i in range(1, 8)]  # point 1~7, 5자리까지
            points_frames.append(frame_points)

    return points_frames

# 데이터프레임을 points_frames로 변환
points_frames = dataframe_to_points_frames(df)


# HSSC 특징·키포인트 추출 알고리즘
# 각 프레임에서의 머리와 어깨의 중심 좌표(x, y)
def calculate_center(points):
    center_x = 0.0
    center_y = 0.0
    area = 0.0

    # 유효한 좌표만 필터링
    valid_points = [point for point in points if point != (0.0, 0.0)]
    
    n = len(valid_points)
    if n == 0:
        return (0.0, 0.0)  # 유효한 좌표가 없는 경우 (0.0, 0.0) 반환
    
    # points : 총 데이터 프레임 열 개수
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

# 각 프레임의 중심 좌표 계산
centers = [calculate_center(points) for points in points_frames]

# 두 프레임 사이의 거리 계산
distances = calculate_distances(centers)

# 각 프레임 사이의 시간 간격 (초 단위)
fps = 10 # 예: 10fps
frame_time = 1.0 / fps  # 한 프레임의 시간 간격이 0.1초라는 뜻

# 속도 계산
velocity = calculate_velocities(distances, frame_time)


# velocity 값을 hssc 컬럼으로 추가
df['hssc'] = pd.Series(velocity)

# hssc 컬럼을 point17x, point17y 뒤에 추가하고 Label 앞에 위치시키기
columns = list(df.columns)
columns.remove('hssc')
columns.insert(columns.index('Label'), 'hssc')
df = df[columns]

# 수정된 데이터프레임을 다시 CSV 파일로 저장
df.to_csv(csv_path, index=False)