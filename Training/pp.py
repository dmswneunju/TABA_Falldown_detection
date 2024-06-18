from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# YOLOv8 포즈 모델 로드 (모델 파일 경로를 정확히 지정)
model = YOLO('yolov8m-pose.pt')  # 모델 파일 이름을 변경

# 입력 이미지의 루트 디렉토리 설정
root_input_dir = '/media/leeej/My Passport/시니어 이상행동 영상/Training/output_images'

# 입력 디렉토리의 하위 디렉토리와 이미지를 순회
for root, dirs, files in os.walk(root_input_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            
            if image is None:
                
                continue
            
            # 포즈 추정 수행
            results = model(image)
            
            # 결과 이미지에 키포인트 표시
            annotated_frame = results[0].plot()
            
            # BGR 이미지를 RGB로 변환
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # 결과 이미지 저장 경로 설정
            output_subdir = os.path.join(root, 'point_image')
            os.makedirs(output_subdir, exist_ok=True)
            
            output_image_path = os.path.join(output_subdir, file)
            cv2.imwrite(output_image_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
            # 이미지 출력 (필요시 주석 처리)
            plt.imshow(annotated_frame)
            plt.axis('off')
            plt.show()