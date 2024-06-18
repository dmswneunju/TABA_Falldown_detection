import os
import pickle
import json

def save_labels_from_pickles(frames_pickle_path, bboxes_pickle_path, output_dir):
    # 프레임과 바운딩 박스를 pickle 파일에서 로드
    with open(frames_pickle_path, 'rb') as f:
        frames = pickle.load(f)
    with open(bboxes_pickle_path, 'rb') as f:
        bboxes = pickle.load(f)
    
    # 바운딩 박스를 JSON 파일로 저장
    labels = []
    for frame_count, frame_bboxes in bboxes:
        for bbox in frame_bboxes:
            labels.append({
                "frame": frame_count,
                "bbox": bbox
            })

    # JSON 파일로 저장
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(labels, f)
    
    print(f"Frames and bounding boxes have been saved to {os.path.join(output_dir, 'labels.json')}")

# 경로 설정
frames_pickle_path = r'/media/leeej/My Passport/시니어 이상행동 영상/Training/output_images3/FD_In_H11H21H31_0001_20210112_09/frames.pkl'
bboxes_pickle_path = r'/media/leeej/My Passport/시니어 이상행동 영상/Training/output_images3/FD_In_H11H21H31_0001_20210112_09/bboxes.pkl'
output_dir = r'/media/leeej/My Passport/시니어 이상행동 영상/Training/output_images3/FD_In_H11H21H31_0001_20210112_09/pose_output'

# 출력 디렉토리가 존재하지 않으면 생성
os.makedirs(output_dir, exist_ok=True)

# 라벨 저장 함수 호출
save_labels_from_pickles(frames_pickle_path, bboxes_pickle_path, output_dir)
