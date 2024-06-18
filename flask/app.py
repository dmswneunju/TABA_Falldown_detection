from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import torch
import os
import numpy as np
import time
from io import BytesIO
from PIL import Image

# import 전처리.py // hssc, r, point 좌표 추출
import r_hssc
# 모델
from model import GRUModel


app = Flask(__name__)

# 에러처리
BAD_REQUEST = 'bad request!', 400

# GPU 또는 CPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
def load_model(model_path):
    input_size = 2 # aspect_ratio와 hssc (velocity) 열 선택
    hidden_size1 = 32
    hidden_size2 = 64
    output_size = 1
    model = GRUModel(input_size, hidden_size1, hidden_size2, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# load_model : 모델을 로드하고 평가 모드로 설정
model = load_model('240612gru_fall_detection_model.pth')

# window_size만큼 텐서 나누기
def create_model_input(R_values, HSSC_values, window_size=20, step=2): # step : 몇 칸씩 움직일지
    inputs = []
    for i in range(0, len(R_values) - window_size + 1, step):
        window_r = R_values[i:i + window_size]
        window_hssc = HSSC_values[i:i + window_size]
        if len(window_r) == window_size:  
            combined = np.column_stack((window_r, window_hssc))
            inputs.append(combined)
    inputs = np.array(inputs)
    return torch.tensor(inputs, dtype=torch.float32)



# sample
@app.route('/')
def hello():
	return "Hello World!"


# 파일 업로드 처리, POST 요청을 받음
@app.route('/predict', methods=['POST'])
def predict():
    # 시작 시간 기록
    total_start_time = time.time() 

    # request.files는 클라이언트가 업로드한 파일을 포함하는 딕셔너리. 'file'이라는 키가 있는지 확인
    if 'files' not in request.files:
        # 만약 'file' 키가 없으면 400 error
        return BAD_REQUEST

    files = request.files.getlist('files')
    if len(files) == 0:
        # 파일 리스트가 비어 있으면 400 error
        return BAD_REQUEST
    
    # 파일 전송 시간 측정 시작
    transfer_start_time = time.time()

    # 모든 파일 메모리에 저장
    file_names = []
    frames = []
    for file in files:
        if file and file.filename != '':
            # 파일명을 안전하게 처리
            filename = secure_filename(file.filename)
            file_names.append(filename)

             # 파일을 메모리에 저장
            img = Image.open(BytesIO(file.read()))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frames.append(frame)

    # 파일 전송 시간 측정 종료
    transfer_end_time = time.time()

    # 전처리 시간 측정 시작
    preprocess_start_time = time.time()

    # =====전처리=====
    # 바운딩 박스와 키포인트 추출
    results = r_hssc.infer_pose(frames, r_hssc.yolo_model)
    
    # HSSC와 R 값을 numpy 배열로 변환
    np_results = r_hssc.results_to_numpy(results)

    # 전처리 시간 측정 종료
    preprocess_end_time = time.time()

    # 모델 예측 시간 측정 시작
    model_start_time = time.time()

    # =====모델=====
    # GRU 모델 입력 데이터 생성
    features = np_results[:, [5, 6]].astype(np.float32)  # aspect_ratio와 hssc (velocity) 열 선택
    r_value = features[:, 0]
    hssc_value = features[:,1]
    inputs = create_model_input(r_value, hssc_value).to(device)  # 배치 차원 추가 및 디바이스 이동
    
    # GRU 모델 예측
    outputs = model(inputs)
    fall_predictions = torch.sigmoid(outputs).detach().squeeze().cpu().numpy() > 0.7
    
    # 모델 예측 시간 측정 종료
    model_end_time = time.time()

    # 예측 결과를 리스트로 변환 -> 낙상 결과를 리스트 형식이 아닌 단일 값으로
    fall_predictions_list = fall_predictions.tolist()

    # 연속으로 true가 두개 있을 경우 true를 반환
    def check_consecutive_true(predictions):
        for i in range(len(predictions) - 1):
            if predictions[i] and predictions[i + 1]:
                return True
        return False
    
    # 연속으로 true가 3개일 경우
    '''
    def check_consecutive_true(predictions):
        for i in range(len(predictions) - 2):  # 마지막 두 요소는 비교할 수 없으므로 -2
            if predictions[i] and predictions[i + 1] and predictions[i + 2]:
                return True
        return False
    '''
    fall_occurred = check_consecutive_true(fall_predictions_list)

    total_end_time = time.time()  # 총 처리 종료 시간 기록

    # 각 단계의 시간 계산
    transfer_time = transfer_end_time - transfer_start_time
    preprocess_time = preprocess_end_time - preprocess_start_time
    model_time = model_end_time - model_start_time
    total_processing_time = total_end_time - total_start_time

    file_id = filename.split('_')[0]  # 첫 번째 언더스코어 앞의 부분을 추출

    # 결과 반환
    return jsonify({
        "id": file_id,  # 첫 번째 파일의 이름을 반환,
        "fall": fall_occurred,
        "processing_time": total_processing_time,  # 총 처리 시간을 JSON 응답에 포함
        "transfer_time": transfer_time,  # 파일 전송 시간을 JSON 응답에 포함
        "preprocess_time": preprocess_time,  # 전처리 시간을 JSON 응답에 포함
        "model_time": model_time  # 모델 예측 시간을 JSON 응답에 포함
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=60001, debug=True)
