import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        
        # GRU 레이어
        self.gru1 = nn.GRU(input_size, hidden_size1, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, num_layers=1, batch_first=True)
        
        # Dropout 레이어
        self.dropout = nn.Dropout(p=0.5)
        
        # Fully connected 레이어
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        h0_1 = torch.zeros(1, x.size(0), self.hidden_size1).to(x.device)
        out1, _ = self.gru1(x, h0_1)
        
        h0_2 = torch.zeros(1, out1.size(0), self.hidden_size2).to(out1.device)
        out2, _ = self.gru2(out1, h0_2)
        
        out = self.dropout(out2)
        out = self.fc(out[:, -1, :])
        return out

# 데이터 준비
def load_data(csv_path, sequence_length=5):
    df = pd.read_csv(csv_path)
    
    # aspect_ratio와 HSSC_value 값 추출
    X = df[['aspect_ratio', 'HSSC_value']].values
    y = df['Label'].values
    
    # 데이터 표준화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 시퀀스 데이터로 변환
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    return X_seq, y_seq

# 학습 및 평가
def train_model(X, y, model, criterion, optimizer, num_epochs=100):
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 모델 평가
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        print(f'Validation Loss: {val_loss.item():.4f}')
    
    return model

def main():
    csv_path = '/media/leeej/My Passport/시니어 이상행동 영상/Training/final_dataframe.csv'
    sequence_length = 5  # 시퀀스 길이 설정
    
    X, y = load_data(csv_path, sequence_length)
    
    input_size = 2  # aspect_ratio와 HSSC 값
    hidden_size1 = 32
    hidden_size2 = 64
    output_size = 1  # 낙상 예측 (0 또는 1)
    num_epochs = 100
    learning_rate = 0.001
    
    model = GRUModel(input_size, hidden_size1, hidden_size2, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    trained_model = train_model(X, y, model, criterion, optimizer, num_epochs)
    
    # 모델 저장
    model_path = 'gru_fall_detection_model.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    main()
