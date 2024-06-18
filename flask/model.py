import torch
import torch.nn as nn
import torch.optim as optim

# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        # GRU 레이어
        self.gru1 = nn.GRU(input_size, hidden_size1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)

        # Dropout 레이어
        self.dropout = nn.Dropout(p=0.3)

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