import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, lstm_hidden_dim, attention_dim, num_classes):
        super(CNNBiLSTMAttention, self).__init__()

        # CNN 레이어 (Pooling 제거)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # BiLSTM 레이어
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)

        # Attention Mechanism 개선
        self.attention = nn.Linear(2 * lstm_hidden_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

        # Fully Connected Layers
        self.fc1 = nn.Linear(2 * lstm_hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN
        x = self.cnn(x)  # [batch, 1, num_samples] -> [batch, 128, num_samples]
        x = x.permute(0, 2, 1)  # [batch, channels, time] -> [batch, time, channels]

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # [batch, time, 2*lstm_hidden_dim]

        # Attention Mechanism
        attention_scores = torch.tanh(self.attention(lstm_out))  # [batch, time, attention_dim]
        attention_weights = F.softmax(self.context_vector(attention_scores), dim=1)  # [batch, time, 1]
        weighted_sum = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, 2*lstm_hidden_dim]

        # Fully Connected Layers
        fc1_out = torch.relu(self.fc1(weighted_sum))
        output = self.fc2(fc1_out)

        # Softmax 추가하여 확률 출력
        output = F.softmax(output, dim=1)

        return output