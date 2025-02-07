
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
import torch

def select_channel(signal, channel_idx=0):

    return signal[channel_idx]


dataset = SEEDDataset(
    root_path='/content/drive/MyDrive/prometheus/SEED_Preprocessed_EEG',
    online_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: select_channel(x, channel_idx=30)),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ]),
    label_transform=transforms.Compose([
        transforms.Select('emotion'),
        transforms.Lambda(lambda x: x + 1)
    ])
)

from torch.utils.data import random_split

# 전체 데이터셋 크기
total_size = len(dataset)

# 훈련 세트 비율 (예: 80%)
train_ratio = 0.8
train_size = int(total_size * train_ratio)
test_size = total_size - train_size

# 랜덤 분할
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(len(train_dataset), len(test_dataset))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim

### loss : CrossEntropyLoss    optimizer : AdamW    learning rate : 0.0005

class ResNetBiLSTMAttentionEEG(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetBiLSTMAttentionEEG, self).__init__()

        # pretrained resnet34
        self.resnet = models.resnet34(pretrained=True)

        # first layer as eeg shape
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)

        # flatten
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Remove fc
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # BiLSTM Layer, use dropout
        self.bilstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)

        # Attention Layer
        self.attention = nn.Linear(256, 1)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # (batch_size, 1, 1, 200)
        x = x.unsqueeze(2)

        # (batch_size, 512, 1, 6)
        x = self.resnet(x)

        # (batch_size, 512, 6)
        x = x.squeeze(-2)

        # (batch_size, 6, 512)
        x = x.permute(0, 2, 1)

        # (batch_size, 6, 256)
        lstm_out, _ = self.bilstm(x)

        attention_weights = torch.tanh(self.attention(lstm_out))  # (batch_size, 6, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, 6, 1)
        weighted_lstm_out = lstm_out * attention_weights  # (batch_size, 6, 256)
        context_vector = torch.sum(weighted_lstm_out, dim=1)  # (batch_size, 256)

        # Fully connected layer
        output = self.fc(context_vector)
        return output

model = ResNetBiLSTMAttentionEEG()
import torch.optim as optim
model = model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters())

for epoch in range(10):
    print(f"\nEpoch {epoch}")

    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        X = X.to('cuda')
        y = y.to('cuda')

        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += y.size(0)
        correct_train += (predicted == y).sum().item()

    train_loss = running_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train


    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.cuda()
            y = y.cuda()

            outputs = model(X)
            loss = criterion(outputs, y)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += y.size(0)
            correct_val += (predicted == y).sum().item()

    val_loss = running_val_loss / len(test_loader)
    val_accuracy = correct_val / total_val

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")