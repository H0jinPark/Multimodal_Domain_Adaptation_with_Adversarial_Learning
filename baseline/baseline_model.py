import torch
import torch.nn as nn

class Baseline1DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Baseline1DCNN, self).__init__()
        
        # 입력 형태: [Batch, Channel(5), Length(2000)]
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Global Average Pooling: 길이에 상관없이 채널별 평균을 뽑아내어 과적합을 방지합니다.
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # 정규화
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten (Batch, 128)
        x = self.classifier(x)
        return x

# 모델 형태 확인용 테스트 코드
if __name__ == '__main__':
    model = Baseline1DCNN()
    dummy_input = torch.randn(32, 5, 2000)
    output = model(dummy_input)
    print(f"모델 출력 형태: {output.shape}") # 예상: torch.Size([32, 10])