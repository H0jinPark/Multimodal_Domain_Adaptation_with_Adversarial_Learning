import torch
import torch.nn as nn
from torch.autograd import Function

# ==========================================
# 💡 1. 마법의 GRL (Gradient Reversal Layer)
# ==========================================
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 순전파(Forward) 때는 아무것도 건드리지 않고 그대로 통과!
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 역전파(Backward) 때는 기울기에 -alpha를 곱해서 반대로 전달! (Adversarial)
        output = grad_output.neg() * ctx.alpha
        return output, None

# ==========================================
# 💡 2. DANN 1D CNN 아키텍처
# ==========================================
class DANN1DCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=5):
        super(DANN1DCNN, self).__init__()
        
        # 🟢 [Part 1] 특징 추출기 (Feature Extractor)
        # 센서 데이터(N, 5, 2000)에서 유의미한 패턴을 뽑아냅니다.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # 윈도우 길이가 2000이든 1000이든 무조건 128차원의 벡터로 압축해 주는 마법의 풀링
            nn.AdaptiveAvgPool1d(1) 
        )
        
        # 🔵 [Part 2] 운동 분류기 (Task Label Predictor)
        # 특징 벡터(128차원)를 받아 10개의 운동 클래스 중 하나를 맞힙니다.
        self.label_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # 🔴 [Part 3] 도메인 분류기 (Domain Classifier)
        # 특징 벡터를 받아 이것이 Source(0)인지 Target(1)인지 맞힙니다.
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2) # Binary Classification (Source vs Target)
        )

    def forward(self, x, alpha=1.0):
        # 1. 공통 특징 추출 (N, 5, 2000) -> (N, 128)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # 평탄화 (Flatten)
        
        # 2. 운동 예측
        class_preds = self.label_predictor(features)
        
        # 3. 도메인 예측 (GRL 통과)
        # 여기서 alpha 값만큼 기울기가 뒤집혀서 feature_extractor로 흘러 들어갑니다.
        reverse_features = GradientReversalLayer.apply(features, alpha)
        domain_preds = self.domain_classifier(reverse_features)
        
        return class_preds, domain_preds