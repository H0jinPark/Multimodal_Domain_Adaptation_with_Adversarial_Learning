import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class SensorDataset(Dataset):
    def __init__(self, X, y):
        # [N, 2000, 5] -> [N, 5, 2000]
        X_transposed = np.transpose(X, (0, 2, 1))
        self.X = torch.FloatTensor(X_transposed)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir='./preprocessed_data', batch_size=64):
    print(f">>> '{data_dir}'에서 전처리된 데이터 로드 중...")
    
    # 1. 파일 불러오기
    X_src = np.load(os.path.join(data_dir, 'X_source.npy'))
    y_src = np.load(os.path.join(data_dir, 'y_source.npy'), allow_pickle=True)
    X_tgt = np.load(os.path.join(data_dir, 'X_target.npy'))
    y_tgt = np.load(os.path.join(data_dir, 'y_target.npy'), allow_pickle=True)

    # ==========================================
    # 💡 [긴급 처방] 데이터 안의 nan 값을 0.0으로 덮어씌우기
    # ==========================================
    X_src = np.nan_to_num(X_src, nan=0.0)
    X_tgt = np.nan_to_num(X_tgt, nan=0.0)

    # ==========================================
    # 💡 [STEP 1] Source 데이터를 Train / Validation으로 분할
    # ==========================================
    X_train, X_val, y_train, y_val = train_test_split(
        X_src, y_src, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_src
    )
    print(f">>> Data Split 완료: Train({len(X_train)}개) / Val({len(X_val)}개) / Target({len(X_tgt)}개)")

    # ==========================================
    # 💡 [STEP 2 수정] 도메인 독립적 스케일링 (Domain-wise Normalization)
    # ==========================================
    N_train, L, C = X_train.shape
    N_val, _, _ = X_val.shape
    N_tgt, _, _ = X_tgt.shape
    
    # 2차원으로 펼치기
    X_train_flat = X_train.reshape(-1, C)
    X_val_flat = X_val.reshape(-1, C)
    X_tgt_flat = X_tgt.reshape(-1, C)
    
    # 1. Source 도메인 스케일러 (Train 기준으로 Val까지 변환)
    scaler_src = StandardScaler()
    X_train_scaled_flat = scaler_src.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler_src.transform(X_val_flat)
    
    # 2. Target 도메인 스케일러 (독립적으로 자기 자신의 평균/분산 사용)
    scaler_tgt = StandardScaler()
    X_tgt_scaled_flat = scaler_tgt.fit_transform(X_tgt_flat)
    
    # 원래 3차원 형태로 복구
    X_train = X_train_scaled_flat.reshape(N_train, L, C)
    X_val = X_val_scaled_flat.reshape(N_val, L, C)
    X_tgt = X_tgt_scaled_flat.reshape(N_tgt, L, C)
    
    print(">>> 도메인 독립적 StandardScaler 적용 완료 (Source/Target 각자 스케일링)")

    # ==========================================
    # 💡 [STEP 3] 라벨 인코딩 (일관성 유지)
    # ==========================================
    fixed_classes = [
        'barbellcurl', 'barbellrow', 'benchpress', 'bte', 'deadlift', 
        'dips', 'latpulldown', 'ohp', 'pullup', 'pushup'
    ]
    le = LabelEncoder()
    le.classes_ = np.array(fixed_classes)
    
    y_train_encoded = le.transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_tgt_encoded = le.transform(y_tgt)

    # ==========================================
    # 💡 [STEP 4] Dataset 및 DataLoader 생성
    # ==========================================
    train_dataset = SensorDataset(X_train, y_train_encoded)
    val_dataset = SensorDataset(X_val, y_val_encoded)
    tgt_dataset = SensorDataset(X_tgt, y_tgt_encoded)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False)

    print(">>> 3개의 DataLoader (Train, Val, Target) 생성 완료!")
    return train_loader, val_loader, tgt_loader, le

if __name__ == '__main__':
    # 🚀 테스트 코드
    train_loader, val_loader, tgt_loader, label_encoder = get_dataloaders(batch_size=32)
    
    for X_batch, y_batch in train_loader:
        print(f"\n[Train Batch 확인]")
        print(f"X batch shape: {X_batch.shape}")
        
        # Train 스케일링 확인
        print(f"채널별 평균: {X_batch.mean(dim=[0, 2]).round(decimals=4)}") 
        print(f"채널별 표준편차: {X_batch.std(dim=[0, 2]).round(decimals=4)}")
        break