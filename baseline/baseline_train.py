import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # tqdm 임포트

# 상위 폴더(루트)에 있는 data_loader.py를 가져오기 위한 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader import get_dataloaders
from baseline_model import Baseline1DCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 기기: {device}")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessed_data')
    
    # [수정 포인트] 로더 3개를 받아옵니다. (src_loader -> train_loader)
    train_loader, val_loader, tgt_loader, _ = get_dataloaders(data_dir=data_dir, batch_size=64)
    
    model = Baseline1DCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    
    # [추가된 변수] 최고 검증 정확도를 기록할 변수와 저장 경로
    best_val_acc = 0.0 
    save_path = os.path.join(os.path.dirname(__file__), 'baseline_model.pt')
    
    print("\n🔥 베이스라인 모델 학습 시작 (Train/Val Split 적용)...")
    for epoch in range(epochs):
        
        # ==========================================
        # 1. 학습 (Training Phase)
        # ==========================================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_description(f"Epoch [{epoch+1}/{epochs}] [Train]")
            train_pbar.set_postfix(loss=loss.item(), acc=correct/total)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # ==========================================
        # 2. 검증 (Validation Phase)
        # ==========================================
        model.eval() # 평가 모드 전환 (Dropout, BatchNorm 고정)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 기울기 계산 비활성화 (메모리 절약, 속도 향상)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # 결과 요약 출력
        print(f"✨ Epoch {epoch+1} Summary - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        # ==========================================
        # 3. 최고 성능 모델 저장 (Best Model Save)
        # ==========================================
        if val_epoch_acc > best_val_acc:
            print(f"   🏆 Best Validation Accuracy 갱신! ({best_val_acc:.4f} -> {val_epoch_acc:.4f}). 모델을 저장합니다.")
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"\n✅ 학습 완전 종료! 최고 성능의 모델이 안전하게 보관되어 있습니다. (Best Val Acc: {best_val_acc:.4f})")

if __name__ == '__main__':
    train()