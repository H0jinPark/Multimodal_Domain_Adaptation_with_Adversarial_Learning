import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 상위 폴더 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader import get_dataloaders
from DANN_model import DANN1DCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 DANN 학습 기기: {device}")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessed_data')
    # train_loader(Source), val_loader(Source-Val), tgt_loader(Target)
    train_loader, val_loader, tgt_loader, _ = get_dataloaders(data_dir=data_dir, batch_size=64)

    model = DANN1DCNN(num_classes=10).to(device)
    
    # Loss 함수 설정
    criterion_class = nn.CrossEntropyLoss() # 운동 분류용
    criterion_domain = nn.CrossEntropyLoss() # 도메인 분류용 (Source vs Target)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30 # DANN은 베이스라인보다 조금 더 긴 호흡이 필요합니다.
    best_val_acc = 0.0
    save_path = os.path.join(os.path.dirname(__file__), 'dann_model.pt')

    print("\n🎭 DANN Adversarial Training 시작...")
    
    for epoch in range(epochs):
        model.train()
        
        # DANN의 핵심: Source와 Target 로더를 하나로 묶어(zip) 동시에 꺼내기
        # 두 로더의 길이가 다를 수 있으므로 짧은 쪽이 끝나면 다시 시작하도록 구성
        len_dataloader = min(len(train_loader), len(tgt_loader))
        data_zip = zip(train_loader, tgt_loader)
        
        running_loss = 0.0
        correct_label = 0
        total_label = 0
        
        pbar = tqdm(enumerate(data_zip), total=len_dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for i, (src_data, tgt_data) in pbar:
            # 1. 학습 진행도에 따른 alpha 스케줄링 (0 -> 1)
            p = float(i + epoch * len_dataloader) / (epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # --- Source 데이터 처리 ---
            src_inputs, src_labels = src_data
            src_inputs, src_labels = src_inputs.to(device), src_labels.to(device)
            batch_size = src_inputs.size(0)
            
            # Source의 도메인 정답은 0
            domain_labels_src = torch.zeros(batch_size).long().to(device)
            
            # --- Target 데이터 처리 ---
            tgt_inputs, _ = tgt_data # Target 라벨은 학습에 절대 쓰지 않음 (Unsupervised)
            tgt_inputs = tgt_inputs.to(device)
            
            # Target의 도메인 정답은 1
            domain_labels_tgt = torch.ones(tgt_inputs.size(0)).long().to(device)
            
            # --- 통합 전방향 연산 ---
            optimizer.zero_grad()
            
            # 1️⃣ Source 학습 (Label Loss + Domain Loss)
            src_class_preds, src_domain_preds = model(src_inputs, alpha=alpha)
            loss_label = criterion_class(src_class_preds, src_labels)
            loss_domain_src = criterion_domain(src_domain_preds, domain_labels_src)
            
            # 2️⃣ Target 학습 (Domain Loss만 계산)
            _, tgt_domain_preds = model(tgt_inputs, alpha=alpha)
            loss_domain_tgt = criterion_domain(tgt_domain_preds, domain_labels_tgt)
            
            # 3️⃣ 전체 손실함수 (Adversarial Loss)
            # 도메인 분류기가 멍청해지도록 유도함
            total_loss = loss_label + (loss_domain_src + loss_domain_tgt)
            
            total_loss.backward()
            optimizer.step()
            
            # 통계 기록
            running_loss += total_loss.item()
            _, predicted = torch.max(src_class_preds, 1)
            total_label += src_labels.size(0)
            correct_label += (predicted == src_labels).sum().item()
            
            pbar.set_postfix(L_label=f"{loss_label.item():.3f}", L_dom=f"{(loss_domain_src+loss_domain_tgt).item():.3f}", acc=correct_label/total_label)

        # ==========================================
        # 검증 (Validation) - 베이스라인과 동일
        # ==========================================
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs, alpha=0) # 검증 시에는 GRL 무의미
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f"✨ Epoch {epoch+1} Summary - Train Acc: {correct_label/total_label:.4f}, Val Acc: {val_acc:.4f}, Alpha: {alpha:.3f}")
        
        # Best 모델 저장
        if val_acc > best_val_acc:
            print(f"   🏆 Best Val Acc 갱신! ({best_val_acc:.4f} -> {val_acc:.4f})")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ DANN 학습 완료! 최고 Val Acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    train()