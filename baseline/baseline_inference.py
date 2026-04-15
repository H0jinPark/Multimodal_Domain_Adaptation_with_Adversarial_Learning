import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader import get_dataloaders
from baseline_model import Baseline1DCNN

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 원본 식별자 로드
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessed_data')
    y_tgt_raw = np.load(os.path.join(data_dir, 'y_target.npy'), allow_pickle=True)
    sub_tgt = np.load(os.path.join(data_dir, 'sub_target.npy'), allow_pickle=True) 

    # 2. 로더 및 라벨 인코더 세팅
    _, _, tgt_loader, le = get_dataloaders(data_dir=data_dir, batch_size=64)
    class_names = le.classes_
    
    # 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), 'baseline_model.pt')
    model = Baseline1DCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 3. 윈도우별 예측값 계산
    print("\n🎯 Target Domain 윈도우별 확률 계산 중...")
    
    X_tgt_tensor = tgt_loader.dataset.X.to(device)
    
    # ==========================================
    # 💡 [디버깅] 스케일링 된 Target 데이터의 상태 확인
    # X_tgt_tensor 형태: [샘플 수, 5(채널), 2000(시간)]
    # dim=[0, 2]를 주면 샘플과 시간에 대해 평균을 내서 '채널별(5개)' 통계량만 남습니다.
    # ==========================================
    print("\n🔍 [데이터 상태 점검] 스케일링된 Target Tensor 통계량")
    # Tensor를 보기 편하게 numpy로 바꾸고 소수점 4자리까지 출력
    ch_mean = X_tgt_tensor.mean(dim=[0, 2]).cpu().numpy().round(4)
    ch_std = X_tgt_tensor.std(dim=[0, 2]).cpu().numpy().round(4)
    
    print(f"채널별 평균(Mean): {ch_mean}")
    print(f"채널별 표준편차(Std): {ch_std}")
    print("==========================================\n")
    
    all_probs = []
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tgt_tensor), batch_size)):
            batch_x = X_tgt_tensor[i:i+batch_size]
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1) 
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)

    # 4. 운동 단위(Trial-level) 예측값 취합
    print("\n📊 운동 단위(Trial-level) 예측값 취합 중...")
    results_df = pd.DataFrame({
        'filename': sub_tgt,
        'true_label': y_tgt_raw
    })
    
    prob_cols = [f'prob_{c}' for c in class_names]
    results_df[prob_cols] = all_probs

    final_eval = results_df.groupby('filename').agg({
        'true_label': 'first',
        **{col: 'mean' for col in prob_cols}
    }).reset_index()

    final_probs = final_eval[prob_cols].values
    final_preds_idx = np.argmax(final_probs, axis=1)
    final_preds_label = le.inverse_transform(final_preds_idx)
    
    y_true = final_eval['true_label'].values
    y_pred = final_preds_label

    # 5. 결과 저장 및 시각화
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    report = classification_report(
        y_true, 
        y_pred, 
        labels=class_names,      
        target_names=class_names, 
        zero_division=0
    )
    
    with open(os.path.join(results_dir, 'trial_level_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== Baseline Model Trial-Level Performance (Target Domain) ===\n")
        f.write(f"Total Trials: {len(final_eval)}\n\n")
        f.write(report)
    
    print("\n✅ 운동 단위 평가 완료!")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Trial-Level Confusion Matrix (Target Domain)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    print(f"\n[마지막 확인]")
    print(f"y_true 샘플: {y_true[:5]}")
    print(f"y_pred 샘플: {y_pred[:5]}")
    
    plt.savefig(os.path.join(results_dir, 'trial_confusion_matrix.png'))

if __name__ == '__main__':
    inference()