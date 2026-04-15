import numpy as np
import pandas as pd
import os

def create_sliding_windows(df, window_size=2000, step_size=1000):
    """
    데이터프레임에 슬라이딩 윈도우를 적용하여 3차원 배열로 변환합니다.
    
    Args:
        df (pd.DataFrame): 2000Hz로 리샘플링 및 병합이 완료된 데이터프레임
        window_size (int): 윈도우 크기 (기본값: 2000 -> 1.0초)
        step_size (int): 윈도우가 이동하는 보폭 (기본값: 1000 -> 50% 겹침)
        
    Returns:
        X (np.ndarray): 형태 [N, window_size, 5] 의 센서 데이터
        y (np.ndarray): 형태 [N] 의 동작 라벨
        subjects (np.ndarray): 형태 [N] 의 피험자 정보 (도메인 적응 타겟팅용)
    """
    
    # 사용할 센서 채널 정의
    sensor_cols = ['emg_3', 'emg_4', 'IMU_3_X', 'IMU_3_Y', 'IMU_3_Z']
    
    X_list = []
    y_list = []
    subject_list = []
    
    # 파일(또는 피험자+동작) 단위로 그룹화하여 경계선이 섞이는 것을 방지
    for filename, group in df.groupby('filename'):
        # 안전을 위해 시간순 정렬
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        # 센서 데이터, 라벨, 피험자 정보를 Numpy 배열로 추출
        data_values = group[sensor_cols].values
        label_val = group['label'].iloc[0]
        subject_val = group['subject'].iloc[0]
        
        n_samples = len(group)
        
        # 슬라이딩 윈도우 적용
        for start_idx in range(0, n_samples - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = data_values[start_idx:end_idx, :]
            
            X_list.append(window_data)
            y_list.append(label_val)
            subject_list.append(subject_val)
            
    # 리스트를 Numpy 배열로 변환
    X = np.array(X_list)
    y = np.array(y_list)
    subjects = np.array(subject_list)
    
    return X, y, subjects

def main():
    import numpy as np
    import pandas as pd
    import os
    
    # ---------------------------------------------------------
    # [STEP 1] Parquet 실데이터 개별 로드
    # ---------------------------------------------------------
    print("--- [STEP 1] Parquet 데이터 로드 ---")
    
    # Source Domain (실험 1)과 Target Domain (실험 2)을 각각 따로 부릅니다.
    df_source = pd.read_parquet('data/fpyinal_first.parquet')
    df_target = pd.read_parquet('data/final_second.parquet')
    
    print(f"Source (실험 1) 크기: {df_source.shape}")
    print(f"Target (실험 2) 크기: {df_target.shape}")

    # ---------------------------------------------------------
    # [STEP 2] 슬라이딩 윈도우 개별 적용
    # ---------------------------------------------------------
    window_size = 2000 # 1.0초
    step_size = 1000   # 0.5초 겹침
    
    print(f"\n--- [STEP 2] 슬라이딩 윈도우 적용 (Window: {window_size}, Step: {step_size}) ---")
    
    print("Source Domain 처리 중...")
    X_src, y_src, sub_src = create_sliding_windows(df_source, window_size=window_size, step_size=step_size)
    
    print("Target Domain 처리 중...")
    X_tgt, y_tgt, sub_tgt = create_sliding_windows(df_target, window_size=window_size, step_size=step_size)
    
    # ---------------------------------------------------------
    # [STEP 3] 검증 및 상태 출력
    # ---------------------------------------------------------
    print("\n--- [STEP 3] 분할 결과 검증 ---")
    print(f"[Source Domain] X: {X_src.shape}, y: {y_src.shape}, subjects: {sub_src.shape}")
    print(f"[Target Domain] X: {X_tgt.shape}, y: {y_tgt.shape}, subjects: {sub_tgt.shape}")
    
    # ---------------------------------------------------------
    # [STEP 4] 넘파이 배열 분리 저장
    # ---------------------------------------------------------
    save_dir = './preprocessed_data'
    os.makedirs(save_dir, exist_ok=True)
    
    # Source 저장
    np.save(os.path.join(save_dir, 'X_source.npy'), X_src)
    np.save(os.path.join(save_dir, 'y_source.npy'), y_src)
    np.save(os.path.join(save_dir, 'sub_source.npy'), sub_src)
    
    # Target 저장
    np.save(os.path.join(save_dir, 'X_target.npy'), X_tgt)
    np.save(os.path.join(save_dir, 'y_target.npy'), y_tgt)
    np.save(os.path.join(save_dir, 'sub_target.npy'), sub_tgt)
    
    print(f"\n✅ 전처리 완료! Source와 Target 데이터가 완벽히 분리되어 '{save_dir}' 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main()