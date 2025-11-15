"""
ASL 제스처 데이터 수집 스크립트
웹캠으로 손 제스처를 실시간 캡처하여 data/gestures.csv에 저장
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime
from core.feature_extractor import extract_features_from_mediapipe


# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 수집할 제스처 목록 (필요에 따라 수정 가능)
GESTURE_LIST = ["A", "B", "C", "L", "R", "U", "Y", "Idle"]

# 데이터 저장 경로
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "gestures.csv")


def setup_data_directory():
    """데이터 디렉토리 생성"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✓ '{DATA_DIR}' 디렉토리 생성 완료")


def collect_gesture_data(gesture_name: str, duration: int = 5, fps: int = 30) -> list:
    """
    특정 제스처의 데이터를 수집
    
    Args:
        gesture_name: 수집할 제스처 이름 (예: 'A', 'B', 'Idle')
        duration: 수집 시간 (초)
        fps: 카메라 프레임 레이트
        
    Returns:
        수집된 데이터 리스트 (각 항목: [feature_vector, label])
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    collected_data = []
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        print(f"\n{'='*50}")
        print(f"제스처 '{gesture_name}' 수집 준비 중...")
        print(f"3초 후 자동으로 {duration}초간 녹화를 시작합니다.")
        print("여러 각도로 손을 움직이며 제스처를 취해주세요!")
        print(f"{'='*50}\n")
        
        # 3초 카운트다운
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Ready in {i}...", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1000)
        
        # 데이터 수집 시작
        print(f"✓ 녹화 시작! ({duration}초)")
        frame_count = 0
        total_frames = duration * fps
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
                break
            
            # 좌우 반전
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 손 검출
            results = hands.process(rgb_frame)
            
            # 진행 상황 표시
            progress = int((frame_count / total_frames) * 100)
            cv2.putText(frame, f"Recording: {gesture_name}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Progress: {progress}%", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Frames: {len(collected_data)}", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 손이 감지되면 특징 추출
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 손 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 특징 추출
                    features = extract_features_from_mediapipe(hand_landmarks)
                    if features is not None:
                        collected_data.append([features, gesture_name])
            
            cv2.imshow('Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 사용자가 수집을 중단했습니다.")
                break
            
            frame_count += 1
        
        print(f"✓ 녹화 완료! 수집된 샘플 수: {len(collected_data)}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return collected_data


def save_to_csv(data_list: list):
    """
    수집된 데이터를 CSV 파일로 저장
    
    Args:
        data_list: [[feature_vector, label], ...] 형태의 데이터
    """
    if len(data_list) == 0:
        print("[WARNING] 저장할 데이터가 없습니다.")
        return
    
    # 데이터프레임 생성
    features = np.array([item[0] for item in data_list])
    labels = [item[1] for item in data_list]
    
    # 컬럼 이름 생성 (feature_0, feature_1, ..., feature_39, label)
    columns = [f"feature_{i}" for i in range(features.shape[1])] + ["label"]
    
    # 데이터프레임 생성
    df_new = pd.DataFrame(
        np.column_stack([features, labels]),
        columns=columns
    )
    
    # 기존 CSV 파일이 있으면 추가, 없으면 새로 생성
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_FILE, index=False)
        print(f"✓ 기존 데이터에 {len(df_new)}개 샘플 추가됨")
        print(f"  총 샘플 수: {len(df_combined)}")
    else:
        df_new.to_csv(CSV_FILE, index=False)
        print(f"✓ 새 CSV 파일 생성: {CSV_FILE}")
        print(f"  저장된 샘플 수: {len(df_new)}")


def show_data_summary():
    """저장된 데이터의 요약 정보 출력"""
    if not os.path.exists(CSV_FILE):
        print("[INFO] 아직 수집된 데이터가 없습니다.")
        return
    
    df = pd.read_csv(CSV_FILE)
    print("\n" + "="*50)
    print("현재 저장된 데이터 요약")
    print("="*50)
    print(f"총 샘플 수: {len(df)}")
    print(f"특징 벡터 차원: {len(df.columns) - 1}")
    print("\n제스처별 샘플 수:")
    print(df['label'].value_counts().to_string())
    print("="*50 + "\n")


def main():
    """메인 실행 함수"""
    print("\n" + "="*50)
    print("ASL 제스처 데이터 수집 시스템")
    print("="*50)
    
    setup_data_directory()
    show_data_summary()
    
    print("\n수집할 제스처 목록:")
    for i, gesture in enumerate(GESTURE_LIST, 1):
        print(f"  {i}. {gesture}")
    
    while True:
        print("\n" + "-"*50)
        gesture_name = input("녹화할 제스처 이름을 입력하세요 (종료: q): ").strip()
        
        if gesture_name.lower() == 'q':
            print("\n프로그램을 종료합니다.")
            break
        
        if not gesture_name:
            print("[ERROR] 제스처 이름을 입력해주세요.")
            continue
        
        # 수집 시간 입력
        try:
            duration_input = input(f"수집 시간(초)을 입력하세요 (기본값: 10초): ").strip()
            duration = int(duration_input) if duration_input else 10
        except ValueError:
            print("[WARNING] 잘못된 입력입니다. 기본값 10초로 설정합니다.")
            duration = 10
        
        # 데이터 수집
        collected = collect_gesture_data(gesture_name, duration)
        
        # CSV로 저장
        save_to_csv(collected)
        
        # 요약 정보 출력
        show_data_summary()


if __name__ == "__main__":
    main()
