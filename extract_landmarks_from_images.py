"""
이미지 파일에서 손 랜드마크를 추출하여 gestures.csv에 추가하는 스크립트
사용법:
  - images/ 폴더에 제스처별 이미지를 저장 (예: images/A_01.jpg, images/C_02.jpg)
  - 각 이미지 파일명에 제스처 라벨 포함 (예: A_01.jpg → 라벨 'A')
  - 실행 후 gestures.csv에 특징 벡터와 라벨이 추가됨
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import random
from core.feature_extractor import normalize_landmarks

# IMAGE_ROOT = os.path.join("data", "images", "archive", "asl_alphabet_train", "asl_alphabet_train")
IMAGE_ROOT = os.path.join("data", "images", "archive2", "ASL_Alphabet_Dataset", "asl_alphabet_train")
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "gestures.csv")

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands


def extract_label_from_path(img_path):
    """이미지 경로에서 라벨 추출 (폴더명 기반)"""
    # 예: .../asl_alphabet_train/A/A123.jpg → 'A'
    parts = os.path.normpath(img_path).split(os.sep)
    # 폴더 구조: .../asl_alphabet_train/{alphabet}/{alphabet}123.jpg
    # 라벨은 폴더명
    if len(parts) >= 2:
        return parts[-2]
    return "Unknown"


def process_images(target_gestures=None, max_samples_per_gesture=None):
    """
    제스처별 하위 폴더를 순회하며 이미지에서 랜드마크 추출 및 저장
    
    Args:
        target_gestures: 추출할 제스처 리스트 (예: ['A', 'C', 'L']) None이면 모든 제스처
        max_samples_per_gesture: 제스처당 최대 샘플 수 (None이면 제한 없음)
    """
    print(f"\n[DEBUG] 이미지 루트 경로: {IMAGE_ROOT}")
    print(f"[DEBUG] 절대 경로: {os.path.abspath(IMAGE_ROOT)}")
    
    if not os.path.exists(IMAGE_ROOT):
        print(f"[ERROR] 이미지 루트 폴더가 없습니다: {IMAGE_ROOT}")
        print(f"[ERROR] 절대 경로가 존재하지 않습니다: {os.path.abspath(IMAGE_ROOT)}")
        return
    
    print(f"[DEBUG] 루트 폴더 존재 확인 완료\n")

    collected_data = []
    gesture_counts = {}  # 제스처별 수집 개수 추적

    # 루트 폴더의 모든 항목 확인
    all_items = os.listdir(IMAGE_ROOT)
    print(f"[DEBUG] 루트 폴더 내 항목 수: {len(all_items)}")
    print(f"[DEBUG] 전체 항목: {all_items[:10]}")  # 처음 10개만 출력
    
    folders = [item for item in all_items if os.path.isdir(os.path.join(IMAGE_ROOT, item))]
    print(f"[DEBUG] 폴더 항목: {folders}\n")
    
    # 이미지 반전 여부 입력
    use_flip = input("이미지를 좌우 반전하여 데이터 증강을 하시겠습니까? (y/n, 기본값: y): ").strip().lower()
    if use_flip != 'n':
        use_flip = True
        print("✓ 이미지 반전 적용 - 데이터가 2배로 증강됩니다.\n")
    else:
        use_flip = False
        print("✓ 원본 이미지만 사용합니다.\n")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        # 제스처별 폴더 순회
        for gesture_folder in folders:
            # 타겟 제스처 필터링
            if target_gestures and gesture_folder not in target_gestures:
                print(f"[DEBUG] '{gesture_folder}' 폴더는 타겟 제스처가 아니므로 건너뜁니다.")
                continue
                
            gesture_path = os.path.join(IMAGE_ROOT, gesture_folder)
            
            print(f"\n[{gesture_folder}] 폴더 처리 중...")
            print(f"[DEBUG] 제스처 폴더 경로: {gesture_path}")
            gesture_counts[gesture_folder] = 0
            
            image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"[DEBUG] 발견된 이미지 파일 수: {len(image_files)}")
            
            # 랜덤 샘플링 (최대 샘플 수가 설정되어 있고, 이미지가 더 많을 경우)
            if max_samples_per_gesture and len(image_files) > max_samples_per_gesture:
                # 반전 사용 시 절반만 샘플링 (나머지는 반전으로 채움)
                sample_count = max_samples_per_gesture // 2 if use_flip else max_samples_per_gesture
                image_files = random.sample(image_files, sample_count)
                print(f"[DEBUG] 랜덤 샘플링: {len(image_files)}개 선택됨")
            
            if len(image_files) > 0:
                print(f"[DEBUG] 첫 5개 이미지: {image_files[:5]}")
            
            for img_name in image_files:
                # 최대 샘플 수 체크 (반전 이미지 포함)
                if max_samples_per_gesture and gesture_counts[gesture_folder] >= max_samples_per_gesture:
                    print(f"  → {gesture_folder}: 최대 {max_samples_per_gesture}개 도달, 다음 제스처로 이동")
                    break
                
                img_path = os.path.join(gesture_path, img_name)
                label = extract_label_from_path(img_path)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARNING] 이미지를 읽을 수 없습니다: {img_name}")
                    continue
                
                # 원본 이미지 처리
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    features = normalize_landmarks(landmarks)
                    if features is not None:
                        collected_data.append([features, label])
                        gesture_counts[gesture_folder] += 1
                        if gesture_counts[gesture_folder] % 10 == 0:
                            print(f"  → {gesture_folder}: {gesture_counts[gesture_folder]}개 수집됨...")
                else:
                    if gesture_counts[gesture_folder] < 3:
                        print(f"[WARNING] 손 미검출 (원본): {img_name}")
                
                # 반전 이미지 처리
                if use_flip and (max_samples_per_gesture is None or gesture_counts[gesture_folder] < max_samples_per_gesture):
                    flipped_image = cv2.flip(image, 1)  # 좌우 반전
                    rgb_flipped = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
                    results_flipped = hands.process(rgb_flipped)
                    
                    if results_flipped.multi_hand_landmarks:
                        hand_landmarks_flipped = results_flipped.multi_hand_landmarks[0]
                        landmarks_flipped = [(lm.x, lm.y) for lm in hand_landmarks_flipped.landmark]
                        features_flipped = normalize_landmarks(landmarks_flipped)
                        if features_flipped is not None:
                            collected_data.append([features_flipped, label])
                            gesture_counts[gesture_folder] += 1
                            if gesture_counts[gesture_folder] % 10 == 0:
                                print(f"  → {gesture_folder}: {gesture_counts[gesture_folder]}개 수집됨...")
    
    # 최종 통계 출력
    print(f"\n{'='*50}")
    print("수집 완료 요약:")
    if gesture_counts:
        for gesture, count in gesture_counts.items():
            print(f"  {gesture}: {count}개")
    else:
        print("  수집된 제스처가 없습니다.")
    print(f"{'='*50}\n")
    
    # 저장
    save_to_csv(collected_data)


def save_to_csv(data_list):
    """수집된 데이터를 gestures.csv에 추가 저장"""
    if len(data_list) == 0:
        print("[WARNING] 저장할 데이터가 없습니다.")
        return
    
    features = np.array([item[0] for item in data_list])
    labels = [item[1] for item in data_list]
    columns = [f"feature_{i}" for i in range(features.shape[1])] + ["label"]
    df_new = pd.DataFrame(
        np.column_stack([features, labels]),
        columns=columns
    )
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
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


def main():
    print("\n" + "="*50)
    print("이미지에서 손 제스처 랜드마크 추출")
    print("="*50)
    
    # 사용자 입력: 추출할 제스처 선택
    print("\n추출할 제스처를 입력하세요.")
    print("예: A,C,L,S (쉼표로 구분) 또는 Enter (모든 제스처)")
    gestures_input = input("제스처: ").strip()
    
    target_gestures = None
    if gestures_input:
        target_gestures = [g.strip().upper() for g in gestures_input.split(',')]
        print(f"선택된 제스처: {target_gestures}")
    else:
        print("모든 제스처를 추출합니다.")
    
    # 사용자 입력: 제스처당 최대 샘플 수
    print("\n제스처당 최대 샘플 수를 입력하세요.")
    print("예: 100 또는 Enter (제한 없음)")
    max_samples_input = input("최대 샘플 수: ").strip()
    
    max_samples = None
    if max_samples_input:
        try:
            max_samples = int(max_samples_input)
            print(f"제스처당 최대 {max_samples}개 샘플 추출")
        except ValueError:
            print("잘못된 입력입니다. 제한 없이 추출합니다.")
    else:
        print("제한 없이 모든 샘플을 추출합니다.")
    
    print()
    process_images(target_gestures, max_samples)
    print("\n작업 완료!")


if __name__ == "__main__":
    main()
