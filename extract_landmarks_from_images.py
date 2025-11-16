"""
Script to extract hand landmarks from image files and add to gestures.csv
Usage:
  - Save gesture images in images/ folder (e.g., images/A_01.jpg, images/C_02.jpg)
  - Include gesture label in each image filename (e.g., A_01.jpg → label 'A')
  - After execution, feature vectors and labels are added to gestures.csv
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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands


def extract_label_from_path(img_path):
    """Extract label from image path (based on folder name)"""
    # Example: .../asl_alphabet_train/A/A123.jpg → 'A'
    parts = os.path.normpath(img_path).split(os.sep)
    # Folder structure: .../asl_alphabet_train/{alphabet}/{alphabet}123.jpg
    # Label is the folder name
    if len(parts) >= 2:
        return parts[-2]
    return "Unknown"


def process_images(target_gestures=None, max_samples_per_gesture=None):
    """
    Traverse gesture subfolders and extract landmarks from images and save
    
    Args:
        target_gestures: List of gestures to extract (e.g., ['A', 'C', 'L']) None for all gestures
        max_samples_per_gesture: Max samples per gesture (None for no limit)
    """
    print(f"\n[DEBUG] Image root path: {IMAGE_ROOT}")
    print(f"[DEBUG] Absolute path: {os.path.abspath(IMAGE_ROOT)}")
    
    if not os.path.exists(IMAGE_ROOT):
        print(f"[ERROR] Image root folder does not exist: {IMAGE_ROOT}")
        print(f"[ERROR] Absolute path does not exist: {os.path.abspath(IMAGE_ROOT)}")
        return
    
    print(f"[DEBUG] Root folder existence confirmed\n")

    collected_data = []
    gesture_counts = {}  # Track collection count per gesture

    # Check all items in root folder
    all_items = os.listdir(IMAGE_ROOT)
    print(f"[DEBUG] Number of items in root folder: {len(all_items)}")
    print(f"[DEBUG] All items: {all_items[:10]}")  # Print first 10 only
    
    folders = [item for item in all_items if os.path.isdir(os.path.join(IMAGE_ROOT, item))]
    print(f"[DEBUG] Folder items: {folders}\n")
    
    # Input for image flipping
    use_flip = input("Do you want to flip images for data augmentation? (y/n, default: y): ").strip().lower()
    flip_mode = None
    
    if use_flip != 'n':
        use_flip = True
        print("\nSelect flip mode:")
        print("  1. Half flip - 50% original + 50% flipped (faster, maintains variety)")
        print("  2. Full flip - 0% original + 100% flipped (complete switch to opposite hand)")
        flip_choice = input("Choice (1-2, default: 1): ").strip() or "1"
        
        if flip_choice == "2":
            flip_mode = "full"
            print("✓ Full flip mode - Process all images as flipped only.\n")
        else:
            flip_mode = "half"
            print("✓ Half flip mode - Randomly flip half of images.\n")
    else:
        use_flip = False
        print("✓ Use original images only.\n")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        # 제스처별 폴더 순회
        for gesture_folder in folders:
            # Target gesture filtering
            if target_gestures and gesture_folder not in target_gestures:
                print(f"[DEBUG] '{gesture_folder}' folder is not a target gesture, skipping.")
                continue
                
            gesture_path = os.path.join(IMAGE_ROOT, gesture_folder)
            
            print(f"\n[{gesture_folder}] Processing folder...")
            print(f"[DEBUG] Gesture folder path: {gesture_path}")
            gesture_counts[gesture_folder] = 0
            
            image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"[DEBUG] Number of image files found: {len(image_files)}")
            
            # Random sampling (if max samples set and more images available)
            if max_samples_per_gesture and len(image_files) > max_samples_per_gesture:
                if flip_mode == "half":
                    # Half flip mode: Select only half of max samples (rest filled with flips)
                    sample_count = max_samples_per_gesture // 2
                    image_files = random.sample(image_files, sample_count)
                    print(f"[DEBUG] Random sampling (half flip): {len(image_files)} selected")
                elif flip_mode == "full":
                    # Full flip mode: Select max samples (process flips only, skip originals)
                    image_files = random.sample(image_files, max_samples_per_gesture)
                    print(f"[DEBUG] Random sampling (full flip): {len(image_files)} selected (flips only)")
                else:
                    # No flip: Select only max samples
                    image_files = random.sample(image_files, max_samples_per_gesture)
                    print(f"[DEBUG] Random sampling: {len(image_files)} selected")
            
            if len(image_files) > 0:
                print(f"[DEBUG] First 5 images: {image_files[:5]}")
            
            for img_name in image_files:
                # Max samples check (including flipped images)
                if max_samples_per_gesture and gesture_counts[gesture_folder] >= max_samples_per_gesture:
                    print(f"  → {gesture_folder}: Reached max {max_samples_per_gesture}, moving to next gesture")
                    break
                
                img_path = os.path.join(gesture_path, img_name)
                label = extract_label_from_path(img_path)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARNING] Cannot read image: {img_name}")
                    continue
                
                # Skip original image if full flip mode
                if flip_mode != "full":
                    # Process original image
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
                                print(f"  → {gesture_folder}: {gesture_counts[gesture_folder]} collected...")
                    else:
                        if gesture_counts[gesture_folder] < 3:
                            print(f"[WARNING] Hand not detected (original): {img_name}")
                
                # Process flipped image
                should_flip = False
                if use_flip and (max_samples_per_gesture is None or gesture_counts[gesture_folder] < max_samples_per_gesture):
                    should_flip = True
                
                if should_flip:
                    flipped_image = cv2.flip(image, 1)  # Flip left-right
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
                                print(f"  → {gesture_folder}: {gesture_counts[gesture_folder]} collected...")
    
    # Final statistics output
    print(f"\n{'='*50}")
    print("Collection completed summary:")
    if gesture_counts:
        for gesture, count in gesture_counts.items():
            print(f"  {gesture}: {count}")
    else:
        print("  No gestures collected.")
    print(f"{'='*50}\n")
    
    # 저장
    save_to_csv(collected_data)


def save_to_csv(data_list):
    """Save collected data to gestures.csv"""
    if len(data_list) == 0:
        print("[WARNING] No data to save.")
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
        print(f"✓ Added {len(df_new)} samples to existing data")
        print(f"  Total samples: {len(df_combined)}")
    else:
        df_new.to_csv(CSV_FILE, index=False)
        print(f"✓ Created new CSV file: {CSV_FILE}")
        print(f"  Saved samples: {len(df_new)}")


def main():
    print("\n" + "="*50)
    print("Extract Hand Gesture Landmarks from Images")
    print("="*50)
    
    # User input: Select gestures to extract
    print("\nEnter gestures to extract.")
    print("Example: A,C,L,S (comma-separated) or Enter (all gestures)")
    gestures_input = input("Gestures: ").strip()
    
    target_gestures = None
    if gestures_input:
        target_gestures = [g.strip().upper() for g in gestures_input.split(',')]
        print(f"Selected gestures: {target_gestures}")
    else:
        print("Extracting all gestures.")
    
    # User input: Max samples per gesture
    print("\nEnter max samples per gesture.")
    print("Example: 100 or Enter (no limit)")
    max_samples_input = input("Max samples: ").strip()
    
    max_samples = None
    if max_samples_input:
        try:
            max_samples = int(max_samples_input)
            print(f"Extracting max {max_samples} samples per gesture")
        except ValueError:
            print("Invalid input. Extracting without limit.")
    else:
        print("Extracting all samples without limit.")
    
    print()
    process_images(target_gestures, max_samples)
    print("\nTask completed!")


if __name__ == "__main__":
    main()
