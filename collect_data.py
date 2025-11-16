"""
ASL gesture data collection script
Captures hand gestures in real-time from webcam and saves to data/gestures.csv
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime
from core.feature_extractor import extract_features_from_mediapipe


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# List of gestures to collect (can be modified as needed)
GESTURE_LIST = ["A", "B", "C", "L", "R", "U", "Y", "Idle"]

# Data save path
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "gestures.csv")


def setup_data_directory():
    """Create data directory"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✓ Created '{DATA_DIR}' directory")


def collect_gesture_data(gesture_name: str, duration: int = 5, fps: int = 30) -> list:
    """
    Collect data for a specific gesture
    
    Args:
        gesture_name: Name of gesture to collect (e.g., 'A', 'B', 'Idle')
        duration: Collection time (seconds)
        fps: Camera frame rate
        
    Returns:
        List of collected data (each item: [feature_vector, label])
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
        print(f"Preparing to collect gesture '{gesture_name}'...")
        print(f"Recording will start automatically in 3 seconds for {duration} seconds.")
        print("Move your hand in different angles while performing the gesture!")
        print(f"{'='*50}\n")
        
        # 3-second countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Ready in {i}...", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1000)
        
        # Data collection start
        print(f"✓ Recording started! ({duration} seconds)")
        frame_count = 0
        total_frames = duration * fps
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame from camera.")
                break
            
            # Flip left-right
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand with MediaPipe
            results = hands.process(rgb_frame)
            
            # Progress display
            progress = int((frame_count / total_frames) * 100)
            cv2.putText(frame, f"Recording: {gesture_name}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Progress: {progress}%", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Frames: {len(collected_data)}", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # If hand is detected, extract features
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract features
                    features = extract_features_from_mediapipe(hand_landmarks)
                    if features is not None:
                        collected_data.append([features, gesture_name])
            
            cv2.imshow('Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User stopped collection.")
                break
            
            frame_count += 1
        
        print(f"✓ Recording completed! Collected samples: {len(collected_data)}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return collected_data


def save_to_csv(data_list: list):
    """
    Save collected data to CSV file
    
    Args:
        data_list: Data in format [[feature_vector, label], ...]
    """
    if len(data_list) == 0:
        print("[WARNING] No data to save.")
        return
    
    # Create dataframe
    features = np.array([item[0] for item in data_list])
    labels = [item[1] for item in data_list]
    
    # Generate column names (feature_0, feature_1, ..., feature_39, label)
    columns = [f"feature_{i}" for i in range(features.shape[1])] + ["label"]
    
    # Create dataframe
    df_new = pd.DataFrame(
        np.column_stack([features, labels]),
        columns=columns
    )
    
    # If existing CSV file exists, append; otherwise create new
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


def show_data_summary():
    """Print summary of saved data"""
    if not os.path.exists(CSV_FILE):
        print("[INFO] No data collected yet.")
        return
    
    df = pd.read_csv(CSV_FILE)
    print("\n" + "="*50)
    print("Current Saved Data Summary")
    print("="*50)
    print(f"Total samples: {len(df)}")
    print(f"Feature vector dimension: {len(df.columns) - 1}")
    print("\nSamples per gesture:")
    print(df['label'].value_counts().to_string())
    print("="*50 + "\n")


def main():
    """Main execution function"""
    print("\n" + "="*50)
    print("ASL Gesture Data Collection System")
    print("="*50)
    
    setup_data_directory()
    show_data_summary()
    
    print("\nGestures to collect:")
    for i, gesture in enumerate(GESTURE_LIST, 1):
        print(f"  {i}. {gesture}")
    
    while True:
        print("\n" + "-"*50)
        gesture_name = input("Enter gesture name to record (quit: q): ").strip()
        
        if gesture_name.lower() == 'q':
            print("\nExiting program.")
            break
        
        if not gesture_name:
            print("[ERROR] Please enter a gesture name.")
            continue
        
        # Input collection time
        try:
            duration_input = input(f"Enter collection time (seconds) (default: 10 seconds): ").strip()
            duration = int(duration_input) if duration_input else 10
        except ValueError:
            print("[WARNING] Invalid input. Setting to default 10 seconds.")
            duration = 10
        
        # Data collection
        collected = collect_gesture_data(gesture_name, duration)
        
        # Save to CSV
        save_to_csv(collected)
        
        # Print summary info
        show_data_summary()


if __name__ == "__main__":
    main()
