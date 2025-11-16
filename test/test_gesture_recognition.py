"""
Real-time gesture recognition test script
Loads trained model and recognizes hand gestures in real-time from webcam
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import sys

# 상위 디렉토리를 path에 추가 (core 모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_extractor import extract_features_from_mediapipe


# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 모델 파일 경로
MODEL_FILE = os.path.join("..", "models", "asl_skill_model.pkl")
SCALER_FILE = os.path.join("..", "models", "scaler.pkl")

# 확신도 임계값 (이 값보다 낮으면 "Unknown"으로 처리)
CONFIDENCE_THRESHOLD = 0.95  # 60% 미만이면 제스처로 인식하지 않음


def load_model():
    """
    Load trained model
    
    Returns:
        model: Loaded model
        scaler: Scaler (if exists), otherwise None
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"[ERROR] Model file not found: {MODEL_FILE}\n"
            f"Please run train.py first to train the model."
        )
    
    print(f"Loading model: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # Load scaler if exists
    scaler = None
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        print(f"✓ Scaler loaded")
    
    return model, scaler


def predict_gesture(model, features, scaler=None, threshold=CONFIDENCE_THRESHOLD):
    """
    Predict gesture from feature vector
    
    Args:
        model: Trained model
        features: Feature vector (40,)
        scaler: Scaler (optional)
        threshold: Confidence threshold (return "Unknown" if below this value)
        
    Returns:
        predicted_label: Predicted gesture label or "Unknown"
        confidence: Prediction confidence (0~1)
    """
    # 특징을 2D 배열로 변환 (1, 40)
    features_2d = features.reshape(1, -1)
    
    # 스케일러가 있으면 적용
    if scaler is not None:
        features_2d = scaler.transform(features_2d)
    
    # 예측
    predicted_label = model.predict(features_2d)[0]
    
    # Confidence calculation (for models that support probabilities)
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        # Probability-based confidence (SVM, Random Forest, etc.)
        probabilities = model.predict_proba(features_2d)[0]
        confidence = np.max(probabilities)
        
        # Additional validation: Reduce confidence if margin between top 2 is small
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1:
            margin = sorted_probs[0] - sorted_probs[1]
            # If margin is small (< 0.1), reduce confidence
            if margin < 0.1:
                confidence = confidence * 0.7  # Reduce confidence by 30%
    elif hasattr(model, 'decision_function'):
        # For SVM when probability=False
        decision_values = model.decision_function(features_2d)[0]
        if isinstance(decision_values, np.ndarray):
            confidence = np.max(decision_values) / np.sum(np.abs(decision_values))
        else:
            confidence = 1.0
    else:
        # For models that don't support probabilities like KNN
        confidence = 0.5  # Lower default to encourage Unknown
    
    # If confidence is below threshold, treat as Unknown
    if confidence < threshold:
        predicted_label = "Unknown"
    
    return predicted_label, confidence


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("Real-time ASL Gesture Recognition Test")
    print("="*60)
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%} (Treat as Unknown if below)")
    print("Press ESC or Q to exit\n")
    
    # 모델 로드
    try:
        model, scaler = load_model()
    except FileNotFoundError as e:
        print(e)
        return
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Prediction buffer for stabilization (store recent N predictions)
    prediction_buffer = []
    buffer_size = 5
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        print("✓ Webcam started\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame from camera.")
                break
            
            # 좌우 반전
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 손 검출
            results = hands.process(rgb_frame)
            
            # Text to display on screen
            status_text = "No hand detected"
            gesture_text = ""
            confidence_text = ""
            color = (0, 0, 255)  # Red (no hand)
            
            # 손이 감지되면 예측
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # Extract features
                    features = extract_features_from_mediapipe(hand_landmarks)
                    
                    if features is not None:
                        # Predict gesture
                        predicted_label, confidence = predict_gesture(model, features, scaler)
                        
                        # Add to prediction buffer (stabilization)
                        prediction_buffer.append(predicted_label)
                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)
                        
                        # Select most common gesture
                        if len(prediction_buffer) > 0:
                            from collections import Counter
                            most_common = Counter(prediction_buffer).most_common(1)[0][0]
                            
                            # Handle Unknown gesture
                            if most_common == "Unknown":
                                status_text = "Hand detected"
                                gesture_text = "Gesture: Unknown"
                                confidence_text = f"Confidence: {confidence:.2%} (Too low)"
                                color = (0, 165, 255)  # Orange (uncertain)
                            else:
                                status_text = "Recognizing gesture..."
                                gesture_text = f"Gesture: {most_common}"
                                confidence_text = f"Confidence: {confidence:.2%}"
                                color = (0, 255, 0)  # Green (success)
                            
                            # Also print to console
                            # print(f"[PREDICT] {most_common} (Confidence: {confidence:.2%})", end='\r')
            else:
                # Clear buffer if no hand
                prediction_buffer.clear()
            
            # Display info on screen
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if gesture_text:
                cv2.putText(frame, gesture_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                cv2.putText(frame, confidence_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display usage and threshold
            cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD:.0%} | ESC or Q: Exit", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Gesture Recognition Test', frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                print("\n\nExiting program.")
                break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
