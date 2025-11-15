"""
실시간 제스처 인식 테스트 스크립트
훈련된 모델을 로드하여 웹캠으로 손 제스처를 실시간으로 판별
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
    훈련된 모델 로드
    
    Returns:
        model: 로드된 모델
        scaler: 스케일러 (있으면), 없으면 None
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"[ERROR] 모델 파일을 찾을 수 없습니다: {MODEL_FILE}\n"
            f"먼저 train.py를 실행하여 모델을 훈련시켜주세요."
        )
    
    print(f"모델 로드 중: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print(f"✓ 모델 로드 완료: {type(model).__name__}")
    
    # 스케일러가 있으면 로드
    scaler = None
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        print(f"✓ 스케일러 로드 완료")
    
    return model, scaler


def predict_gesture(model, features, scaler=None, threshold=CONFIDENCE_THRESHOLD):
    """
    특징 벡터로부터 제스처 예측
    
    Args:
        model: 훈련된 모델
        features: 특징 벡터 (40,)
        scaler: 스케일러 (선택)
        threshold: 확신도 임계값 (이 값보다 낮으면 "Unknown" 반환)
        
    Returns:
        predicted_label: 예측된 제스처 라벨 또는 "Unknown"
        confidence: 예측 확신도 (0~1)
    """
    # 특징을 2D 배열로 변환 (1, 40)
    features_2d = features.reshape(1, -1)
    
    # 스케일러가 있으면 적용
    if scaler is not None:
        features_2d = scaler.transform(features_2d)
    
    # 예측
    predicted_label = model.predict(features_2d)[0]
    
    # 확신도 계산 (확률을 지원하는 모델인 경우)
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        # 확률 기반 확신도 (SVM, Random Forest 등)
        probabilities = model.predict_proba(features_2d)[0]
        confidence = np.max(probabilities)
        
        # 추가 검증: 1위와 2위 확률 차이가 작으면 확신도 감소
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1:
            margin = sorted_probs[0] - sorted_probs[1]
            # margin이 작으면 (0.1 미만) 확신도를 낮춤
            if margin < 0.1:
                confidence = confidence * 0.7  # 확신도 30% 감소
    elif hasattr(model, 'decision_function'):
        # SVM의 경우 decision_function 사용 (probability=False일 때)
        decision_values = model.decision_function(features_2d)[0]
        if isinstance(decision_values, np.ndarray):
            confidence = np.max(decision_values) / np.sum(np.abs(decision_values))
        else:
            confidence = 1.0
    else:
        # KNN 등 확률을 지원하지 않는 모델
        confidence = 0.5  # 기본값을 낮춤 (Unknown 처리 유도)
    
    # 확신도가 임계값보다 낮으면 Unknown으로 처리
    if confidence < threshold:
        predicted_label = "Unknown"
    
    return predicted_label, confidence


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("실시간 ASL 제스처 인식 테스트")
    print("="*60)
    print(f"확신도 임계값: {CONFIDENCE_THRESHOLD:.0%} (이 값 미만은 Unknown 처리)")
    print("ESC 또는 Q 키를 눌러 종료\n")
    
    # 모델 로드
    try:
        model, scaler = load_model()
    except FileNotFoundError as e:
        print(e)
        return
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 예측 결과 안정화를 위한 버퍼 (최근 N개 예측 저장)
    prediction_buffer = []
    buffer_size = 5
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        print("✓ 웹캠 시작\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
                break
            
            # 좌우 반전
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 손 검출
            results = hands.process(rgb_frame)
            
            # 화면에 표시할 텍스트
            status_text = "No hand detected"
            gesture_text = ""
            confidence_text = ""
            color = (0, 0, 255)  # 빨간색 (손 없음)
            
            # 손이 감지되면 예측
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 손 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # 특징 추출
                    features = extract_features_from_mediapipe(hand_landmarks)
                    
                    if features is not None:
                        # 제스처 예측
                        predicted_label, confidence = predict_gesture(model, features, scaler)
                        
                        # 예측 버퍼에 추가 (안정화)
                        prediction_buffer.append(predicted_label)
                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)
                        
                        # 가장 많이 예측된 제스처 선택
                        if len(prediction_buffer) > 0:
                            from collections import Counter
                            most_common = Counter(prediction_buffer).most_common(1)[0][0]
                            
                            # Unknown 제스처 처리
                            if most_common == "Unknown":
                                status_text = "Hand detected"
                                gesture_text = "Gesture: Unknown"
                                confidence_text = f"Confidence: {confidence:.2%} (Too low)"
                                color = (0, 165, 255)  # 주황색 (불확실)
                            else:
                                status_text = "Recognizing gesture..."
                                gesture_text = f"Gesture: {most_common}"
                                confidence_text = f"Confidence: {confidence:.2%}"
                                color = (0, 255, 0)  # 초록색 (인식 성공)
                            
                            # 콘솔에도 출력
                            # print(f"[PREDICT] {most_common} (Confidence: {confidence:.2%})", end='\r')
            else:
                # 손이 없으면 버퍼 초기화
                prediction_buffer.clear()
            
            # 화면에 정보 표시
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if gesture_text:
                cv2.putText(frame, gesture_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                cv2.putText(frame, confidence_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 사용법 및 임계값 표시
            cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD:.0%} | ESC or Q: Exit", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Gesture Recognition Test', frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                print("\n\n프로그램을 종료합니다.")
                break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
