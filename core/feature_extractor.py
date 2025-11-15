"""
ASL 제스처 인식을 위한 특징 추출 모듈
MediaPipe Hands 랜드마크를 정규화하여 위치/크기 불변 특징 생성
"""

import numpy as np
from typing import List, Tuple, Optional


def normalize_landmarks(landmarks: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    """
    MediaPipe 손 랜드마크를 정규화하여 위치/크기 불변 특징 벡터 생성
    
    정규화 과정:
    1. 위치 정규화: 손목(0번)을 원점(0,0)으로 이동
    2. 크기 정규화: 손목(0번)~중지 바닥(9번) 거리로 전체 스케일 조정
    
    Args:
        landmarks: 21개 랜드마크의 (x, y) 좌표 리스트
        
    Returns:
        정규화된 40개 특징값 (20개 점의 x, y 좌표) 또는 None (에러 시)
    """
    if len(landmarks) != 21:
        print(f"[WARNING] 랜드마크 개수가 21개가 아닙니다: {len(landmarks)}")
        return None
    
    # NumPy 배열로 변환 (21, 2)
    points = np.array(landmarks, dtype=np.float32)
    
    # 1. 위치 정규화: 손목(0번)을 원점으로
    wrist = points[0].copy()
    points_centered = points - wrist
    
    # 2. 크기 정규화: 손목(0번) ~ 중지 바닥(9번) 거리로 스케일링
    middle_mcp = points_centered[9]  # 중지 바닥 (MCP: Metacarpophalangeal joint)
    scale = np.linalg.norm(middle_mcp)
    
    # 스케일이 너무 작으면 (손이 너무 멀거나 감지 오류) None 반환
    if scale < 1e-6:
        print("[WARNING] 손 크기가 너무 작습니다. 랜드마크 감지 오류일 수 있습니다.")
        return None
    
    points_normalized = points_centered / scale
    
    # 3. 손목 제외한 20개 점만 사용 (손목은 항상 (0,0)이므로 정보 없음)
    features = points_normalized[1:].flatten()  # (20, 2) -> (40,)
    
    return features


def extract_features_from_mediapipe(hand_landmarks) -> Optional[np.ndarray]:
    """
    MediaPipe Hands 객체에서 직접 특징 추출
    
    Args:
        hand_landmarks: mediapipe.python.solutions.hands.HandLandmark 객체
        
    Returns:
        정규화된 40개 특징값 또는 None
    """
    if hand_landmarks is None:
        return None
    
    # MediaPipe 랜드마크를 (x, y) 튜플 리스트로 변환
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    
    return normalize_landmarks(landmarks)


def batch_normalize_landmarks(landmarks_batch: List[List[Tuple[float, float]]]) -> np.ndarray:
    """
    여러 프레임의 랜드마크를 배치로 정규화
    
    Args:
        landmarks_batch: 각 프레임의 21개 랜드마크 리스트
        
    Returns:
        정규화된 특징 배열 (N, 40) - N은 유효한 프레임 수
    """
    features_list = []
    
    for landmarks in landmarks_batch:
        features = normalize_landmarks(landmarks)
        if features is not None:
            features_list.append(features)
    
    if len(features_list) == 0:
        print("[WARNING] 유효한 랜드마크가 하나도 없습니다.")
        return np.array([])
    
    return np.array(features_list)


def get_landmark_names() -> List[str]:
    """
    MediaPipe Hands 21개 랜드마크 이름 반환 (디버깅/시각화용)
    
    Returns:
        랜드마크 이름 리스트
    """
    return [
        "WRIST",           # 0
        "THUMB_CMC",       # 1
        "THUMB_MCP",       # 2
        "THUMB_IP",        # 3
        "THUMB_TIP",       # 4
        "INDEX_FINGER_MCP",# 5
        "INDEX_FINGER_PIP",# 6
        "INDEX_FINGER_DIP",# 7
        "INDEX_FINGER_TIP",# 8
        "MIDDLE_FINGER_MCP",# 9
        "MIDDLE_FINGER_PIP",# 10
        "MIDDLE_FINGER_DIP",# 11
        "MIDDLE_FINGER_TIP",# 12
        "RING_FINGER_MCP", # 13
        "RING_FINGER_PIP", # 14
        "RING_FINGER_DIP", # 15
        "RING_FINGER_TIP", # 16
        "PINKY_MCP",       # 17
        "PINKY_PIP",       # 18
        "PINKY_DIP",       # 19
        "PINKY_TIP",       # 20
    ]


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 50)
    print("Feature Extractor 테스트")
    print("=" * 50)
    
    # 가짜 랜드마크 데이터 (21개 점)
    test_landmarks = [(i * 0.05, i * 0.03) for i in range(21)]
    
    features = normalize_landmarks(test_landmarks)
    if features is not None:
        print(f"✓ 정규화 성공! 특징 벡터 크기: {features.shape}")
        print(f"  특징값 범위: [{features.min():.3f}, {features.max():.3f}]")
    else:
        print("✗ 정규화 실패")
    
    print("\n랜드마크 이름:")
    for i, name in enumerate(get_landmark_names()):
        print(f"  {i:2d}: {name}")