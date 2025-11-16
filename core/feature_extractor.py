"""
Feature extraction module for ASL gesture recognition
Normalizes MediaPipe Hands landmarks to create position/size invariant features
"""

import numpy as np
from typing import List, Tuple, Optional


def normalize_landmarks(landmarks: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    """
    Normalizes MediaPipe hand landmarks to create position/size invariant feature vectors
    
    Normalization process:
    1. Position normalization: Move wrist (landmark 0) to origin (0,0)
    2. Size normalization: Scale entire hand using distance from wrist (0) to middle finger base (9)
    
    Args:
        landmarks: List of (x, y) coordinates for 21 landmarks
        
    Returns:
        Normalized 40 feature values (x, y coordinates of 20 points) or None (on error)
    """
    if len(landmarks) != 21:
        print(f"[WARNING] Number of landmarks is not 21: {len(landmarks)}")
        return None
    
    # NumPy 배열로 변환 (21, 2)
    points = np.array(landmarks, dtype=np.float32)
    
    # 1. 위치 정규화: 손목(0번)을 원점으로
    wrist = points[0].copy()
    points_centered = points - wrist
    
    # 2. 크기 정규화: 손목(0번) ~ 중지 바닥(9번) 거리로 스케일링
    middle_mcp = points_centered[9]  # 중지 바닥 (MCP: Metacarpophalangeal joint)
    scale = np.linalg.norm(middle_mcp)
    
    # Scale is too small (hand too far or detection error) return None
    if scale < 1e-6:
        print("[WARNING] Hand size is too small. May be a landmark detection error.")
        return None
    
    points_normalized = points_centered / scale
    
    # 3. 손목 제외한 20개 점만 사용 (손목은 항상 (0,0)이므로 정보 없음)
    features = points_normalized[1:].flatten()  # (20, 2) -> (40,)
    
    return features


def extract_features_from_mediapipe(hand_landmarks) -> Optional[np.ndarray]:
    """
    Extract features directly from MediaPipe Hands object
    
    Args:
        hand_landmarks: mediapipe.python.solutions.hands.HandLandmark object
        
    Returns:
        Normalized 40 feature values or None
    """
    if hand_landmarks is None:
        return None
    
    # MediaPipe 랜드마크를 (x, y) 튜플 리스트로 변환
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    
    return normalize_landmarks(landmarks)


def batch_normalize_landmarks(landmarks_batch: List[List[Tuple[float, float]]]) -> np.ndarray:
    """
    Batch normalize landmarks from multiple frames
    
    Args:
        landmarks_batch: List of 21 landmarks for each frame
        
    Returns:
        Normalized feature array (N, 40) - N is number of valid frames
    """
    features_list = []
    
    for landmarks in landmarks_batch:
        features = normalize_landmarks(landmarks)
        if features is not None:
            features_list.append(features)
    
    if len(features_list) == 0:
        print("[WARNING] No valid landmarks found.")
        return np.array([])
    
    return np.array(features_list)


def get_landmark_names() -> List[str]:
    """
    Return names of 21 MediaPipe Hands landmarks (for debugging/visualization)
    
    Returns:
        List of landmark names
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
    # Test code
    print("=" * 50)
    print("Feature Extractor Test")
    print("=" * 50)
    
    # Fake landmark data (21 points)
    test_landmarks = [(i * 0.05, i * 0.03) for i in range(21)]
    
    features = normalize_landmarks(test_landmarks)
    if features is not None:
        print(f"✓ Normalization successful! Feature vector size: {features.shape}")
        print(f"  Feature value range: [{features.min():.3f}, {features.max():.3f}]")
    else:
        print("✗ Normalization failed")
    
    print("\nLandmark names:")
    for i, name in enumerate(get_landmark_names()):
        print(f"  {i:2d}: {name}")