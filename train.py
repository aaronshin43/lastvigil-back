"""
ASL 제스처 인식 모델 훈련 스크립트
data/gestures.csv를 읽어 KNN/SVM 모델을 훈련하고 models/asl_skill_model.pkl로 저장
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# 경로 설정
DATA_FILE = os.path.join("data", "gestures.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "asl_skill_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")


def setup_model_directory():
    """모델 디렉토리 생성"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"✓ '{MODEL_DIR}' 디렉토리 생성 완료")


def load_data():
    """
    CSV 파일에서 데이터 로드
    
    Returns:
        X: 특징 벡터 (N, 40)
        y: 라벨 (N,)
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"[ERROR] 데이터 파일이 없습니다: {DATA_FILE}")
    
    print(f"데이터 로드 중: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    print(f"✓ 데이터 로드 완료")
    print(f"  총 샘플 수: {len(df)}")
    print(f"  특징 차원: {len(df.columns) - 1}")
    
    # 제스처별 샘플 수 출력
    print("\n제스처별 샘플 수:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # 클래스 개수 확인
    num_classes = len(label_counts)
    if num_classes < 2:
        raise ValueError(
            f"\n[ERROR] 훈련에는 최소 2개 이상의 제스처가 필요합니다.\n"
            f"현재 수집된 제스처: {num_classes}개 ({list(label_counts.keys())})\n\n"
            f"해결 방법:\n"
            f"1. collect_data.py를 실행하여 추가 제스처 데이터를 수집하세요.\n"
            f"   예: A, C, L, S, Idle 등 최소 2개 이상\n"
            f"2. 각 제스처당 20~30장 이상 수집을 권장합니다.\n"
        )
    
    # 특징과 라벨 분리
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values
    
    return X, y


def train_knn(X_train, y_train, X_test, y_test):
    """
    K-Nearest Neighbors 모델 훈련
    
    Returns:
        best_model: 최적 하이퍼파라미터로 훈련된 KNN 모델
        accuracy: 테스트 정확도
    """
    print("\n" + "="*50)
    print("KNN 모델 훈련 중...")
    print("="*50)
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ 최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"✓ 교차 검증 정확도: {grid_search.best_score_:.4f}")
    
    # 테스트 세트 평가
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ 테스트 정확도: {accuracy:.4f}")
    
    return best_model, accuracy


def train_svm(X_train, y_train, X_test, y_test):
    """
    Support Vector Machine 모델 훈련
    
    Returns:
        best_model: 최적 하이퍼파라미터로 훈련된 SVM 모델
        accuracy: 테스트 정확도
    """
    print("\n" + "="*50)
    print("SVM 모델 훈련 중...")
    print("="*50)
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'C': [0.1, 1, 10, 100],  # 더 작은 C 추가 (정규화 강화)
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 0.01, 0.1, 1]  # 더 작은 gamma로 결정 경계 부드럽게
    }
    
    # probability=True로 확률 계산 활성화
    # class_weight='balanced'로 클래스 불균형 처리
    # decision_function_shape='ovr'로 더 엄격한 결정 경계
    svm = SVC(
        probability=True, 
        class_weight='balanced', 
        random_state=42,
        # decision_function_shape='ovr',  # One-vs-Rest 방식
        cache_size=500  # 메모리 캐시 증가로 속도 향상
    )
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ 최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"✓ 교차 검증 정확도: {grid_search.best_score_:.4f}")
    
    # 테스트 세트 평가
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ 테스트 정확도: {accuracy:.4f}")
    
    return best_model, accuracy


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Random Forest 모델 훈련
    
    Returns:
        best_model: 최적 하이퍼파라미터로 훈련된 RF 모델
        accuracy: 테스트 정확도
    """
    print("\n" + "="*50)
    print("Random Forest 모델 훈련 중...")
    print("="*50)
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ 최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"✓ 교차 검증 정확도: {grid_search.best_score_:.4f}")
    
    # 테스트 세트 평가
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ 테스트 정확도: {accuracy:.4f}")
    
    return best_model, accuracy


def evaluate_model(model, X_test, y_test):
    """
    모델 상세 평가
    
    Args:
        model: 훈련된 모델
        X_test: 테스트 특징
        y_test: 테스트 라벨
    """
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("모델 평가 결과")
    print("="*50)
    
    # 분류 리포트
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred))
    
    # 혼동 행렬
    print("혼동 행렬:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 제스처별 정확도
    print("\n제스처별 정확도:")
    labels = np.unique(y_test)
    for i, label in enumerate(labels):
        mask = y_test == label
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {label}: {acc:.4f} ({mask.sum()} 샘플)")


def save_model(model, scaler=None):
    """
    모델과 스케일러를 파일로 저장
    
    Args:
        model: 훈련된 모델
        scaler: StandardScaler 객체 (선택)
    """
    joblib.dump(model, MODEL_FILE)
    print(f"\n✓ 모델 저장 완료: {MODEL_FILE}")
    
    if scaler is not None:
        joblib.dump(scaler, SCALER_FILE)
        print(f"✓ 스케일러 저장 완료: {SCALER_FILE}")


def main():
    """메인 실행 함수"""
    print("\n" + "="*50)
    print("ASL 제스처 인식 모델 훈련 시스템")
    print("="*50 + "\n")
    
    setup_model_directory()
    
    # 데이터 로드
    try:
        X, y = load_data()
    except ValueError as e:
        print(str(e))
        return
    except Exception as e:
        print(f"[ERROR] 데이터 로드 중 오류 발생: {e}")
        return
    
    # 데이터 분할
    print("\n데이터 분할 중 (80% 훈련, 20% 테스트)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ 훈련 세트: {len(X_train)} 샘플")
    print(f"✓ 테스트 세트: {len(X_test)} 샘플")
    
    # 특징 스케일링 (옵션)
    use_scaling = input("\n특징 스케일링을 사용하시겠습니까? (y/n, 기본값: n): ").strip().lower()
    scaler = None
    
    if use_scaling == 'y':
        print("StandardScaler로 특징 스케일링 중...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✓ 스케일링 완료")
    
    # 모델 선택
    print("\n" + "="*50)
    print("훈련할 모델을 선택하세요")
    print("="*50)
    print("\n1. KNN (K-Nearest Neighbors)")
    print("   ├ 특징: 가장 빠른 훈련 속도")
    print("   ├ 장점: 구현이 단순하고 직관적")
    print("   ├ 단점: 확률 계산 부정확, Unknown 판별 어려움")
    print("   └ 추천: ❌ 실시간 제스처 인식에는 비추천\n")
    
    print("2. SVM (Support Vector Machine) ⭐ 추천")
    print("   ├ 특징: 고차원 데이터에 강력")
    print("   ├ 장점: 정확한 확률 계산, 과적합 방지")
    print("   ├ 단점: 훈련 시간이 다소 소요됨")
    print("   └ 추천: ✅ 데이터 양이 적을 때 최고 성능\n")
    
    print("3. Random Forest ⭐ 추천")
    print("   ├ 특징: 여러 결정 트리의 앙상블")
    print("   ├ 장점: 안정적인 확률, 노이즈에 강함")
    print("   ├ 단점: 모델 파일 크기가 큼")
    print("   └ 추천: ✅ 데이터 양이 많을 때 안정적\n")
    
    print("4. 모두 비교 후 최적 모델 자동 선택")
    print("   └ 3가지 모델을 모두 훈련하여 가장 높은 정확도의 모델 저장\n")
    
    choice = input("선택 (1-4, 기본값: 2): ").strip() or "2"
    
    models = {}
    
    if choice == "1" or choice == "4":
        model, acc = train_knn(X_train, y_train, X_test, y_test)
        models['KNN'] = (model, acc)
    
    if choice == "2" or choice == "4":
        model, acc = train_svm(X_train, y_train, X_test, y_test)
        models['SVM'] = (model, acc)
    
    if choice == "3" or choice == "4":
        model, acc = train_random_forest(X_train, y_train, X_test, y_test)
        models['Random Forest'] = (model, acc)
    
    # 최적 모델 선택
    if len(models) > 1:
        print("\n" + "="*50)
        print("모델 비교 결과")
        print("="*50)
        for name, (model, acc) in models.items():
            print(f"{name}: {acc:.4f}")
        
        best_model_name = max(models, key=lambda k: models[k][1])
        best_model = models[best_model_name][0]
        print(f"\n✓ 최적 모델: {best_model_name}")
    else:
        best_model_name = list(models.keys())[0]
        best_model = models[best_model_name][0]
    
    # 상세 평가
    evaluate_model(best_model, X_test, y_test)
    
    # 모델 저장
    save_model(best_model, scaler)
    
    print("\n" + "="*50)
    print("훈련 완료!")
    print("="*50)
    print(f"선택된 모델: {best_model_name}")
    print(f"모델 파일: {MODEL_FILE}")
    print("\n💡 팁:")
    print("  - Unknown 제스처가 많이 인식되면 test/test_gesture_recognition.py의")
    print("    CONFIDENCE_THRESHOLD 값을 낮춰보세요 (0.6 → 0.5)")
    print("  - 잘못된 제스처가 인식되면 더 많은 데이터를 수집하거나")
    print("    다른 모델을 시도해보세요 (SVM ↔ Random Forest)")
    print("\n이제 test/test_gesture_recognition.py로 모델을 테스트할 수 있습니다.")


if __name__ == "__main__":
    main()
