"""
ASL gesture recognition model training script
Reads data/gestures.csv, trains KNN/SVM model, and saves to models/asl_skill_model.pkl
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


# Path settings
DATA_FILE = os.path.join("data", "gestures.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "asl_skill_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")


def setup_model_directory():
    """Create model directory"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"✓ Created '{MODEL_DIR}' directory")


def load_data():
    """
    Load data from CSV file
    
    Returns:
        X: Feature vectors (N, 40)
        y: Labels (N,)
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"[ERROR] Data file not found: {DATA_FILE}")
    
    print(f"Loading data: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    print(f"✓ Data loaded")
    print(f"  Total samples: {len(df)}")
    print(f"  Feature dimension: {len(df.columns) - 1}")
    
    # Print samples per gesture
    print("\nSamples per gesture:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Check number of classes
    num_classes = len(label_counts)
    if num_classes < 2:
        raise ValueError(
            f"\n[ERROR] Training requires at least 2 gestures.\n"
            f"Currently collected gestures: {num_classes} ({list(label_counts.keys())})\n\n"
            f"Solutions:\n"
            f"1. Run collect_data.py to collect additional gesture data.\n"
            f"   Example: A, C, L, S, Idle - at least 2 or more\n"
            f"2. Recommended to collect 20-30 images per gesture.\n"
        )
    
    # Separate features and labels
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values
    
    return X, y


def train_knn(X_train, y_train, X_test, y_test):
    """
    Train K-Nearest Neighbors model
    
    Returns:
        best_model: KNN model trained with optimal hyperparameters
        accuracy: Test accuracy
    """
    print("\n" + "="*50)
    print("Training KNN model...")
    print("="*50)
    
    # Hyperparameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ Best hyperparameters: {grid_search.best_params_}")
    print(f"✓ Cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Test set evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Test accuracy: {accuracy:.4f}")
    
    return best_model, accuracy


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train Support Vector Machine model
    
    Returns:
        best_model: SVM model trained with optimal hyperparameters
        accuracy: Test accuracy
    """
    print("\n" + "="*50)
    print("Training SVM model...")
    print("="*50)
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Added smaller C (stronger regularization)
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 0.01, 0.1, 1]  # Smaller gamma for smoother decision boundary
    }
    
    # Enable probability calculation with probability=True
    # Handle class imbalance with class_weight='balanced'
    # Stricter decision boundary with decision_function_shape='ovr'
    svm = SVC(
        probability=True, 
        class_weight='balanced', 
        random_state=42,
        # decision_function_shape='ovr',  # One-vs-Rest approach
        cache_size=500  # Increase memory cache for speed
    )
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ Best hyperparameters: {grid_search.best_params_}")
    print(f"✓ Cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Test set evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Test accuracy: {accuracy:.4f}")
    
    return best_model, accuracy


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model
    
    Returns:
        best_model: RF model trained with optimal hyperparameters
        accuracy: Test accuracy
    """
    print("\n" + "="*50)
    print("Training Random Forest model...")
    print("="*50)
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ Best hyperparameters: {grid_search.best_params_}")
    print(f"✓ Cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Test set evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Test accuracy: {accuracy:.4f}")
    
    return best_model, accuracy


def evaluate_model(model, X_test, y_test):
    """
    Detailed model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("Model evaluation results")
    print("="*50)
    
    # Classification report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Accuracy per gesture
    print("\nAccuracy per gesture:")
    labels = np.unique(y_test)
    for i, label in enumerate(labels):
        mask = y_test == label
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {label}: {acc:.4f} ({mask.sum()} samples)")


def save_model(model, scaler=None):
    """
    Save model and scaler to files
    
    Args:
        model: Trained model
        scaler: StandardScaler object (optional)
    """
    joblib.dump(model, MODEL_FILE)
    print(f"\n✓ Model saved: {MODEL_FILE}")
    
    if scaler is not None:
        joblib.dump(scaler, SCALER_FILE)
        print(f"✓ Scaler saved: {SCALER_FILE}")


def main():
    """Main execution function"""
    print("\n" + "="*50)
    print("ASL Gesture Recognition Model Training System")
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
    
    # Data splitting
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Feature scaling (optional)
    use_scaling = input("\nUse feature scaling? (y/n, default: n): ").strip().lower()
    scaler = None
    
    if use_scaling == 'y':
        print("Scaling features with StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✓ Scaling completed")
    
    # Model selection
    print("\n" + "="*50)
    print("Select model to train")
    print("="*50)
    print("\n1. KNN (K-Nearest Neighbors)")
    print("   ├ Features: Fastest training speed")
    print("   ├ Pros: Simple and intuitive implementation")
    print("   ├ Cons: Inaccurate probability calculation, hard to detect Unknown")
    print("   └ Recommended: ❌ Not recommended for real-time gesture recognition\n")
    
    print("2. SVM (Support Vector Machine) ⭐ Recommended")
    print("   ├ Features: Powerful for high-dimensional data")
    print("   ├ Pros: Accurate probability calculation, prevents overfitting")
    print("   ├ Cons: Training takes some time")
    print("   └ Recommended: ✅ Best performance with small datasets\n")
    
    print("3. Random Forest ⭐ Recommended")
    print("   ├ Features: Ensemble of multiple decision trees")
    print("   ├ Pros: Stable probabilities, robust to noise")
    print("   ├ Cons: Large model file size")
    print("   └ Recommended: ✅ Stable with large datasets\n")
    
    print("4. Compare all and auto-select best model")
    print("   └ Train all 3 models and save the one with highest accuracy\n")
    
    choice = input("Choice (1-4, default: 2): ").strip() or "2"
    
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
    
    # Select best model
    if len(models) > 1:
        print("\n" + "="*50)
        print("Model comparison results")
        print("="*50)
        for name, (model, acc) in models.items():
            print(f"{name}: {acc:.4f}")
        
        best_model_name = max(models, key=lambda k: models[k][1])
        best_model = models[best_model_name][0]
        print(f"\n✓ Best model: {best_model_name}")
    else:
        best_model_name = list(models.keys())[0]
        best_model = models[best_model_name][0]
    
    # 상세 평가
    evaluate_model(best_model, X_test, y_test)
    
    # 모델 저장
    save_model(best_model, scaler)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Selected model: {best_model_name}")
    print(f"Model file: {MODEL_FILE}")
    print("\n💡 Tips:")
    print("  - If many Unknown gestures are recognized, try lowering")
    print("    CONFIDENCE_THRESHOLD in test/test_gesture_recognition.py (0.6 → 0.5)")
    print("  - If wrong gestures are recognized, collect more data or")
    print("    try different models (SVM ↔ Random Forest)")
    print("\nYou can now test the model with test/test_gesture_recognition.py.")


if __name__ == "__main__":
    main()
