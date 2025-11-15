/lastvigil-back/ 
|
|-- 📂 venv/
|   `-- ...
|
|-- 📂 core/         (핵심 AI 로직 및 유틸리티 함수 보관)
|   |-- __init__.py      
|   `-- feature_extractor.py (손 랜드마크 정규화, 특징 추출 등 순수 계산 함수)
|
|-- 📂 data/             (훈련에 사용할 원시 데이터)
|   |-- gestures.csv       (수집된 랜드마크 특징(X,Y)과 라벨(A,B,C)이 저장된 파일)
|   `-- .gitignore       (Git 사용 시, .csv 파일을 업로드에서 제외)
|
|-- 📂 models/           (훈련이 완료된 머신러닝 모델 파일)
|   |-- asl_skill_model.pkl  (scikit-learn의 'joblib'으로 저장된 KNN/SVM 모델)
|   `-- .gitignore       (Git 사용 시, .pkl 파일을 업로드에서 제외)
|
|-- 📜 collect_data.py    (1. [실행] 데이터 수집 스크립트)
|   |                      (웹캠을 켜고, 랜드마크를 추출/정규화해서 data/gestures.csv에 저장)
|
|-- 📜 train.py           (2. [실행] 모델 훈련 스크립트)
|   |                      (data/gestures.csv를 읽어, sklearn 모델을 훈련시키고 models/asl_skill_model.pkl로 저장)
|
|-- 📜 main.py            (3. [실행] 메인 FastAPI 서버 스크립트)
|   |                      (models/asl_skill_model.pkl을 로드하고, /ws 엔드포인트에서 실시간 예측 제공)
|
|-- 📜 requirements.txt   (프로젝트 의존성 목록)
|   |                      (fastapi, uvicorn, scikit-learn, mediapipe, numpy, pandas...)
|
`-- 📜 .gitignore         (Git 버전 관리에서 제외할 파일 목록 - venv/, data/, models/ 등)