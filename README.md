# Last Vigil Backend

A real-time ASL gesture recognition game backend using FastAPI WebSocket, MediaPipe, and machine learning models. The system processes webcam input to detect ASL alphabet gestures and gaze direction for controlling a 2D defense game.

## Features

- **Real-time ASL Gesture Recognition**: Uses MediaPipe hand landmarks and scikit-learn ML models to recognize ASL alphabet gestures
- **Gaze Tracking**: Calculates gaze direction from face landmarks for targeting in the game
- **Session-based Game Logic**: Independent game sessions with enemy spawning, collision detection, and wave progression
- **WebSocket Communication**: Real-time bidirectional communication between client and server
- **Feature Extraction**: Normalizes MediaPipe landmarks to position/size-invariant 40D feature vectors
- **Model Training**: Supports training KNN, SVM, and Random Forest models on gesture data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaronshin43/lastvigil-back.git
cd lastvigil-back
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download or train the ASL gesture recognition model:
   - Place the trained model file as `models/asl_skill_model.pkl`
   - Or run the training script: `python train.py`

## Usage

### Running the Server

Start the FastAPI WebSocket server:
```bash
uvicorn main:app
```

The server will run on `http://localhost:8000`

### Data Collection

Collect gesture data for training:
```bash
python collect_data.py
```

### Training Models

Train ML models on collected data:
```bash
python train.py
```

### Testing Gesture Recognition

Test real-time gesture recognition:
```bash
python test_gesture_recognition.py
```

### Extracting Landmarks from Images

Process image datasets for landmark extraction:
```bash
python extract_landmarks_from_images.py
```

## Architecture

### Core Components

- **Feature Extractor** (`core/feature_extractor.py`): Normalizes MediaPipe hand landmarks to create position-invariant features
- **Game Logic** (`main.py`): Handles session management, enemy spawning, collision detection, and WebSocket communication
- **AI Processing**: Real-time analysis of webcam frames for gesture and gaze detection
- **Model Training** (`train.py`): Trains and evaluates ML models using cross-validation

### WebSocket Endpoints

- `/ws`: Main game WebSocket endpoint with session ID parameter for real-time game interaction

### Data Flow

1. Client sends Base64-encoded webcam frames via WebSocket
2. Server processes frames with MediaPipe for hand/face detection
3. ASL gesture recognition using trained ML model
4. Gaze calculation from face landmarks
5. Game logic updates based on AI input
6. Server sends full state sync to client

## Project Structure

```
lastvigil-back/
├── main.py                          # FastAPI WebSocket server and game logic
├── core/
│   └── feature_extractor.py         # Landmark normalization and feature extraction
├── models/
│   └── asl_skill_model.pkl          # Trained ML model (not included)
├── data/
│   └── gestures.csv                 # Training data
├── test/
│   └── test_gesture_recognition.py  # Real-time testing script
├── collect_data.py                  # Data collection script
├── extract_landmarks_from_images.py # Image processing script
├── train.py                         # Model training script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Dependencies

- fastapi: Web framework for WebSocket server
- mediapipe: Hand and face landmark detection
- opencv-python: Image processing
- scikit-learn: Machine learning models
- numpy: Numerical computations
- pandas: Data manipulation
- joblib: Model serialization
- uvicorn: ASGI server

## Game Mechanics

- **Gesture Sequence**: Players must match ASL gestures in sequence to cast skills
- **Gaze Targeting**: Eye gaze determines skill targeting direction
- **Wave System**: Progressive difficulty with enemy HP/speed increases
- **Real-time Processing**: 20fps game loop with AI input processing
- **Session Management**: Independent game sessions for multiple players