import uvicorn
import cv2
import numpy as np
import base64
import mediapipe as mp
import joblib
import asyncio
import time
import uuid
import json
from pathlib import Path
from typing import List, Dict, Set
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware

# ASL gesture recognition feature extraction module
from core.feature_extractor import extract_features_from_mediapipe

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

app = FastAPI()
# test
# CORS settings (allow all origins - for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MediaPipe module initialization ---
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5
)
# Enhanced Face Mesh settings
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5 # Added tracking sensitivity
)

# --- ASL gesture recognition model loading ---
MODEL_PATH = Path(__file__).parent / "models" / "asl_skill_model.pkl"
try:
    asl_model = joblib.load(MODEL_PATH)
    print(f"✓ ASL gesture recognition model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    asl_model = None

# --- Session-based storage (independent game state for each user) ---
game_sessions: Dict[str, dict] = {}  # {session_id: gameState}
ai_sessions: Dict[str, dict] = {}     # {session_id: latestAIInput}
session_tasks: Dict[str, asyncio.Task] = {}  # {session_id: game_loop_task}

# --- Skill system: available alphabets ---
AVAILABLE_GESTURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# --- Skill type settings (for random selection) ---
SKILL_TYPES = [
    {"type": "fireSlash", "damage": 50, "range": 200},
    {"type": "skyBeam", "damage": 80, "range": 200},
    {"type": "fireVortexRed", "damage": 60, "range": 200},
    {"type": "fireHammerRed", "damage": 100, "range": 200},
    {"type": "lightningV1", "damage": 70, "range": 200},
    {"type": "lightningV2", "damage": 75, "range": 200},
    {"type": "meteorShowerRed", "damage": 90, "range": 200},
    {"type": "fireHurricaneBlue", "damage": 85, "range": 200},
    {"type": "tornado", "damage": 65, "range": 200},
]


# ==================== Game Logic Classes ====================

# Enemy info (HP, speed, score)
ENEMY_CONFIG = {
    # Tier 1: Basic monsters (10 points)
    "slime": {"hp": 50, "speed": 45, "score": 10},
    "skeleton": {"hp": 80, "speed": 50, "score": 10},
    "orc": {"hp": 100, "speed": 45, "score": 10},
    # Tier 2: Intermediate monsters (15 points)
    "skeletonArcher": {"hp": 120, "speed": 55, "score": 15},
    "armoredSkeleton": {"hp": 150, "speed": 40, "score": 15},
    "greatswordSkeleton": {"hp": 180, "speed": 35, "score": 15},
    # Tier 3: Advanced monsters (20 points)
    "armoredOrc": {"hp": 250, "speed": 50, "score": 20},
    "elitOrc": {"hp": 300, "speed": 55, "score": 20},
    "orcRider": {"hp": 350, "speed": 60, "score": 20},
}

# Enemy tiers by wave
WAVE_ENEMY_TIERS = {
    1: ["slime", "skeleton"],
    2: ["slime", "skeleton", "orc"],
    3: ["skeleton", "orc", "skeletonArcher"],
    4: ["orc", "skeletonArcher", "armoredSkeleton"],
    5: ["skeletonArcher", "armoredSkeleton", "greatswordSkeleton", "armoredOrc", "elitOrc", "orcRider"],
}

# Target scores by wave (advance to next wave when reached)
WAVE_SCORE_THRESHOLDS = {
    1: 80,   # Reach 100 points for wave 2
    2: 170,   # Reach 250 points for wave 3
    3: 270,   # Reach 450 points for wave 4
    4: 380,   # Reach 700 points for wave 5 (final)
    5: float('inf')  # Wave 5 is infinite
}

class Enemy:
    """Enemy entity"""
    def __init__(self, enemy_id: str, type_id: str, x: float, y: float, hp_bonus: int = 0, speed_multiplier: float = 1.0):
        self.id = enemy_id
        self.typeId = type_id
        self.x = x
        self.y = y
        
        # Set HP and speed based on enemy type
        config = ENEMY_CONFIG.get(type_id, {"hp": 100, "speed": 50, "score": 10})
        self.maxHP = config["hp"] + hp_bonus  # HP bonus added
        self.currentHP = self.maxHP
        self.speed = config["speed"] * speed_multiplier  # Speed multiplier applied
        self.score = config["score"]  # Score gained on death
        
        self.animationState = "walk"
        self.currentFrame = 0
        self.isDead = False
        self.scoreGiven = False  # Score given flag
        self.direction = -1 if x > 500 else 1  # Move left if spawned on right
        self.hurtFrameCount = 0  # Hurt animation duration frames
    
    def update(self, delta_time: float):
        """Update enemy movement"""
        if not self.isDead:
            self.x += self.direction * self.speed * delta_time
            self.currentFrame = (self.currentFrame + 1) % 8
            
            # Hurt state lasts only 8 frames
            if self.animationState == "hurt":
                self.hurtFrameCount += 1
                if self.hurtFrameCount >= 8:
                    self.animationState = "walk"
                    self.hurtFrameCount = 0
    
    def take_damage(self, damage: int):
        """Take damage"""
        self.currentHP -= damage
        if self.currentHP <= 0:
            self.currentHP = 0
            self.isDead = True
            self.animationState = "death"
        else:
            self.animationState = "hurt"
            self.hurtFrameCount = 0  # Reset hurt counter
    
    def to_dict(self) -> dict:
        """JSON serialization (normalized coordinates)"""
        return {
            "id": self.id,
            "typeId": self.typeId,
            "x": self.x / 2148,  # 0.0~1.0 normalized
            "y": self.y / 1080,  # 0.0~1.0 normalized
            "currentHP": self.currentHP,
            "maxHP": self.maxHP,
            "animationState": self.animationState,
            "currentFrame": self.currentFrame,
            "isDead": self.isDead
        }


class Effect:
    """Skill effect"""
    def __init__(self, effect_id: str, effect_type: str, x: float, damage: int, skill_range: int = 200, duration: float = 0.5):
        self.id = effect_id
        self.type = effect_type
        self.x = x
        self.damage = damage
        self.range = skill_range
        self.duration = duration
        self.createdAt = time.time()
    
    def is_expired(self) -> bool:
        return time.time() - self.createdAt > self.duration
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "x": self.x / 2148,  # 0.0~1.0 normalized
            "damage": self.damage
        }


class Player:
    """Player (no longer used - skills handled by frontend)"""
    pass


player = Player()


def generate_gesture_sequence(count: int = 5) -> List[str]:
    """Generate random alphabet sequence (5 letters)"""
    return [np.random.choice(AVAILABLE_GESTURES) for _ in range(count)]


# ==================== Game Logic Functions ====================

def spawn_enemy(session_id: str):
    """Spawn enemies based on wave"""
    enemy_id = f"enemy_{uuid.uuid4().hex[:8]}"
    gameState = game_sessions[session_id]
    current_wave = gameState["waveNumber"]
    
    # Select enemies by wave
    enemy_pool = WAVE_ENEMY_TIERS.get(current_wave, ["skeleton", "orc"])
    type_id = np.random.choice(enemy_pool)
    
    # HP increase in wave 5 (proportional to number of enemies spawned)
    hp_bonus = 0
    if current_wave == 5:
        hp_bonus = gameState.get("wave5EnemyCount", 0) * 5  # +5 HP per enemy spawned
        gameState["wave5EnemyCount"] = gameState.get("wave5EnemyCount", 0) + 1
    
    # Speed increase by wave (10% per wave)
    speed_multiplier = 1.0 + (current_wave - 1) * 0.1  # Wave 1: 1.0x, 2: 1.1x, 3: 1.2x, 4: 1.3x, 5: 1.4x
    
    # Spawn on right (map width 2148 pixels)
    x = 2000
    # Y-axis: bottom 50~70% (540~756 based on 1080)
    y = 648 + np.random.randint(0, 217)  # 540 + [0~216]
    
    enemy = Enemy(enemy_id, type_id, x, y, hp_bonus, speed_multiplier)
    gameState["enemies"].append(enemy)
    
    config = ENEMY_CONFIG.get(type_id, {})
    # print(f"[Game] Enemy spawned: {enemy_id} ({type_id}) HP:{config.get('hp', 100)} at ({x}, {y})")


def check_collision(session_id: str, effect: Effect) -> List[Enemy]:
    """Skill-enemy collision detection (x-axis only, pixel-based)"""
    hit_enemies = []
    target_x = effect.x  # Map pixels (0~2148)
    skill_range = effect.range  # Pixel units
    gameState = game_sessions[session_id]
    
    for enemy in gameState["enemies"]:
        if enemy.isDead:
            continue
        
        # Calculate x-axis pixel distance
        distance = abs(enemy.x - target_x)
        
        if distance <= skill_range:
            hit_enemies.append(enemy)
    
    return hit_enemies


# ==================== Game Loop (20fps) ====================

async def game_loop(websocket: WebSocket, session_id: str):
    """
    Session-based game loop
    - 20fps (execute every 0.05 seconds)
    - Read AI input from session and process game logic
    - Broadcast only to the client of that session
    """
    print(f"[Game] Game loop started (session: {session_id[:8]}...)")
    
    gameState = game_sessions[session_id]
    latestAIInput = ai_sessions[session_id]
    
    last_spawn_time = time.time()
    spawn_interval = 3.0  # Spawn enemies every 3 seconds
    
    try:
        while True:
            loop_start = time.time()
            delta_time = 0.05  # 20fps
            
            # 0. Wave progress check (score-based)
            current_wave = gameState["waveNumber"]
            current_score = gameState["playerScore"]
            threshold = WAVE_SCORE_THRESHOLDS.get(current_wave, float('inf'))
            
            if current_wave < 5 and current_score >= threshold:
                gameState["waveNumber"] += 1
                print(f"[Game] Wave {gameState['waveNumber']} started! (session: {session_id[:8]}..., score: {current_score})")
            
            # 1. Enemy spawner
            if time.time() - last_spawn_time > spawn_interval:
                spawn_enemy(session_id)
                last_spawn_time = time.time()
            
            # 2. Enemy updates (movement and screen exit check)
            for enemy in gameState["enemies"]:
                enemy.update(delta_time)
                
                # Reduce player HP if enemy reaches player position
                if enemy.x <= 530 and not enemy.isDead:
                    gameState["playerHP"] -= 10
                    enemy.isDead = True  # Remove enemy that reached
                    enemy.scoreGiven = True  # Prevent score gain (no score for reaching enemies)
                    print(f"[Game] Enemy passed! HP: {gameState['playerHP']} (session: {session_id[:8]}...)")
            
            # 3. Remove dead enemies and gain score (only once)
            for enemy in gameState["enemies"]:
                if enemy.isDead and not enemy.scoreGiven:
                    # Gain score by tier on enemy death (Tier 1: 10, Tier 2: 15, Tier 3: 20)
                    gameState["playerScore"] += enemy.score
                    enemy.scoreGiven = True
            
            gameState["enemies"] = [e for e in gameState["enemies"] if not e.isDead or time.time() - gameState["lastSkillTime"] < 1.0]
            
            # 4. Game over check
            if gameState["playerHP"] <= 0:
                print(f"[Game] Game over! (session: {session_id[:8]}..., final score: {gameState['playerScore']})")
                
                # Send game over signal
                game_over_message = {
                    "type": "gameOver",
                    "finalScore": gameState["playerScore"],
                    "finalWave": gameState["waveNumber"]
                }
                try:
                    await websocket.send_json(game_over_message)
                except:
                    pass
                break
            
            # 5. AI input check and gesture matching check
            gesture = latestAIInput.get("gesture", "NONE")
            gesture_matched = False
            current_time = time.time()
            gesture_cooldown = 0.5  # Gesture recognition cooldown (0.5 seconds)
            
            if gesture != "NONE" and len(gameState["gestureSequence"]) > 0:
                # Cooldown check
                if current_time - gameState["lastGestureTime"] < gesture_cooldown:
                    pass  # Ignore if in cooldown
                # Compare with first letter of sequence
                elif gesture == gameState["gestureSequence"][0]:
                    gesture_matched = True
                    gameState["lastGestureTime"] = current_time  # Start cooldown
                    
                    # Remove from sequence
                    gameState["gestureSequence"].pop(0)
                    # Add new letter (maintain 5 letters)
                    gameState["gestureSequence"].append(np.random.choice(AVAILABLE_GESTURES))
                    
                    # Select random skill
                    skill = np.random.choice(SKILL_TYPES)
                    
                    # Create effect at gaze position
                    effect_id = f"effect_{uuid.uuid4().hex[:8]}"
                    effect = Effect(
                        effect_id,
                        skill["type"],
                        latestAIInput["gaze_x"],  # Map pixel coordinates
                        skill["damage"],
                        skill["range"]
                    )
                    gameState["effects"].append(effect)
                    
                    # Collision detection and damage processing
                    hit_enemies = check_collision(session_id, effect)
                    for enemy in hit_enemies:
                        enemy.take_damage(skill["damage"])
                    
                    # print(f"[Game] Gesture matched! {gesture} | Skill: {skill['type']} (Damage: {skill['damage']}) | New sequence: {gameState['gestureSequence']} (session: {session_id[:8]}...)")
            # Store matching success
            latestAIInput["gestureMatched"] = gesture_matched
            
            # 6. Remove expired effects
            gameState["effects"] = [e for e in gameState["effects"] if not e.is_expired()]
            
            # 7. Generate Full State Sync (normalized coordinates)
            gaze_x_norm = latestAIInput["gaze_x"] / 2148
            gaze_y_norm = min(latestAIInput["gaze_y"] / 1080 - 0.15, 0)
            
            state_sync = {
                "gameState": {
                    "enemies": [e.to_dict() for e in gameState["enemies"]],
                    "effects": [e.to_dict() for e in gameState["effects"]],
                    "gazePosition": {
                        "x": gaze_x_norm,  # 0.0~1.0 normalized
                        "y": gaze_y_norm   # 0.0~1.0 normalized
                    },
                    "playerGold": gameState["playerGold"],
                    "playerScore": gameState["playerScore"],
                    "playerHP": gameState["playerHP"],
                    "waveNumber": gameState["waveNumber"],
                    "gestureSequence": gameState["gestureSequence"],  # Alphabet sequence
                    "gestureMatched": latestAIInput.get("gestureMatched", False)  # Matching success
                }
            }
            
            # 8. Send only to the session's client
            try:
                await websocket.send_json(state_sync)
            except:
                print(f"[Game] Session {session_id[:8]}... connection lost")
                break
            
            # 9. Maintain 20fps
            elapsed = time.time() - loop_start
            sleep_time = max(0, delta_time - elapsed)
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        print(f"[Game] Session {session_id[:8]}... game loop terminated")


def calculate_gaze(face_key_points: dict) -> dict:
    """
    Calculate gaze direction referring to frontend logic
    
    Returns:
        dict: {
            "gaze_x": map pixel (0~2148),
            "gaze_y": map pixel (0~1080),
            "yaw_ratio": -1.0~1.0,
            "pitch_ratio": -1.0~1.0
        }
    """
    data = face_key_points
    
    # Default value (map center)
    result = {
        "gaze_x": 1074.0,  # 2148 / 2
        "gaze_y": 540.0,   # 1080 / 2
        "yaw_ratio": 0.0,
        "pitch_ratio": 0.0
    }
    
    # Check if all required points are present
    required_points = ["nose_tip", "chin", "forehead", "left_face", "right_face", "left_eye", "right_eye"]
    if not all(data.get(point) for point in required_points):
        return result
    
    nose_tip = data["nose_tip"]
    chin = data["chin"]
    forehead = data["forehead"]
    left_face = data["left_face"]
    right_face = data["right_face"]
    left_eye = data["left_eye"]
    right_eye = data["right_eye"]
    
    # Calculate face center point
    face_center_x = (left_eye["x"] + right_eye["x"]) / 2
    face_center_y = (left_eye["y"] + right_eye["y"]) / 2
    
    # Calculate Yaw (left-right rotation)
    left_distance = abs(nose_tip["x"] - left_face["x"])
    right_distance = abs(nose_tip["x"] - right_face["x"])
    face_width = abs(right_face["x"] - left_face["x"])
    
    yaw_ratio = 0.0
    if face_width > 0:
        yaw_ratio = (left_distance - right_distance) / face_width
    
    # Calculate Pitch (up-down rotation)
    nose_to_forehead = abs(nose_tip["y"] - forehead["y"])
    nose_to_chin = abs(nose_tip["y"] - chin["y"])
    face_height = abs(chin["y"] - forehead["y"])
    
    pitch_ratio = 0.0
    if face_height > 0:
        pitch_ratio = (nose_to_chin - nose_to_forehead) / face_height + 0.15
    
    # Map gaze coordinates (same scale as frontend)
    gaze_scale_x = 1.5
    gaze_scale_y = 6.0
    
    gaze_x_norm = face_center_x - yaw_ratio * gaze_scale_x
    gaze_y_norm = face_center_y - pitch_ratio * gaze_scale_y
    
    # Clamp to 0.0 ~ 1.0 range
    gaze_x_norm = max(0.0, min(1.0, gaze_x_norm))
    gaze_y_norm = max(0.0, min(1.0, gaze_y_norm))
    
    # Convert to map pixel coordinates (2148x1080)
    gaze_x_px = gaze_x_norm * 2148
    gaze_y_px = gaze_y_norm * 1080
    
    result["gaze_x"] = gaze_x_px
    result["gaze_y"] = gaze_y_px
    result["yaw_ratio"] = yaw_ratio
    result["pitch_ratio"] = pitch_ratio
    
    return result

# WebSocket endpoint to handle communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Generate and initialize session ID
    session_id = str(uuid.uuid4())
    print(f"[Session] Client connected (session: {session_id[:8]}...)")
    
    # Initialize game state per session
    game_sessions[session_id] = {
        "enemies": [],
        "effects": [],
        "playerGold": 0,
        "playerScore": 0,
        "playerHP": 100,
        "waveNumber": 1,
        "lastSkillTime": 0.0,
        "lastGestureTime": 0.0,
        "wave5EnemyCount": 0,
        "gestureSequence": generate_gesture_sequence(5)
    }
    
    # Initialize AI input per session (map pixel coordinates)
    ai_sessions[session_id] = {
        "gesture": "NONE",
        "gaze_x": 1074.0,  # 2148 / 2
        "gaze_y": 540.0,   # 1080 / 2
        "yaw_ratio": 0.0,
        "pitch_ratio": 0.0,
        "gestureMatched": False
    }
    
    # Start game loop per session
    game_task = asyncio.create_task(game_loop(websocket, session_id))
    session_tasks[session_id] = game_task

    try:
        latestAIInput = ai_sessions[session_id]  # Reference AI input per session
        
        while True:
            # Receive data from client (Base64 image or JSON command)
            data = await websocket.receive_text()
            
            # Handle JSON commands (skip button)
            try:
                message = json.loads(data)
                if message.get("type") == "skipGesture":
                    gameState = game_sessions.get(session_id)
                    if gameState and len(gameState["gestureSequence"]) > 0:
                        # Remove first letter and add new one
                        gameState["gestureSequence"].pop(0)
                        gameState["gestureSequence"].append(np.random.choice(AVAILABLE_GESTURES))
                    continue
            except json.JSONDecodeError:
                pass  # Continue if it's Base64 image
            
            # Parse Base64 image
            img_data = data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img_np = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run AI analysis
            results_hands = mp_hands.process(frame_rgb)
            results_face = mp_face_mesh.process(frame_rgb)

            # Prepare response data
            response_data = {
                "hand": "NONE",
                "gesture": "NONE",
                "hand_landmarks": [],
                "gaze": {
                    "x": 0.5,
                    "y": 0.5,
                    "yaw_ratio": 0.0,
                    "pitch_ratio": 0.0
                }
            }

            # Hand detection + ASL gesture recognition
            if results_hands.multi_hand_landmarks:
                response_data["hand"] = "DETECTED"
                
                # Add all hand landmark coordinates (up to 2 hands)
                response_data["hand_landmarks"] = [
                    [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in hand.landmark
                    ]
                    for hand in results_hands.multi_hand_landmarks
                ]
                
                # ASL gesture recognition (use first hand only)
                if asl_model is not None:
                    try:
                        first_hand = results_hands.multi_hand_landmarks[0]
                        features = extract_features_from_mediapipe(first_hand)
                        
                        if features is not None:
                            # Model prediction (convert features to 2D array: (1, 40))
                            prediction = asl_model.predict(features.reshape(1, -1))[0]
                            
                            # Check prediction probability (send only if 30% or higher)
                            if hasattr(asl_model, 'predict_proba'):
                                probabilities = asl_model.predict_proba(features.reshape(1, -1))[0]
                                max_probability = max(probabilities)
                                
                                if max_probability >= 0.30:
                                    response_data["gesture"] = prediction
                            else:
                                # Use default if predict_proba not available
                                response_data["gesture"] = prediction
                    except Exception as e:
                        print(f"[ASL] Gesture recognition error: {e}")

            # Face key point extraction + Gaze calculation
            if results_face.multi_face_landmarks:
                landmarks = results_face.multi_face_landmarks[0].landmark
                
                # Extract key points (for internal calculation)
                face_key_points = {
                    "nose_tip": {"x": landmarks[1].x, "y": landmarks[1].y, "z": landmarks[1].z},
                    "chin": {"x": landmarks[152].x, "y": landmarks[152].y, "z": landmarks[152].z},
                    "forehead": {"x": landmarks[10].x, "y": landmarks[10].y, "z": landmarks[10].z},
                    "left_face": {"x": landmarks[234].x, "y": landmarks[234].y, "z": landmarks[234].z},
                    "right_face": {"x": landmarks[454].x, "y": landmarks[454].y, "z": landmarks[454].z},
                    "left_eye": {"x": landmarks[33].x, "y": landmarks[33].y, "z": landmarks[33].z},
                    "right_eye": {"x": landmarks[263].x, "y": landmarks[263].y, "z": landmarks[263].z}
                }
                
                # Calculate gaze
                gaze_data = calculate_gaze(face_key_points)
                
                # Send normalized values to frontend
                response_data["gaze"] = {
                    "gaze_x": gaze_data["gaze_x"] / 2148,  # 0.0~1.0 normalized
                    "gaze_y": gaze_data["gaze_y"] / 1080,  # 0.0~1.0 normalized
                    "yaw_ratio": gaze_data["yaw_ratio"],
                    "pitch_ratio": gaze_data["pitch_ratio"]
                }
                
                # Update session AI input (store as pixel values)
                ai_sessions[session_id]["gaze_x"] = gaze_data["gaze_x"]
                ai_sessions[session_id]["gaze_y"] = gaze_data["gaze_y"]
                ai_sessions[session_id]["yaw_ratio"] = gaze_data["yaw_ratio"]
                ai_sessions[session_id]["pitch_ratio"] = gaze_data["pitch_ratio"]
                ai_sessions[session_id]["gesture"] = response_data["gesture"]

            # Send analysis results to client
            await websocket.send_json(response_data)

    except Exception as e:
        print(f"[Session] Connection lost or error (session: {session_id[:8]}...): {e}")
    finally:
        # Session cleanup
        game_task.cancel()
        if session_id in game_sessions:
            del game_sessions[session_id]
        if session_id in ai_sessions:
            del ai_sessions[session_id]
        if session_id in session_tasks:
            del session_tasks[session_id]
        print(f"[Session] Client connection terminated (session: {session_id[:8]}...)")

@app.on_event("startup")
async def startup_event():
    """Server startup complete"""
    print("✓ Server startup complete! Session-based game running...")


# Server execution (only when this file is run directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
