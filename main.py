import uvicorn
import cv2
import numpy as np
import base64
import mediapipe as mp
import joblib
import asyncio
import time
import uuid
from pathlib import Path
from typing import List, Dict, Set
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# ASL 제스처 인식을 위한 특징 추출 모듈
from core.feature_extractor import extract_features_from_mediapipe

app = FastAPI()
# test
# CORS 설정 (모든 출처 허용 - 테스트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MediaPipe 모듈 초기화 (설정 변경) ---
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5
)
# ★ 변경: Face Mesh 설정 강화
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5 # 트래킹 민감도 추가
)

# --- ASL 제스처 인식 모델 로드 ---
MODEL_PATH = Path(__file__).parent / "models" / "asl_skill_model.pkl"
try:
    asl_model = joblib.load(MODEL_PATH)
    print(f"✓ ASL 제스처 인식 모델 로드 완료: {MODEL_PATH}")
except Exception as e:
    print(f"✗ 모델 로드 실패: {e}")
    asl_model = None

# --- 전역 변수: 최신 AI 입력 데이터 ---
latestAIInput = {
    "gesture": "NONE",
    "gaze_x": 0.5,  # 0.0 (왼쪽) ~ 1.0 (오른쪽)
    "gaze_y": 0.5,  # 0.0 (위) ~ 1.0 (아래)
    "yaw_ratio": 0.0,  # -1.0 (왼쪽) ~ +1.0 (오른쪽)
    "pitch_ratio": 0.0  # -1.0 (위) ~ +1.0 (아래)
}

# --- 게임 상태 전역 변수 ---
connected_clients: Set[WebSocket] = set()
gameState = {
    "enemies": [],
    "effects": [],
    "playerGold": 0,
    "playerScore": 0,
    "waveNumber": 1,
    "lastSkillTime": 0.0
}


# ==================== 게임 로직 클래스 ====================

class Enemy:
    """적 엔티티"""
    def __init__(self, enemy_id: str, type_id: str, x: float, y: float, speed: float = 50.0):
        self.id = enemy_id
        self.typeId = type_id
        self.x = x
        self.y = y
        self.currentHP = 100
        self.maxHP = 100
        self.animationState = "walk"
        self.currentFrame = 0
        self.isDead = False
        self.speed = speed  # pixels per second
        self.direction = -1 if x > 500 else 1  # 오른쪽에서 시작하면 왼쪽으로
    
    def update(self, delta_time: float):
        """적 이동 업데이트"""
        if not self.isDead:
            self.x += self.direction * self.speed * delta_time
            self.currentFrame = (self.currentFrame + 1) % 8
    
    def take_damage(self, damage: int):
        """데미지 입기"""
        self.currentHP -= damage
        if self.currentHP <= 0:
            self.currentHP = 0
            self.isDead = True
            self.animationState = "death"
    
    def to_dict(self) -> dict:
        """JSON 직렬화"""
        return {
            "id": self.id,
            "typeId": self.typeId,
            "x": self.x,
            "y": self.y,
            "currentHP": self.currentHP,
            "maxHP": self.maxHP,
            "animationState": self.animationState,
            "currentFrame": self.currentFrame,
            "isDead": self.isDead
        }


class Effect:
    """스킬 이펙트"""
    def __init__(self, effect_id: str, effect_type: str, x: float, y: float, duration: float = 0.5):
        self.id = effect_id
        self.type = effect_type
        self.x = x
        self.y = y
        self.duration = duration
        self.createdAt = time.time()
    
    def is_expired(self) -> bool:
        return time.time() - self.createdAt > self.duration
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "x": self.x,
            "y": self.y
        }


class Player:
    """플레이어 (스킬 시전)"""
    def __init__(self):
        self.skill_cooldown = 1.0  # 스킬 쿨다운 (초)
        self.skill_range = 200  # 스킬 범위 (pixels)
        self.skill_damage = 50
    
    def cast_skill(self, gesture: str, gaze_x: float, gaze_y: float) -> Dict:
        """스킬 시전"""
        current_time = time.time()
        
        # 쿨다운 체크
        if current_time - gameState["lastSkillTime"] < self.skill_cooldown:
            return None
        
        gameState["lastSkillTime"] = current_time
        
        # 스킬 타입 결정 (제스처에 따라)
        skill_types = {
            "A": "fireSlash",           # A - 불 베기
            "C": "fireVortexRed",       # C - 붉은 화염 소용돌이
            "L": "lightningV1",         # L - 번개
            "S": "skyBeam",             # S - 하늘 광선
            "T": "tornado",             # T - 토네이도 (추가)
            "M": "meteorShowerRed"      # M - 운석 샤워 (추가)
        }
        skill_type = skill_types.get(gesture, "fireSlash")
        
        # 시선 위치를 화면 좌표로 변환 (예: 1920x1080)
        target_x = gaze_x * 1920
        target_y = gaze_y * 1080
        
        return {
            "skill_type": skill_type,
            "target_x": target_x,
            "target_y": target_y,
            "damage": self.skill_damage,
            "range": self.skill_range
        }


player = Player()


# ==================== 게임 로직 함수 ====================

def spawn_enemy():
    """적 생성"""
    enemy_id = f"enemy_{uuid.uuid4().hex[:8]}"
    enemy_types = ["skeleton", "orc", "slime", "skeletonArcher"]
    type_id = enemy_types[len(gameState["enemies"]) % len(enemy_types)]
    
    # 좌우 랜덤 스폰
    x = 1800 if np.random.random() > 0.5 else 100
    y = 400 + np.random.randint(-100, 100)
    
    enemy = Enemy(enemy_id, type_id, x, y)
    gameState["enemies"].append(enemy)
    print(f"[Game] 적 생성: {enemy_id} ({type_id}) at ({x}, {y})")


def check_collision(skill_data: Dict) -> List[Enemy]:
    """스킬과 적 충돌 판정"""
    hit_enemies = []
    target_x = skill_data["target_x"]
    target_y = skill_data["target_y"]
    skill_range = skill_data["range"]
    
    for enemy in gameState["enemies"]:
        if enemy.isDead:
            continue
        
        # 거리 계산
        distance = np.sqrt((enemy.x - target_x)**2 + (enemy.y - target_y)**2)
        
        if distance <= skill_range:
            hit_enemies.append(enemy)
    
    return hit_enemies


# ==================== 게임 루프 (20fps) ====================

async def game_loop():
    """
    README 설계대로 asyncio 기반 게임 루프
    - 20fps (0.05초마다 실행)
    - latestAIInput을 읽어서 게임 로직 처리
    - Full State Sync JSON을 모든 클라이언트에게 브로드캐스트
    """
    print("[Game] 게임 루프 시작 (20fps)")
    
    last_spawn_time = time.time()
    spawn_interval = 3.0  # 3초마다 적 생성
    
    while True:
        loop_start = time.time()
        delta_time = 0.05  # 20fps
        
        # 1. 적 스포너
        if time.time() - last_spawn_time > spawn_interval:
            spawn_enemy()
            last_spawn_time = time.time()
        
        # 2. 적 업데이트 (이동)
        for enemy in gameState["enemies"]:
            enemy.update(delta_time)
        
        # 3. 죽은 적 제거
        gameState["enemies"] = [e for e in gameState["enemies"] if not e.isDead or time.time() - gameState["lastSkillTime"] < 1.0]
        
        # 4. AI 입력 확인 및 스킬 시전
        gesture = latestAIInput.get("gesture", "NONE")
        if gesture != "NONE" and gesture in ["A", "C", "L", "S", "T", "M"]:
            skill_data = player.cast_skill(
                gesture,
                latestAIInput["gaze_x"],
                latestAIInput["gaze_y"]
            )
            
            if skill_data:
                # 이펙트 생성
                effect_id = f"effect_{uuid.uuid4().hex[:8]}"
                effect = Effect(effect_id, skill_data["skill_type"], skill_data["target_x"], skill_data["target_y"])
                gameState["effects"].append(effect)
                
                # 충돌 판정
                hit_enemies = check_collision(skill_data)
                for enemy in hit_enemies:
                    enemy.take_damage(skill_data["damage"])
                    gameState["playerScore"] += 10
                    print(f"[Game] 적 {enemy.id} 타격! HP: {enemy.currentHP}/{enemy.maxHP}")
        
        # 5. 만료된 이펙트 제거
        gameState["effects"] = [e for e in gameState["effects"] if not e.is_expired()]
        
        # 6. Full State Sync 생성
        state_sync = {
            "gameState": {
                "enemies": [e.to_dict() for e in gameState["enemies"]],
                "effects": [e.to_dict() for e in gameState["effects"]],
                "gazePosition": {
                    "x": latestAIInput["gaze_x"] * 1920,
                    "y": latestAIInput["gaze_y"] * 1080
                },
                "playerGold": gameState["playerGold"],
                "playerScore": gameState["playerScore"],
                "waveNumber": gameState["waveNumber"]
            }
        }
        
        # 7. 모든 연결된 클라이언트에게 브로드캐스트
        disconnected = set()
        for client in connected_clients:
            try:
                await client.send_json(state_sync)
            except:
                disconnected.add(client)
        
        # 연결 끊긴 클라이언트 제거
        connected_clients.difference_update(disconnected)
        
        # 8. 20fps 유지
        elapsed = time.time() - loop_start
        sleep_time = max(0, delta_time - elapsed)
        await asyncio.sleep(sleep_time)


def calculate_gaze(face_key_points: dict) -> dict:
    """
    프론트엔드 로직을 참고하여 시선(Gaze) 방향 계산
    
    Returns:
        dict: {
            "gaze_x": 0.0~1.0,
            "gaze_y": 0.0~1.0,
            "yaw_ratio": -1.0~1.0,
            "pitch_ratio": -1.0~1.0
        }
    """
    data = face_key_points
    
    # 기본값
    result = {
        "gaze_x": 0.5,
        "gaze_y": 0.5,
        "yaw_ratio": 0.0,
        "pitch_ratio": 0.0
    }
    
    # 필수 포인트가 모두 있는지 확인
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
    
    # 얼굴 중심점 계산
    face_center_x = (left_eye["x"] + right_eye["x"]) / 2
    face_center_y = (left_eye["y"] + right_eye["y"]) / 2
    
    # Yaw (좌우 회전) 계산
    left_distance = abs(nose_tip["x"] - left_face["x"])
    right_distance = abs(nose_tip["x"] - right_face["x"])
    face_width = abs(right_face["x"] - left_face["x"])
    
    yaw_ratio = 0.0
    if face_width > 0:
        yaw_ratio = (left_distance - right_distance) / face_width
    
    # Pitch (상하 회전) 계산
    nose_to_forehead = abs(nose_tip["y"] - forehead["y"])
    nose_to_chin = abs(nose_tip["y"] - chin["y"])
    face_height = abs(chin["y"] - forehead["y"])
    
    pitch_ratio = 0.0
    if face_height > 0:
        pitch_ratio = (nose_to_chin - nose_to_forehead) / face_height + 0.15
    
    # 시선 좌표 매핑 (프론트엔드와 동일한 스케일)
    gaze_scale_x = 1.5
    gaze_scale_y = 6.0
    
    gaze_x = face_center_x - yaw_ratio * gaze_scale_x
    gaze_y = face_center_y - pitch_ratio * gaze_scale_y
    
    # 0.0 ~ 1.0 범위로 클램핑
    gaze_x = max(0.0, min(1.0, gaze_x))
    gaze_y = max(0.0, min(1.0, gaze_y))
    
    result["gaze_x"] = gaze_x
    result["gaze_y"] = gaze_y
    result["yaw_ratio"] = yaw_ratio
    result["pitch_ratio"] = pitch_ratio
    
    return result

# WebSocket 통신을 처리할 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"클라이언트 연결됨. (총 {len(connected_clients)}명)")

    try:
        while True:
            # 1. 클라이언트(JS)로부터 Base64 이미지(텍스트) 수신
            data = await websocket.receive_text()
            img_data = data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img_np = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. Vultr에서 AI 분석 실행
            results_hands = mp_hands.process(frame_rgb)
            results_face = mp_face_mesh.process(frame_rgb)

            # 3. 결과 분석 및 JSON 생성
            response_data = {
                "hand": "NONE",
                "gesture": "NONE",  # ASL 제스처 인식 결과
                "hand_landmarks": [],
                "face_key_points": {
                    "nose_tip": None,
                    "chin": None,
                    "forehead": None,
                    "left_face": None,
                    "right_face": None,
                    "left_eye": None,
                    "right_eye": None
                },
                "gaze": {
                    "x": 0.5,
                    "y": 0.5,
                    "yaw_ratio": 0.0,
                    "pitch_ratio": 0.0
                }
            }

            # 손동작 인식 + ASL 제스처 인식
            if results_hands.multi_hand_landmarks:
                response_data["hand"] = "DETECTED"
                
                # 모든 손의 랜드마크 좌표 추가 (최대 2개 손)
                response_data["hand_landmarks"] = [
                    [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in hand.landmark
                    ]
                    for hand in results_hands.multi_hand_landmarks
                ]
                
                # ASL 제스처 인식 (첫 번째 손만 사용)
                if asl_model is not None:
                    try:
                        first_hand = results_hands.multi_hand_landmarks[0]
                        features = extract_features_from_mediapipe(first_hand)
                        
                        if features is not None:
                            # 모델 예측 (features를 2D 배열로 변환: (1, 40))
                            prediction = asl_model.predict(features.reshape(1, -1))[0]
                            response_data["gesture"] = prediction
                            print(f"[ASL] 제스처 인식: {prediction}")
                    except Exception as e:
                        print(f"[ASL] 제스처 인식 오류: {e}") 

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ 변경: 얼굴 주요 포인트만 추출 + Gaze 계산
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if results_face.multi_face_landmarks:
                landmarks = results_face.multi_face_landmarks[0].landmark
                
                # 주요 포인트 추출 및 전송
                response_data["face_key_points"] = {
                    "nose_tip": {"x": landmarks[1].x, "y": landmarks[1].y, "z": landmarks[1].z},
                    "chin": {"x": landmarks[152].x, "y": landmarks[152].y, "z": landmarks[152].z},
                    "forehead": {"x": landmarks[10].x, "y": landmarks[10].y, "z": landmarks[10].z},
                    "left_face": {"x": landmarks[234].x, "y": landmarks[234].y, "z": landmarks[234].z},
                    "right_face": {"x": landmarks[454].x, "y": landmarks[454].y, "z": landmarks[454].z},
                    "left_eye": {"x": landmarks[33].x, "y": landmarks[33].y, "z": landmarks[33].z},
                    "right_eye": {"x": landmarks[263].x, "y": landmarks[263].y, "z": landmarks[263].z}
                }
                
                # Gaze 계산
                gaze_data = calculate_gaze(response_data["face_key_points"])
                response_data["gaze"] = gaze_data
                
                # latestAIInput 전역 변수 업데이트
                latestAIInput["gaze_x"] = gaze_data["gaze_x"]
                latestAIInput["gaze_y"] = gaze_data["gaze_y"]
                latestAIInput["yaw_ratio"] = gaze_data["yaw_ratio"]
                latestAIInput["pitch_ratio"] = gaze_data["pitch_ratio"]
                latestAIInput["gesture"] = response_data["gesture"]
                
                print(f"[Gaze] x={gaze_data['gaze_x']:.2f}, y={gaze_data['gaze_y']:.2f}, yaw={gaze_data['yaw_ratio']:.2f}, pitch={gaze_data['pitch_ratio']:.2f}")

            # 4. 분석 결과를 클라이언트(JS)로 전송
            await websocket.send_json(response_data)

    except Exception as e:
        print(f"연결 끊김 또는 오류: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"클라이언트 연결 종료. (남은 클라이언트: {len(connected_clients)}명)")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 게임 루프 시작"""
    asyncio.create_task(game_loop())
    print("✓ 서버 시작 완료! 게임 루프 실행 중...")


# 서버 실행 (이 파일이 직접 실행될 때만)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
