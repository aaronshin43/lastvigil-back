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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

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

# --- 세션 기반 저장소 (각 유저마다 독립적인 게임 상태) ---
game_sessions: Dict[str, dict] = {}  # {session_id: gameState}
ai_sessions: Dict[str, dict] = {}     # {session_id: latestAIInput}
session_tasks: Dict[str, asyncio.Task] = {}  # {session_id: game_loop_task}


# ==================== 게임 로직 클래스 ====================

# 적군 정보 (HP, 속도)
ENEMY_CONFIG = {
    # Tier 1: 기본 몬스터
    "slime": {"hp": 50, "speed": 40},
    "skeleton": {"hp": 80, "speed": 50},
    "orc": {"hp": 100, "speed": 45},
    # Tier 2: 중급 몬스터
    "skeletonArcher": {"hp": 120, "speed": 55},
    "armoredSkeleton": {"hp": 150, "speed": 40},
    "greatswordSkeleton": {"hp": 180, "speed": 35},
    # Tier 3: 고급 몬스터
    "armoredOrc": {"hp": 250, "speed": 50},
    "elitOrc": {"hp": 300, "speed": 55},
    "orcRider": {"hp": 350, "speed": 60},
}

# 웨이브별 적군 티어
WAVE_ENEMY_TIERS = {
    1: ["slime", "skeleton"],
    2: ["slime", "skeleton", "orc"],
    3: ["skeleton", "orc", "skeletonArcher"],
    4: ["orc", "skeletonArcher", "armoredSkeleton"],
    5: ["skeletonArcher", "armoredSkeleton", "greatswordSkeleton"],
    # 6웨이브 이상부터 Tier 3 등장
}

class Enemy:
    """적 엔티티"""
    def __init__(self, enemy_id: str, type_id: str, x: float, y: float):
        self.id = enemy_id
        self.typeId = type_id
        self.x = x
        self.y = y
        
        # 적 타입에 따른 HP와 속도 설정
        config = ENEMY_CONFIG.get(type_id, {"hp": 100, "speed": 50})
        self.maxHP = config["hp"]
        self.currentHP = self.maxHP
        self.speed = config["speed"]  # pixels per second
        
        self.animationState = "walk"
        self.currentFrame = 0
        self.isDead = False
        self.direction = -1 if x > 500 else 1  # 오른쪽에서 시작하면 왼쪽으로
        self.hurtFrameCount = 0  # hurt 애니메이션 지속 프레임
    
    def update(self, delta_time: float):
        """적 이동 업데이트"""
        if not self.isDead:
            self.x += self.direction * self.speed * delta_time
            self.currentFrame = (self.currentFrame + 1) % 8
            
            # hurt 상태는 8프레임만 유지
            if self.animationState == "hurt":
                self.hurtFrameCount += 1
                if self.hurtFrameCount >= 8:
                    self.animationState = "walk"
                    self.hurtFrameCount = 0
    
    def take_damage(self, damage: int):
        """데미지 입기"""
        self.currentHP -= damage
        if self.currentHP <= 0:
            self.currentHP = 0
            self.isDead = True
            self.animationState = "death"
        else:
            self.animationState = "hurt"
            self.hurtFrameCount = 0  # hurt 카운터 리셋
    
    def to_dict(self) -> dict:
        """JSON 직렬화 (정규화된 좌표로 전송)"""
        return {
            "id": self.id,
            "typeId": self.typeId,
            "x": self.x / 1920,  # 0.0~1.0 정규화
            "y": self.y / 1080,  # 0.0~1.0 정규화
            "currentHP": self.currentHP,
            "maxHP": self.maxHP,
            "animationState": self.animationState,
            "currentFrame": self.currentFrame,
            "isDead": self.isDead
        }


class Effect:
    """스킬 이펙트"""
    def __init__(self, effect_id: str, effect_type: str, x: float, duration: float = 0.5):
        self.id = effect_id
        self.type = effect_type
        self.x = x
        self.duration = duration
        self.createdAt = time.time()
    
    def is_expired(self) -> bool:
        return time.time() - self.createdAt > self.duration
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "x": self.x
        }


class Player:
    """플레이어 (스킬 시전)"""
    def __init__(self):
        self.skill_cooldown = 1.0  # 스킬 쿨다운 (초)
        self.skill_range = 0.15  # 스킬 범위 (정규화, 0.0~1.0)
        self.skill_damage = 50
    
    def cast_skill(self, session_id: str, gesture: str, gaze_x: float, gaze_y: float) -> Dict:
        """스킬 시전"""
        current_time = time.time()
        gameState = game_sessions[session_id]
        
        # 쿨다운 체크
        if current_time - gameState["lastSkillTime"] < self.skill_cooldown:
            return None
        
        gameState["lastSkillTime"] = current_time
        
        # 스킬 타입과 데미지 결정 (제스처에 따라)
        skill_config = {
            "A": {"type": "fireSlash", "damage": 50},              # A - 불 베기 (기본)
            "B": {"type": "skyBeam", "damage": 80},                # B - 하늘 광선 (강력)
            "C": {"type": "fireVortexRed", "damage": 60},          # C - 불 소용돌이
            "D": {"type": "fireHammerRed", "damage": 100},         # D - 불 망치 (최강)
            "L": {"type": "lightningV1", "damage": 70},            # L - 번개 V1
            "K": {"type": "lightningV2", "damage": 75},            # K - 번개 V2
            "R": {"type": "meteorShowerRed", "damage": 90},        # R - 메테오 샤워
            "V": {"type": "fireHurricaneBlue", "damage": 85},      # V - 불 허리케인
            "W": {"type": "tornado", "damage": 65},                # W - 토네이도
        }
        
        config = skill_config.get(gesture, {"type": "fireSlash", "damage": 50})
        
        # x축만 사용 (0.0~1.0)
        # 프론트엔드에서 화면 크기에 맞게 스케일링
        return {
            "skill_type": config["type"],
            "target_x": gaze_x,
            "damage": config["damage"],
            "range": self.skill_range
        }


player = Player()


# ==================== 게임 로직 함수 ====================

def spawn_enemy(session_id: str):
    """웨이브에 따라 적 생성"""
    enemy_id = f"enemy_{uuid.uuid4().hex[:8]}"
    gameState = game_sessions[session_id]
    current_wave = gameState["waveNumber"]
    
    # 웨이브별 적군 선택 (낮은 티어도 계속 등장)
    if current_wave <= 5:
        enemy_pool = WAVE_ENEMY_TIERS.get(current_wave, ["skeleton", "orc"])
    else:
        # 6웨이브 이상: 모든 티어 혼합 (낮은 티어 확률 낮춤)
        tier1 = ["slime", "skeleton", "orc"]
        tier2 = ["skeletonArcher", "armoredSkeleton", "greatswordSkeleton"]
        tier3 = ["armoredOrc", "elitOrc", "orcRider"]
        
        # Tier 1: 20%, Tier 2: 40%, Tier 3: 40%
        tier_choice = np.random.random()
        if tier_choice < 0.2:
            enemy_pool = tier1
        elif tier_choice < 0.6:
            enemy_pool = tier2
        else:
            enemy_pool = tier3
    
    type_id = np.random.choice(enemy_pool)
    
    # 오른쪽 스폰
    x = 1800 
    # y축: 화면 하단 50~70% (1080 기준 540~756)
    y = 648 + np.random.randint(0, 217)  # 540 + [0~216]
    
    enemy = Enemy(enemy_id, type_id, x, y)
    gameState["enemies"].append(enemy)
    
    config = ENEMY_CONFIG.get(type_id, {})
    # print(f"[Game] 적 생성: {enemy_id} ({type_id}) HP:{config.get('hp', 100)} at ({x}, {y})")


def check_collision(session_id: str, skill_data: Dict) -> List[Enemy]:
    """스킬과 적 충돌 판정 (x축만 사용)"""
    hit_enemies = []
    target_x = skill_data["target_x"]  # 0.0~1.0
    skill_range = skill_data["range"]  # 0.0~1.0
    gameState = game_sessions[session_id]
    
    print(f"[Collision] 스킬 타겟 x={target_x:.3f}, 범위={skill_range:.3f}")
    
    for enemy in gameState["enemies"]:
        if enemy.isDead:
            continue
        
        # 적 x좌표를 정규화 (1920 기준)
        enemy_x_norm = enemy.x / 1920
        
        # x축 거리만 계산
        distance = abs(enemy_x_norm - target_x)
        
        # print(f"[Collision] 적 {enemy.id}: x={enemy.x:.1f} (정규화={enemy_x_norm:.3f}), 거리={distance:.3f}, 타격={'O' if distance <= skill_range else 'X'}")
        
        if distance <= skill_range:
            hit_enemies.append(enemy)
    
    return hit_enemies


# ==================== 게임 루프 (20fps) ====================

async def game_loop(websocket: WebSocket, session_id: str):
    """
    세션별 게임 루프
    - 20fps (0.05초마다 실행)
    - 해당 세션의 AI 입력을 읽어서 게임 로직 처리
    - 해당 세션의 클라이언트에게만 브로드캐스트
    """
    print(f"[Game] 게임 루프 시작 (세션: {session_id[:8]}...)")
    
    gameState = game_sessions[session_id]
    latestAIInput = ai_sessions[session_id]
    
    last_spawn_time = time.time()
    spawn_interval = 3.0  # 3초마다 적 생성
    gameState["waveStartTime"] = time.time()  # 웨이브 시작 시간 기록
    
    try:
        while True:
            loop_start = time.time()
            delta_time = 0.05  # 20fps
            
            # 0. 웨이브 진행 체크 (10초마다 웨이브 증가)
            wave_elapsed = time.time() - gameState["waveStartTime"]
            if wave_elapsed >= 10.0:
                gameState["waveNumber"] += 1
                gameState["waveStartTime"] = time.time()
                print(f"[Game] 웨이브 {gameState['waveNumber']} 시작! (세션: {session_id[:8]}...)")
            
            # 1. 적 스포너
            if time.time() - last_spawn_time > spawn_interval:
                spawn_enemy(session_id)
                last_spawn_time = time.time()
            
            # 2. 적 업데이트 (이동)
            for enemy in gameState["enemies"]:
                enemy.update(delta_time)
            
            # 3. 죽은 적 제거
            gameState["enemies"] = [e for e in gameState["enemies"] if not e.isDead or time.time() - gameState["lastSkillTime"] < 1.0]
            
            # 4. AI 입력 확인 및 스킬 시전
            gesture = latestAIInput.get("gesture", "NONE")
            if gesture != "NONE" and gesture in ["A", "B", "C", "D", "L", "K", "R", "V", "W"]:
                skill_data = player.cast_skill(
                    session_id,
                    gesture,
                    latestAIInput["gaze_x"],
                    latestAIInput["gaze_y"]
                )
                
                if skill_data:
                    # 이펙트 생성
                    effect_id = f"effect_{uuid.uuid4().hex[:8]}"
                    effect = Effect(effect_id, skill_data["skill_type"], skill_data["target_x"])
                    gameState["effects"].append(effect)
                    
                    # 충돌 판정
                    hit_enemies = check_collision(session_id, skill_data)
                    for enemy in hit_enemies:
                        enemy.take_damage(skill_data["damage"])
                        gameState["playerScore"] += 10
            
            # 5. 만료된 이펙트 제거
            gameState["effects"] = [e for e in gameState["effects"] if not e.is_expired()]
            
            # 6. Full State Sync 생성
            state_sync = {
                "gameState": {
                    "enemies": [e.to_dict() for e in gameState["enemies"]],
                    "effects": [e.to_dict() for e in gameState["effects"]],
                    "gazePosition": {
                        "x": latestAIInput["gaze_x"],
                        "y": latestAIInput["gaze_y"]
                    },
                    "playerGold": gameState["playerGold"],
                    "playerScore": gameState["playerScore"],
                    "waveNumber": gameState["waveNumber"]
                }
            }
            
            # 7. 해당 세션 클라이언트에게만 전송
            try:
                await websocket.send_json(state_sync)
            except:
                print(f"[Game] 세션 {session_id[:8]}... 연결 끊김")
                break
            
            # 8. 20fps 유지
            elapsed = time.time() - loop_start
            sleep_time = max(0, delta_time - elapsed)
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        print(f"[Game] 세션 {session_id[:8]}... 게임 루프 종료")
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
    
    # 세션 ID 생성 및 초기화
    session_id = str(uuid.uuid4())
    print(f"[Session] 클라이언트 연결됨 (세션: {session_id[:8]}...)")
    
    # 세션별 게임 상태 초기화
    game_sessions[session_id] = {
        "enemies": [],
        "effects": [],
        "playerGold": 0,
        "playerScore": 0,
        "waveNumber": 1,
        "lastSkillTime": 0.0,
        "waveStartTime": 0.0
    }
    
    # 세션별 AI 입력 초기화
    ai_sessions[session_id] = {
        "gesture": "NONE",
        "gaze_x": 0.5,
        "gaze_y": 0.5,
        "yaw_ratio": 0.0,
        "pitch_ratio": 0.0
    }
    
    # 세션별 게임 루프 시작
    game_task = asyncio.create_task(game_loop(websocket, session_id))
    session_tasks[session_id] = game_task

    try:
        latestAIInput = ai_sessions[session_id]  # 세션별 AI 입력 참조
        
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
                            
                            # 예측 확률 확인 (90% 이상일 때만 전송)
                            if hasattr(asl_model, 'predict_proba'):
                                probabilities = asl_model.predict_proba(features.reshape(1, -1))[0]
                                max_probability = max(probabilities)
                                
                                if max_probability >= 0.50:
                                    response_data["gesture"] = prediction
                            else:
                                # predict_proba가 없는 모델인 경우 기본값 사용
                                response_data["gesture"] = prediction
                    except Exception as e:
                        print(f"[ASL] 제스처 인식 오류: {e}") 

            # 얼굴 주요 포인트 추출 + Gaze 계산
            if results_face.multi_face_landmarks:
                landmarks = results_face.multi_face_landmarks[0].landmark
                
                # 주요 포인트 추출 (내부 계산용)
                face_key_points = {
                    "nose_tip": {"x": landmarks[1].x, "y": landmarks[1].y, "z": landmarks[1].z},
                    "chin": {"x": landmarks[152].x, "y": landmarks[152].y, "z": landmarks[152].z},
                    "forehead": {"x": landmarks[10].x, "y": landmarks[10].y, "z": landmarks[10].z},
                    "left_face": {"x": landmarks[234].x, "y": landmarks[234].y, "z": landmarks[234].z},
                    "right_face": {"x": landmarks[454].x, "y": landmarks[454].y, "z": landmarks[454].z},
                    "left_eye": {"x": landmarks[33].x, "y": landmarks[33].y, "z": landmarks[33].z},
                    "right_eye": {"x": landmarks[263].x, "y": landmarks[263].y, "z": landmarks[263].z}
                }
                
                # Gaze 계산
                gaze_data = calculate_gaze(face_key_points)
                response_data["gaze"] = gaze_data
                
                # 세션별 AI 입력 업데이트
                latestAIInput["gaze_x"] = gaze_data["gaze_x"]
                latestAIInput["gaze_y"] = gaze_data["gaze_y"]
                latestAIInput["yaw_ratio"] = gaze_data["yaw_ratio"]
                latestAIInput["pitch_ratio"] = gaze_data["pitch_ratio"]
                latestAIInput["gesture"] = response_data["gesture"]

            # 4. 분석 결과를 클라이언트(JS)로 전송
            await websocket.send_json(response_data)

    except Exception as e:
        print(f"[Session] 연결 끊김 또는 오류 (세션: {session_id[:8]}...): {e}")
    finally:
        # 세션 정리
        game_task.cancel()
        if session_id in game_sessions:
            del game_sessions[session_id]
        if session_id in ai_sessions:
            del ai_sessions[session_id]
        if session_id in session_tasks:
            del session_tasks[session_id]
        print(f"[Session] 클라이언트 연결 종료 (세션: {session_id[:8]}...)")

@app.on_event("startup")
async def startup_event():
    """서버 시작 완료"""
    print("✓ 서버 시작 완료! 세션 기반 게임 실행 중...")


# 서버 실행 (이 파일이 직접 실행될 때만)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
