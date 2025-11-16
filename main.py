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

# --- 스킬 시스템: 사용 가능한 알파벳 ---
AVAILABLE_GESTURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

# --- 스킬 타입별 설정 (램덤 선택용) ---
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


# ==================== 게임 로직 클래스 ====================

# 적군 정보 (HP, 속도, 점수)
ENEMY_CONFIG = {
    # Tier 1: 기본 몬스터 (10점)
    "slime": {"hp": 50, "speed": 200, "score": 10},
    "skeleton": {"hp": 80, "speed": 50, "score": 10},
    "orc": {"hp": 100, "speed": 45, "score": 10},
    # Tier 2: 중급 몬스터 (15점)
    "skeletonArcher": {"hp": 120, "speed": 55, "score": 15},
    "armoredSkeleton": {"hp": 150, "speed": 40, "score": 15},
    "greatswordSkeleton": {"hp": 180, "speed": 35, "score": 15},
    # Tier 3: 고급 몬스터 (20점)
    "armoredOrc": {"hp": 250, "speed": 50, "score": 20},
    "elitOrc": {"hp": 300, "speed": 55, "score": 20},
    "orcRider": {"hp": 350, "speed": 60, "score": 20},
}

# 웨이브별 적군 티어
WAVE_ENEMY_TIERS = {
    1: ["slime", "skeleton"],
    2: ["slime", "skeleton", "orc"],
    3: ["skeleton", "orc", "skeletonArcher"],
    4: ["orc", "skeletonArcher", "armoredSkeleton"],
    5: ["skeletonArcher", "armoredSkeleton", "greatswordSkeleton", "armoredOrc", "elitOrc", "orcRider"],
}

# 웨이브별 목표 점수 (해당 점수 도달 시 다음 웨이브)
WAVE_SCORE_THRESHOLDS = {
    1: 80,   # 100점 도달 시 웨이브 2
    2: 170,   # 250점 도달 시 웨이브 3
    3: 270,   # 450점 도달 시 웨이브 4
    4: 380,   # 700점 도달 시 웨이브 5 (최종)
    5: float('inf')  # 웨이브 5는 무한
}

class Enemy:
    """적 엔티티"""
    def __init__(self, enemy_id: str, type_id: str, x: float, y: float, hp_bonus: int = 0, speed_multiplier: float = 1.0):
        self.id = enemy_id
        self.typeId = type_id
        self.x = x
        self.y = y
        
        # 적 타입에 따른 HP와 속도 설정
        config = ENEMY_CONFIG.get(type_id, {"hp": 100, "speed": 50, "score": 10})
        self.maxHP = config["hp"] + hp_bonus  # HP 보너스 추가
        self.currentHP = self.maxHP
        self.speed = config["speed"] * speed_multiplier  # 속도 배율 적용
        self.score = config["score"]  # 사망 시 획듩 점수
        
        self.animationState = "walk"
        self.currentFrame = 0
        self.isDead = False
        self.scoreGiven = False  # 점수 지급 여부
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
            "x": self.x / 2148,  # 0.0~1.0 정규화
            "y": self.y / 1080,  # 0.0~1.0 정규화
            "currentHP": self.currentHP,
            "maxHP": self.maxHP,
            "animationState": self.animationState,
            "currentFrame": self.currentFrame,
            "isDead": self.isDead
        }


class Effect:
    """스킬 이펙트"""
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
            "x": self.x / 2148,  # 0.0~1.0 정규화
            "damage": self.damage
        }


class Player:
    """플레이어 (더 이상 사용 안함 - 프론트에서 스킬 처리)"""
    pass


player = Player()


def generate_gesture_sequence(count: int = 5) -> List[str]:
    """램덤한 알파벳 시퀀스 생성 (5개)"""
    return [np.random.choice(AVAILABLE_GESTURES) for _ in range(count)]


# ==================== 게임 로직 함수 ====================

def spawn_enemy(session_id: str):
    """웨이브에 따라 적 생성"""
    enemy_id = f"enemy_{uuid.uuid4().hex[:8]}"
    gameState = game_sessions[session_id]
    current_wave = gameState["waveNumber"]
    
    # 웨이브별 적군 선택
    enemy_pool = WAVE_ENEMY_TIERS.get(current_wave, ["skeleton", "orc"])
    type_id = np.random.choice(enemy_pool)
    
    # 웨이브 5에서 HP 증가 (생성한 적 수에 비례)
    hp_bonus = 0
    if current_wave == 5:
        hp_bonus = gameState.get("wave5EnemyCount", 0) * 5  # 적 1마리당 +5 HP
        gameState["wave5EnemyCount"] = gameState.get("wave5EnemyCount", 0) + 1
    
    # 웨이브별 속도 증가 (웨이브당 10% 증가)
    speed_multiplier = 1.0 + (current_wave - 1) * 0.1  # 웨이브 1: 1.0x, 2: 1.1x, 3: 1.2x, 4: 1.3x, 5: 1.4x
    
    # 오른쪽 스폰 (맵 width 2148 픽셀 기준)
    x = 2000
    # y축: 화면 하단 50~70% (1080 기준 540~756)
    y = 648 + np.random.randint(0, 217)  # 540 + [0~216]
    
    enemy = Enemy(enemy_id, type_id, x, y, hp_bonus, speed_multiplier)
    gameState["enemies"].append(enemy)
    
    config = ENEMY_CONFIG.get(type_id, {})
    # print(f"[Game] 적 생성: {enemy_id} ({type_id}) HP:{config.get('hp', 100)} at ({x}, {y})")


def check_collision(session_id: str, effect: Effect) -> List[Enemy]:
    """스킬과 적 충돌 판정 (x축만 사용, 픽셀 기준)"""
    hit_enemies = []
    target_x = effect.x  # 맵 픽셀 (0~2148)
    skill_range = effect.range  # 픽셀 단위
    gameState = game_sessions[session_id]
    
    for enemy in gameState["enemies"]:
        if enemy.isDead:
            continue
        
        # x축 픽셀 거리 계산
        distance = abs(enemy.x - target_x)
        
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
    
    try:
        while True:
            loop_start = time.time()
            delta_time = 0.05  # 20fps
            
            # 0. 웨이브 진행 체크 (점수 기반)
            current_wave = gameState["waveNumber"]
            current_score = gameState["playerScore"]
            threshold = WAVE_SCORE_THRESHOLDS.get(current_wave, float('inf'))
            
            if current_wave < 5 and current_score >= threshold:
                gameState["waveNumber"] += 1
                print(f"[Game] 웨이브 {gameState['waveNumber']} 시작! (세션: {session_id[:8]}..., 점수: {current_score})")
            
            # 1. 적 스포너
            if time.time() - last_spawn_time > spawn_interval:
                spawn_enemy(session_id)
                last_spawn_time = time.time()
            
            # 2. 적 업데이트 (이동 및 화면 이탈 체크)
            for enemy in gameState["enemies"]:
                enemy.update(delta_time)
                
                # 적이 플레이어 위치에 도달하면 플레이어 HP 감소
                if enemy.x <= 530 and not enemy.isDead:
                    gameState["playerHP"] -= 10
                    enemy.isDead = True  # 도달한 적은 제거
                    enemy.scoreGiven = True  # 점수 지급 방지 (도달한 적은 점수 없음)
                    print(f"[Game] 적 통과! HP: {gameState['playerHP']} (세션: {session_id[:8]}...)")
            
            # 3. 죽은 적 제거 및 점수 획듩 (한 번만)
            for enemy in gameState["enemies"]:
                if enemy.isDead and not enemy.scoreGiven:
                    # 적 사망 시 티어별 점수 획듩 (Tier 1: 10, Tier 2: 15, Tier 3: 20)
                    gameState["playerScore"] += enemy.score
                    enemy.scoreGiven = True
            
            gameState["enemies"] = [e for e in gameState["enemies"] if not e.isDead or time.time() - gameState["lastSkillTime"] < 1.0]
            
            # 4. 게임 오버 체크
            if gameState["playerHP"] <= 0:
                print(f"[Game] 게임 오버! (세션: {session_id[:8]}..., 최종 점수: {gameState['playerScore']})")
                
                # 게임 오버 신호 전송
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
            
            # 5. AI 입력 확인 및 제스처 매칭 체크
            gesture = latestAIInput.get("gesture", "NONE")
            gesture_matched = False
            current_time = time.time()
            gesture_cooldown = 0.5  # 제스처 인식 쿨다운 (0.5초)
            
            if gesture != "NONE" and len(gameState["gestureSequence"]) > 0:
                # 쿨다운 체크
                if current_time - gameState["lastGestureTime"] < gesture_cooldown:
                    pass  # 쿨다운 중이면 무시
                # 시퀀스의 첫 번째 알파벳과 비교
                elif gesture == gameState["gestureSequence"][0]:
                    gesture_matched = True
                    gameState["lastGestureTime"] = current_time  # 쿨다운 시작
                    
                    # 시퀀스에서 제거
                    gameState["gestureSequence"].pop(0)
                    # 새로운 알파벳 추가 (리스트 5개 유지)
                    gameState["gestureSequence"].append(np.random.choice(AVAILABLE_GESTURES))
                    
                    # 램덤 스킬 선택
                    skill = np.random.choice(SKILL_TYPES)
                    
                    # 이펙트 생성 (gaze 위치에)
                    effect_id = f"effect_{uuid.uuid4().hex[:8]}"
                    effect = Effect(
                        effect_id,
                        skill["type"],
                        latestAIInput["gaze_x"],  # 맵 픽셀 좌표
                        skill["damage"],
                        skill["range"]
                    )
                    gameState["effects"].append(effect)
                    
                    # 충돌 판정 및 데미지 처리
                    hit_enemies = check_collision(session_id, effect)
                    for enemy in hit_enemies:
                        enemy.take_damage(skill["damage"])
                    
                    print(f"[Game] 제스처 매칭! {gesture} | 스킬: {skill['type']} (데미지: {skill['damage']}) | 새 시퀀스: {gameState['gestureSequence']} (세션: {session_id[:8]}...)")
            # 매칭 성공 여부 저장
            latestAIInput["gestureMatched"] = gesture_matched
            
            # 6. 만료된 이펙트 제거
            gameState["effects"] = [e for e in gameState["effects"] if not e.is_expired()]
            
            # 7. Full State Sync 생성 (정규화된 좌표)
            gaze_x_norm = latestAIInput["gaze_x"] / 2148
            gaze_y_norm = latestAIInput["gaze_y"] / 1080
            
            state_sync = {
                "gameState": {
                    "enemies": [e.to_dict() for e in gameState["enemies"]],
                    "effects": [e.to_dict() for e in gameState["effects"]],
                    "gazePosition": {
                        "x": gaze_x_norm,  # 0.0~1.0 정규화
                        "y": gaze_y_norm   # 0.0~1.0 정규화
                    },
                    "playerGold": gameState["playerGold"],
                    "playerScore": gameState["playerScore"],
                    "playerHP": gameState["playerHP"],
                    "waveNumber": gameState["waveNumber"],
                    "gestureSequence": gameState["gestureSequence"],  # 알파벳 시퀀스
                    "gestureMatched": latestAIInput.get("gestureMatched", False)  # 매칭 성공 여부
                }
            }
            
            # 8. 해당 세션 클라이언트에게만 전송
            try:
                await websocket.send_json(state_sync)
            except:
                print(f"[Game] 세션 {session_id[:8]}... 연결 끊김")
                break
            
            # 9. 20fps 유지
            elapsed = time.time() - loop_start
            sleep_time = max(0, delta_time - elapsed)
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        print(f"[Game] 세션 {session_id[:8]}... 게임 루프 종료")


def calculate_gaze(face_key_points: dict) -> dict:
    """
    프론트엔드 로직을 참고하여 시선(Gaze) 방향 계산
    
    Returns:
        dict: {
            "gaze_x": 맵 픽셀 (0~2148),
            "gaze_y": 맵 픽셀 (0~1080),
            "yaw_ratio": -1.0~1.0,
            "pitch_ratio": -1.0~1.0
        }
    """
    data = face_key_points
    
    # 기본값 (맵 중앙)
    result = {
        "gaze_x": 1074.0,  # 2148 / 2
        "gaze_y": 540.0,   # 1080 / 2
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
    
    gaze_x_norm = face_center_x - yaw_ratio * gaze_scale_x
    gaze_y_norm = face_center_y - pitch_ratio * gaze_scale_y
    
    # 0.0 ~ 1.0 범위로 클램핑
    gaze_x_norm = max(0.0, min(1.0, gaze_x_norm))
    gaze_y_norm = max(0.0, min(1.0, gaze_y_norm))
    
    # 맵 픽셀 좌표로 변환 (2148x1080)
    gaze_x_px = gaze_x_norm * 2148
    gaze_y_px = gaze_y_norm * 1080
    
    result["gaze_x"] = gaze_x_px
    result["gaze_y"] = gaze_y_px
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
        "playerHP": 100,  # 플레이어 HP
        "waveNumber": 1,
        "lastSkillTime": 0.0,
        "lastGestureTime": 0.0,  # 마지막 제스처 인식 시간
        "wave5EnemyCount": 0,  # 웨이브 5 적 카운터
        "gestureSequence": generate_gesture_sequence(5)  # 램덤 5개 알파벳 시퀀스
    }
    
    # 세션별 AI 입력 초기화 (맵 픽셀 좌표)
    ai_sessions[session_id] = {
        "gesture": "NONE",
        "gaze_x": 1074.0,  # 2148 / 2
        "gaze_y": 540.0,   # 1080 / 2
        "yaw_ratio": 0.0,
        "pitch_ratio": 0.0,
        "gestureMatched": False  # 제스처 매칭 여부
    }
    
    # 세션별 게임 루프 시작
    game_task = asyncio.create_task(game_loop(websocket, session_id))
    session_tasks[session_id] = game_task

    try:
        latestAIInput = ai_sessions[session_id]  # 세션별 AI 입력 참조
        
        while True:
            # 1. 클라이언트(JS)로부터 메시지 수신 (Base64 이미지 또는 JSON 명령)
            data = await websocket.receive_text()
            
            # JSON 명령 처리 (스킵 버튼)
            try:
                message = json.loads(data)
                if message.get("type") == "skipGesture":
                    gameState = game_sessions.get(session_id)
                    if gameState and len(gameState["gestureSequence"]) > 0:
                        # 첫 번째 알파벳 제거하고 새로운 알파벳 추가
                        gameState["gestureSequence"].pop(0)
                        gameState["gestureSequence"].append(np.random.choice(AVAILABLE_GESTURES))
                        print(f"[Game] 알파벳 스킵! (세션: {session_id[:8]}...)")
                    continue
            except json.JSONDecodeError:
                pass  # Base64 이미지인 경우 계속 진행
            
            # Base64 이미지 파싱
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
                                
                                if max_probability >= 0.30:
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
                
                # 프론트엔드로 정규화된 값 전송
                response_data["gaze"] = {
                    "gaze_x": gaze_data["gaze_x"] / 2148,  # 0.0~1.0 정규화
                    "gaze_y": gaze_data["gaze_y"] / 1080,  # 0.0~1.0 정규화
                    "yaw_ratio": gaze_data["yaw_ratio"],
                    "pitch_ratio": gaze_data["pitch_ratio"]
                }
                
                # 세션별 AI 입력 업데이트 (픽셀 값으로 저장)
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
