import uvicorn
import cv2
import numpy as np
import base64
import mediapipe as mp
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

# WebSocket 통신을 처리할 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("클라이언트 연결됨.")

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
                "hand_landmarks": [],
                "face_key_points": {
                    "nose_tip": None,
                    "chin": None,
                    "forehead": None,
                    "left_face": None,
                    "right_face": None,
                    "left_eye": None,
                    "right_eye": None
                }
            }

            # 손동작 인식 (이전과 동일)
            if results_hands.multi_hand_landmarks:
                # (향후 여기에 '주먹쥐기' 로직 추가)
                response_data["hand"] = "DETECTED"
                
                # 모든 손의 랜드마크 좌표 추가 (최대 2개 손)
                response_data["hand_landmarks"] = [
                    [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in hand.landmark
                    ]
                    for hand in results_hands.multi_hand_landmarks
                ] 

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ 변경: 얼굴 주요 포인트만 추출
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

            # 4. 분석 결과를 클라이언트(JS)로 전송
            await websocket.send_json(response_data)

    except Exception as e:
        print(f"연결 끊김 또는 오류: {e}")
    finally:
        print("클라이언트 연결 종료.")

# 서버 실행 (이 파일이 직접 실행될 때만)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
