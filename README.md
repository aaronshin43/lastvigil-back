pip install "fastapi[all]" python-socketio opencv-python-headless mediapipe numpy

# 1. 📖 Project: **The Last Vigil**

## 한 줄 요약

"플레이어의 **시선(Gaze)**과 **ASL 알파벳 손 제스처**를 실시간으로 분석해 마법을 시전하는 2D 고딕 액션 디펜스 게임입니다."

## 핵심 컨셉

플레이어는 마우스나 키보드를 사용하지 않고 **웹캠만으로** 게임을 조작합니다.

- **시선(Gaze) = 조준**
  화면의 좌우 어디를 보는지에 따라 공격 방향이 자동으로 결정됩니다.
- **ASL 제스처(Gesture) = 스킬 발동**
  A, C, L, S 같은 특정 알파벳 제스처를 인식해 강력한 마법을 발사합니다.

## 테마 및 아트 스타일

- **테마:** 고딕 호러
- **적:** 좀비, 해골 등
- **비주얼 스타일:** 2D 픽셀 아트
- **맵 구성:** 팔라독처럼 좌우로 길게 이어진 횡스크롤 전장

---

# 🌟 핵심 아키텍처 (Vultr 스폰서 트랙)

이 프로젝트는 웹 게임처럼 보이지만 내부적으로는 **Vultr GPU 서버에서 AI 분석과 게임 로직을 수행하는 클라우드 기반 AI 시스템**입니다.

## 1. 프론트엔드 (Client): 렌더링과 효과 중심

**기술:** JavaScript, Canvas 기반 렌더링, WebSocket

**역할:**

- 2D 맵 이미지와 적 스프라이트를 화면에 그립니다.
- Vultr 서버로부터 받은 명령을 그대로 수행합니다.
  - 예: "적 3번 제거", "폭발 이펙트 재생"

## 2. 백엔드 (Vultr Cloud GPU): AI 분석과 게임 로직

**기술:** Python, FastAPI WebSocket, MediaPipe, OpenCV

**역할:**

- 클라이언트가 보내는 웹캠 프레임을 GPU에서 실시간 분석
- MediaPipe로 직접 학습한 **ASL 제스처 모델**을 사용
- 시선 추적 모듈(OpenCV 기반)로 좌우 시선 방향 분석 및 “정량화된 시선값[-1, 1]” 전달
- 적 생성, 이동, 체력, 충돌 판정 등 게임 로직 전부 서버에서 처리

**어필 포인트:**

로컬 성능에 의존하지 않고 **Vultr GPU가 모든 무거운 연산을 처리하는 클라우드형 디펜스 게임**입니다.

---

# 2. 🚀 MVP 우선순위

해커톤 기준으로 가장 빠른 구현 순서입니다.

## 1단계: 맵과 화면 구성

- [ ] 2D 배경 이미지 1장 로드
- [ ] 화면 좌우가 긴 횡스크롤 맵 구성
- [ ] 카메라는 중앙에서 고정된 형태로 단순하게 처리

## 2단계: 적 시스템

- [ ] 적 스프라이트 로드
- [ ] 적이 왼쪽에서 오른쪽으로, 혹은 오른쪽에서 왼쪽으로 이동
- [ ] 간단한 패턴으로 웨이브 없이 계속 스폰

## 3단계: Vultr AI 브레인

- [ ] Vultr GPU 인스턴스 생성 후 FastAPI WebSocket 서버 오픈
- [ ] 클라이언트에서 웹캠 프레임을 서버로 지속 전송
- [ ] 서버에서 MediaPipe로 특정 ASL 알파벳 한 개만 우선 인식
- [ ] 인식되면 WebSocket으로 클라이언트에 메시지 전송
  - 예: {"gesture":"A"}

## 4단계: 핵심 루프

- [ ] 서버에서 시선 좌우 방향 판독
- [ ] 서버는 ASL 제스처가 감지된 시점의 시선 방향을 기준으로 공격 방향 결정
- [ ] 서버가 적이 공격 범위에 있는지 판정
- [ ] 판정 후 클라이언트에 "적 X 제거" 명령 전송
- [ ] 클라이언트는 해당 적을 화면에서 제거

## 5단계: 피드백

- [ ] 폭발 또는 마법 타격 이펙트 스프라이트 재생
- [ ] 사운드는 외부 에셋을 그대로 재생 (API 사용 없음)

백엔드는 "AI 분석"과 "게임 로직"이라는 두 가지 작업을 2개의 vCPU에 분배해야 합니다.

데이터 수신 및 AI 분석 (Task 1: AI Core)
FastAPI의 WebSocket 엔드포인트(@app.websocket("/ws"))는 클라이언트로부터 10fps로 Base64 문자열을 수신합니다.

문자열을 디코딩하여 OpenCV가 읽을 수 있는 Numpy 배열(이미지)로 변환합니다.

이 이미지를 **MediaPipe**와 OpenCV 모듈에 밀어 넣습니다.

[CPU 병목점] 2개의 vCPU가 이 이미지를 분석하느라 50ms~100ms 동안 "정지"합니다.

분석이 완료되면, "원시 AI 데이터"를 생성합니다. (예: {'gesture': 'A', 'gaze': -0.85})

이 결과값을 서버 내 전역 변수인 latestAIInput에 덮어씌웁니다.

게임 로직 처리 (Task 2: Game Core)
WebSocket 수신 루프와 완전히 별개로, asyncio 기반의 **서버측 "게임 루프"**가 20~30fps (예: await asyncio.sleep(0.05))로 돕니다.

이 루프는 매 틱(tick)마다 latestAIInput 변수를 확인합니다.

모든 게임 로직을 여기서 계산합니다:

latestAIInput.gesture == 'A'인가? -> Player.castSkill() 호출.

적 스포너가 적을 생성할 시간인가? -> Enemy 객체 생성.

Enemy 객체의 웨이포인트 이동 좌표 계산.

Player의 스킬 범위와 Enemy의 히트박스 충돌 판정.

적 체력 감소 및 사망 처리.

latestAIInput.gaze 값을 참조하여 특정 적을 타겟팅하거나 맵 기믹을 발동.

3. 📡 응답 (Server -> Client): "명령"이 아닌 "상태"
   해커톤에서 흔히 하는 실수가 "적 3번 제거" 같은 "명령(Command)"을 보내는 것입니다. 이는 네트워크 지연 시, 클라이언트와 서버의 상태가 엇갈리는 지름길입니다.

응답 형태: "전체 상태 동기화 (Full State Sync)"
서버의 20fps 게임 루프가 끝날 때마다, 현재 게임에 존재하는 모든 것의 상태를 담은 거대한 JSON 객체를 만듭니다.

이 JSON 객체를 WebSocket을 통해 클라이언트에 브로드캐스트합니다.
'''
{
"gameState": {
"enemies": [
{
"id": "enemy_1",
"typeId": "skeleton",
"x": 100,
"y": 200,
"currentHP": 50,
"maxHP": 80,
"animationState": "walk",
"currentFrame": 2,
"isDead": false
}
],
"effects": [
{
"id": "effect_1",
"type": "fireSlash",
"x": 300,
"y": 400
}
],
"gazePosition": { "x": 500, "y": 300 },
"playerGold": 100,
"playerScore": 500,
"waveNumber": 3
}
}
'''
