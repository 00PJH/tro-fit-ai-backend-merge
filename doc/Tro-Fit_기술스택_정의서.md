# Tro-Fit 기술 스택 정의서
**프로젝트명**: Tro-Fit — AI 기반 트로트 맞춤형 노인 근력 운동 솔루션  
**팀명**: 실마리  
**기준 환경**: 로컬(Local) 개발 환경 (AWS 클라우드 미사용)  
**작성일**: 2026년 6월

---

## 🗺️ 전체 시스템 아키텍처 개요 (STEP 1~6 비동기 흐름)

```
[스마트폰 앱 (React Native) / 사용자 기기]
  │
  ├── [STEP 1] FMS 체력 평가 (5~7가지 동작 가이드)
  ▼
[Vision AI 모듈 (On-device TFLite / 비동기 진단)]
  ├── [STEP 2] MediaPipe/RTMPose 기반 33개 관절 3D 좌표 추출 (영상은 즉시 파기)
  └── 관절 가동 범위(ROM) 분석 데이터 (JSON) 생성
  │
  │ JSON 분석 결과 전달 (REST API)
  ▼
[Backend API 서버 (FastAPI / 로컬 Docker)]
  ├── 사용자 데이터 및 누적 ROM 정보 저장 (PostgreSQL)
  └── [STEP 3] RAG + LLM 맞춤 안무 설계 및 초경량 JSON 생성 (무릎/허리 부하 차단)
  │
  │ 맞춤 안무 JSON & Lottie 캐릭터 애니메이션 데이터 송출
  ▼
[TV 화면 출력 (Chromecast 캐스팅 / HDMI 미러링 연동)]
  └── [STEP 4] TV 대화면 운동 실행 
       ├── 음악 재생 + Lottie 캐릭터 안무 시연 (TV)
       ├── 하단 피토그래픽 동작 가이드 실시간 전환
       └── 폰은 테이블에 내려두고 컨트롤러로만 사용 (발열 및 인지 피로 해결)
  │
  ├── [STEP 5] 운동 완료 후 데이터 피드백 및 결과 기록
  ▼
[보호자 알림 (카카오톡 알림톡 링크)]
  └── [STEP 6] 주간 리포트 웹페이지 (앱 설치 없이 브라우저로 ROM 변화 추적)
```

---

## 1. 📱 Frontend (모바일 앱)

### 핵심 목적
- 주 1회 자세 진단용 카메라 UI
- 안무 결과 확인 및 건강 리포트 시각화
- TV 캐스팅 연동 컨트롤

### 기술 스택

| 항목 | 기술 | 선택 이유 |
|------|------|----------|
| **프레임워크** | **React Native (Expo)** | iOS/Android 동시 지원, 빠른 프로토타이핑, 카메라/미디어 API 풍부 |
| **언어** | **TypeScript** | 타입 안전성, 유지보수성 향상 |
| **UI 컴포넌트** | **React Native Paper** 또는 **NativeBase** | 노인 친화적 큰 글씨/버튼 UI 구성 용이 |
| **상태 관리** | **Zustand** 또는 **Redux Toolkit** | 간단한 전역 상태 (사용자 정보, 주간 데이터) |
| **API 통신** | **Axios** + **React Query (TanStack Query)** | API 캐싱, 로딩 상태 자동 처리 |
| **네비게이션** | **React Navigation v6** | 표준 스택/탭 내비게이션 |
| **카메라 접근** | **Expo Camera** / **Vision Camera** | 실시간 카메라 스트림 및 FMS 평가를 위한 고성능 프레임 프로세서 지원 |
| **TV 캐스팅** | **React Native Cast (Chromecast SDK)** | 안드로이드 TV, Chromecast 연동 |
| **로컬 저장** | **AsyncStorage** / **MMKV** | 앱 내 사용자 설정, 캐시 데이터 저장 |
| **콘텐츠 재생** | **lottie-react-native** + **Expo AV** | Lottie 캐릭터 안무 애니메이션 렌더링 및 트로트 오디오(MP3) 재생 |
| **미러링 제어** | **expo-keep-awake** | HDMI 미러링 연결 시 화면 꺼짐 방지(Wake Lock) 제어 |

#### 주요 화면 및 기능
```
📱 앱 화면 구조
├── 홈 화면: 주간 운동 현황, 오늘 안무 시작 버튼
├── FMS 평가 화면: 5~7가지 동작 지시 및 카메라 촬영 → Vision AI 분석 및 가동범위 측정
├── 안무 플레이어: Lottie 캐릭터 안무 & 피토그래픽 렌더링 (HDMI 미러링/Chromecast 캐스팅 제어)
├── 건강 리포트: 주간 ROM 변화 그래프, 관절 상태 트렌드
└── 설정: 선호 트로트 음악, 운동 시간대 알림 설정
```

---

## 2. 📺 Frontend (스마트TV 앱)

### 핵심 목적
- 큰 화면으로 안무 영상을 보며 운동
- 카메라 없이 따라하기 → 주중 매일 사용

### 기술 스택

| 항목 | 기술 | 선택 이유 |
|------|------|----------|
| **LG WebOS 앱** | **Web App (React + webOS SDK)** | LG TV 앱 스토어 배포 표준 방식, HTML5 기반 |
| **Samsung Tizen 앱** | **Tizen Web App (React + Tizen SDK)** | 삼성 TV 표준 개발 방식 |
| **공통 코어** | **React** (웹 기반 TV 앱 공통화) | 모바일 앱과 코드 일부 공유 가능 |
| **콘텐츠 재생** | **lottie-web** + **HTML5 Audio API** | TV 환경 최적화 Lottie 애니메이션 및 트로트 음악 오디오 재생 |
| **리모컨 키 이벤트** | **TV 플랫폼 Key Event API** | 리모컨 D-pad 네비게이션 처리 |
| **폰-TV 연동** | **Chromecast Receiver SDK** | 스마트폰에서 TV로 안무 JSON 데이터 수신 및 재생 연동 |

#### 대안 (HDMI 연결 방식)
> TV 앱 개발이 어려운 경우: **스마트폰 → HDMI 어댑터(C to HDMI) → TV 미러링**으로 MVP 구현 가능

---

## 3. 🤖 Vision AI 모듈

### 핵심 목적
- 주 1회 스마트폰 카메라를 활용한 동작 기반 기능 평가 (FMS 방식) 진행
- 33개 관절 3D 좌표 추출, 주요 신체 동작의 ROM(관절가동범위) 계산 및 낙상위험 점수(FALLS Score) 산출

### 기술 스택

| 항목 | 기술 | 선택 이유 |
|------|------|----------|
| **포즈 추정 (서버/로컬)** | **MediaPipe Tasks API — PoseLandmarker** (Python, `pose_landmarker_full.task`) | Google 공식 최신 API (v0.10.31+), 33개 관절 3D 추정, `.task` 파일이 TFLite 기반으로 모바일 배포에 직접 재사용 가능 |
| **포즈 추정 (대안, Phase 2)** | **RTMPose** via `rtmlib` (ONNX Runtime) | 더 높은 정확도, GPU 환경에서 우위. Phase 1에서 mmpose 의존성 이슈로 Phase 2로 이관 |
| **On-device 경량화** | **MediaPipe `.task` 파일 (TFLite 기반)** + INT8 양자화 | Tasks API의 `.task` 파일이 TFLite 포맷이므로 별도 변환 없이 React Native TFLite 인터프리터에서 직접 사용 가능 |

> **[API 변경 이력 - 2026.06.04]** mediapipe 0.10.31 이후 `mp.solutions.pose` 레거시 API가 공식 제거됨.
> 현재 코드베이스는 전체 `mediapipe.tasks.python.vision.PoseLandmarker` (Tasks API)로 마이그레이션 완료.
> 구버전 다운그레이드를 하지 않은 이유: numpy ≥ 2.0 호환성, Python 3.11 공식 지원, TFLite 모바일 배포 연계성.
| **ROM 분석 알고리즘** | **Python (NumPy, SciPy)** | 관절 각도 계산, 좌우 대칭 분석 |
| **낙상 위험도 모델** | **Scikit-learn / ONNX** | FALLS Score 분류 모델, 경량 ML 사용 |
| **Federated Learning (2단계)** | **Flower (flwr)** | 개인정보 보호형 분산 학습 프레임워크 |
| **배터리 최적화 (2단계)** | **Adaptive Compute 패턴** | 디바이스 상태에 따라 추론 해상도/빈도 자동 조절 |
| **영상 처리** | **OpenCV** | 프레임 전처리, 영상 크기 조정 |
| **개발 언어** | **Python 3.10+** | AI/ML 생태계 표준 |

#### Vision AI 처리 흐름
```
스마트폰 카메라 FMS 평가 촬영 (5~7가지 동작 가이드)
    │
    ▼
[On-device TFLite 추론] → 관절 좌표 메타데이터 JSON 생성 (영상은 즉시 삭제)
    │
    ▼
[FastAPI 서버 수신] → ROM 분석 알고리즘 실행
    │
    ├── 33개 관절 각도 계산
    ├── 좌우 비대칭 탐지
    ├── FALLS Score 산출
    └── 허리/무릎/어깨 등 부위별 ROM 범위 측정
    │
    ▼
[주간 리포트 데이터 업데이트] + [맞춤 안무 생성 요청]
```

---

## 4. ⚙️ Backend (API 서버)

### 핵심 목적
- 사용자 데이터 관리, AI 안무 생성, 건강 리포트 제공
- 로컬 환경에서 Docker로 실행

### 기술 스택

| 항목 | 기술 | 선택 이유 |
|------|------|----------|
| **API 프레임워크** | **FastAPI (Python)** | 비동기(async) 지원, 자동 OpenAPI 문서, 빠른 개발 속도 |
| **언어** | **Python 3.10+** | AI 라이브러리 생태계와 통일 |
| **비동기 처리** | **Celery + Redis** | 안무 생성같은 오래 걸리는 작업 비동기 큐 처리 |
| **준실시간 통신** | **WebSocket (FastAPI WebSocket)** | 운동 중 실시간 피드백 (2단계) |
| **ORM** | **SQLAlchemy 2.0** + **Alembic** | DB 마이그레이션, 타입 안전 쿼리 |
| **데이터 검증** | **Pydantic v2** | 요청/응답 스키마 검증, FastAPI 기본 통합 |
| **인증** | **JWT (python-jose)** | 토큰 기반 사용자 인증 |
| **보안 점검** | **Bandit** (정적 분석) + **OWASP ZAP** (로컬 실행) | 로컬 보안 취약점 점검 |

### 로컬 인프라 스택

| 항목 | 기술 | 원본 (AWS) | 로컬 대체 |
|------|------|-----------|----------|
| **서버 실행** | **로컬 PC / Docker** | AWS EC2 | uvicorn 직접 실행 또는 Docker |
| **관계형 DB** | **PostgreSQL 16 (Docker)** | AWS RDS | `docker run postgres:16` |
| **벡터 DB** | **ChromaDB (로컬 프로세스)** | 클라우드 벡터 DB | Python 프로세스 또는 Docker |
| **캐시 / 메시지 브로커** | **Redis (Docker)** | ElastiCache | `docker run redis:7` |
| **파일 스토리지** | **MinIO (Docker)** | AWS S3 | S3 호환 API 제공, 로컬 스토리지 |
| **모니터링** | **Prometheus + Grafana (Docker)** | AWS CloudWatch | docker-compose로 함께 실행 |
| **부하 테스트** | **Locust (로컬 실행)** | AWS 성능 테스트 | `locust -f locustfile.py` |

#### Docker Compose 구성 예시
```yaml
# docker-compose.yml (로컬 전체 스택)
services:
  api:          # FastAPI 서버
  postgres:     # PostgreSQL DB
  redis:        # Celery 브로커 + 캐시
  chromadb:     # 벡터 DB (안무 RAG)
  minio:        # S3 호환 파일 저장소
  prometheus:   # 메트릭 수집
  grafana:      # 모니터링 대시보드
  celery:       # 비동기 워커
```

---

## 5. 🧠 AI / LLM (안무 생성)

### 핵심 목적
- ROM 분석 결과를 바탕으로 맞춤형 트로트 안무를 자동 생성
- 주 1회 업데이트 (비동기 배치 처리)

### 기술 스택

| 항목 | 기술 | 선택 이유 |
|------|------|----------|
| **LLM 기반** | **Ollama (로컬 LLM 실행)** + **LLaMA 3 / Gemma** | 로컬 환경에서 GPU 없이도 실행 가능, API 비용 0원 |
| **LLM API 대안** | **OpenAI API** (gpt-4o) / **Google Gemini API** | 로컬 LLM 성능 부족 시 외부 API 활용 |
| **RAG 파이프라인** | **LangChain** + **ChromaDB** | 안무 동작 DB 검색 후 LLM에 컨텍스트 제공 |
| **임베딩** | **sentence-transformers** (로컬) | 동작 설명 → 벡터 임베딩, 무료 |
| **Fine-tuning (2단계)** | **LoRA / QLoRA (PEFT)** | 소규모 파인튜닝, 로컬 GPU 또는 Google Colab 활용 |
| **안무 데이터** | **JSON 구조화 동작 DB** | 검증된 노인 운동 동작 라이브러리 구축 |
| **미디어 전처리** | **FFmpeg (백엔드 툴)** | 트로트 음원 인코딩 및 쿨다운 가이드 비디오 전처리 등 리소스 관리 |
| **시계열 이상탐지 (2단계)** | **Prophet / LSTM (PyTorch)** | 주간 ROM 데이터 이상 패턴 탐지, 낙상 예조 |

#### 안무 생성 흐름 (비동기)
```
[ROM 분석 결과 수신]
    │
    ▼
[ChromaDB 검색] → 몸 상태에 맞는 안전한 동작 후보 검색 (무릎/허리 과부하 동작 필터링)
    │
    ▼
[LLM (LangChain)] → "이 사용자의 허리 ROM이 45도로 제한됨.
                     트로트 '사랑의 배터리'에 맞춰
                     안전한 7가지 동작 시퀀스를 조합해줘"
    │
    ▼
[안무 JSON 생성] → 동작 순서, 박자, Lottie 애니메이션 및 피토그래픽 매핑 메타데이터
    │
    ▼
[MinIO 저장] → 안무 JSON 메타데이터를 저장소에 업로드하고 사용자 앱/Chromecast에 다운로드 URL 제공
```

---

## 6. 🔧 DevOps / 인프라 (로컬 환경)

### 핵심 목적
- 팀 전체가 동일한 로컬 개발 환경 유지
- 자동화된 빌드/테스트 파이프라인

### 기술 스택

| 항목 | 기술 | 역할 |
|------|------|------|
| **컨테이너화** | **Docker Desktop** + **Docker Compose** | 전체 백엔드 스택 로컬 실행 통일 |
| **CI/CD** | **GitHub Actions** | Push 시 자동 테스트, Docker 이미지 빌드 |
| **코드 관리** | **Git** + **GitHub** | 브랜치 전략 (main/dev/feature/*) |
| **코드 품질** | **Black, isort, Flake8** (Python) | 자동 코드 포맷팅 |
| **테스트** | **Pytest** (Backend) + **Jest** (Frontend) | 단위/통합 테스트 |
| **부하 테스트** | **Locust** | 로컬 API 서버 성능 측정 |
| **보안 분석** | **Bandit** (정적) + **OWASP ZAP** (동적, 로컬) | 보안 취약점 점검 |
| **모니터링** | **Prometheus** + **Grafana** | 로컬 Docker로 실행, API 메트릭 시각화 |
| **환경 변수 관리** | **.env 파일** + **python-dotenv** | 로컬 시크릿 관리 (gitignore 처리) |

---

## 7. 📊 역할별 기술 스택 요약

### 👤 박준형 (PM & Vision AI)
```
담당 기술:
├── Python 3.10+
├── MediaPipe Pose / BlazePose / RTMPose
├── TensorFlow Lite (INT8 양자화)
├── OpenCV (영상 처리)
├── NumPy, SciPy (관절 각도 계산)
├── Scikit-learn / ONNX (FALLS Score 모델)
├── Flower (Federated Learning, 2단계)
└── 프로젝트 관리 (WBS, 발표 자료)
```

### 👤 한상협 (Cloud/인프라 → 로컬인프라 & Frontend)
```
담당 기술:
├── Docker Desktop + Docker Compose (전체 로컬 인프라)
├── MinIO (S3 대체, 로컬 파일 스토리지)
├── Prometheus + Grafana (로컬 모니터링)
├── GitHub Actions (CI/CD)
├── React Native + Expo (모바일 앱 UI/UX)
├── TypeScript
├── React (WebOS / Tizen TV 앱 & 보호자 웹 리포트)
├── Expo Camera / Vision Camera (FMS 평가 모듈)
├── lottie-react-native (모바일 앱 Lottie 재생)
├── expo-keep-awake (HDMI 미러링 화면 꺼짐 방지)
└── Chromecast SDK (TV 캐스팅 연동)
```

### 👤 이태균 (Backend & LLM)
```
담당 기술:
├── Python / FastAPI
├── PostgreSQL 16 (Docker)
├── SQLAlchemy 2.0 + Alembic
├── Redis + Celery (비동기 큐)
├── ChromaDB (로컬 벡터 DB)
├── LangChain + LLM (Ollama 또는 OpenAI API)
├── sentence-transformers (임베딩)
├── FFmpeg (오디오 인코딩 및 미디어 전처리)
├── WebSocket (준실시간 통신)
├── JWT 인증
├── Bandit + OWASP ZAP (보안 점검)
└── Locust (부하 테스트)
```

### 👤 조영진 (설계 & QA)
```
담당 기술:
├── 시스템 아키텍처 설계 (Async 분리 구조)
├── API 명세서 (OpenAPI / Swagger)
├── QA 테스트 시나리오 작성
├── 통합 테스트 (Pytest)
└── 발표 자료 총괄
```

---

## 8. 🔨 기능별 기술 스택 매핑

| 기능 | 관련 기술 스택 |
|------|--------------|
| **주 1회 자세 진단** | Expo Camera → FMS 평가 시퀀스 안내 → TFLite On-device → MediaPipe Pose → ROM 알고리즘 (NumPy) |
| **낙상 위험도 산출** | FALLS Score 모델 (Scikit-learn/ONNX) → FastAPI → PostgreSQL 저장 |
| **맞춤 안무 생성** | ChromaDB RAG 검색 → LangChain + LLM → 안무 JSON 메타데이터 생성 → MinIO 저장 |
| **TV 연동 재생** | React Native (Cast SDK) → Chromecast / HDMI 미러링 최적화(expo-keep-awake, DND) → TV 화면 렌더링 |
| **건강 리포트** | PostgreSQL 시계열 데이터 → FastAPI 집계 → 보호자 웹 페이지 (React + Recharts/Chart.js) |
| **이상 징후 탐지** | PostgreSQL ROM 데이터 → Prophet/LSTM → 알림 푸시 (Expo Notifications) |
| **비동기 안무 생성** | Celery + Redis → LLM 처리 → 안무 JSON 생성 및 MinIO 저장 → WebSocket 완료 알림 |
| **로컬 모니터링** | FastAPI 메트릭 → Prometheus 수집 → Grafana 대시보드 |
| **CI/CD 자동화** | GitHub Actions → Docker 빌드 → Pytest/Jest 실행 → Docker Compose 재시작 |
| **보안 점검** | Bandit (Python 정적 분석) + OWASP ZAP (API 취약점 스캔, 로컬 실행) |

---

## 9. 🚀 로컬 개발 환경 설정 가이드

### 필수 설치 도구
```bash
# 1. Docker Desktop 설치
# https://www.docker.com/products/docker-desktop/

# 2. Node.js 20+ (React Native)
nvm install 20

# 3. Python 3.10+ 
pyenv install 3.10.14

# 4. Expo CLI (모바일 앱 개발)
npm install -g @expo/cli

# 5. Git + GitHub 설정
git config --global user.name "팀원 이름"
```

### 로컬 백엔드 스택 실행
```bash
# 전체 스택 한번에 실행
docker-compose up -d

# 확인
docker-compose ps

# API 문서 확인 (자동 생성)
open http://localhost:8000/docs

# Grafana 모니터링
open http://localhost:3000

# MinIO 파일 스토리지 콘솔
open http://localhost:9001
```

### 모바일 앱 개발
```bash
cd frontend
npm install
npx expo start

# iOS 시뮬레이터
npx expo run:ios

# Android 에뮬레이터
npx expo run:android
```

---

## 10. ⚠️ 기술 리스크 및 대응 방안

| 리스크 | 대응 방안 |
|--------|----------|
| 로컬 PC 성능 부족 (LLM 추론) | Ollama 경량 모델 (Gemma:2b) 또는 OpenAI API 유료 전환 |
| TFLite 모델 정확도 저하 | 서버에서 MediaPipe 풀 모델 처리 후 결과만 전송 (하이브리드) |
| TV 앱 개발 복잡성 | MVP에서는 Chromecast 미러링으로 대체, TV 앱은 2단계 진행 |
| 팀원 간 환경 불일치 | Docker Compose로 완전 동일 환경 보장 |
| Federated Learning 구현 난이도 | 2단계에서 시도, 실패 시 중앙 집중형 학습으로 대체 |
| LLM 안무 생성 품질 | Rule-based 필터 + LLM 하이브리드, 전문가 검증 데이터셋 활용 |

---

*본 문서는 WBS 및 개발계획서(2026 피우다프로젝트)를 기반으로 작성되었습니다.*  
*AWS 클라우드 환경은 최종 서비스 출시 단계에서 별도 마이그레이션 계획을 통해 적용 예정입니다.*
