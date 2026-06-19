# Tro-Fit Vision AI — 데이터 스키마 및 판단 기준 (Data Schema & Standards)

> **문서 목적:** 파이프라인에서 데이터를 어떻게 규격화(JSON 통일)했는지, 의학적 각도 판단은 어떤 근거(자료)를 통해 이루어지는지, 그리고 DB 연동 시 가장 중요한 **최종 출력 JSON의 상세 필드 해설**을 제공합니다.
> **수신자:** 조영진 팀원 (DB 연동 담당자)

---

## 1. 입·출력 데이터 형식의 통일 (JSON, Key-Value)

이 파이프라인은 최종적으로 외부 서비스(웹/앱/DB)와 통신해야 하므로, **가장 범용적이고 파싱하기 쉬운 JSON 포맷의 Dict(Key-Value)** 구조로 데이터를 완전히 통일했습니다.

### 구조 통일의 핵심: `landmarks.py`
과거 코드나 타 오픈소스에서는 랜드마크를 `landmarks[11]` 처럼 배열의 인덱스로 접근했습니다. 하지만 파이프라인에서는 이를 철저히 금지하고, 반드시 Dict 구조인 `landmarks["left_shoulder"]`로 접근하도록 구조를 뜯어고쳤습니다.

**왜 이렇게 했는가?**
1. **DB 스키마와의 1:1 매칭:** 인덱스 번호는 직관성이 떨어져 DB 컬럼 이름으로 쓸 수 없습니다. `json_key()`를 통해 강제로 `left_shoulder`라는 소문자 스네이크 케이스(Snake Case) 문자열을 생성하게 만듦으로써, 조영진 팀원이 설계할 DB 테이블의 컬럼명(예: `angle_left_shoulder`)과 완벽히 매칭되도록 설계했습니다.
2. **에러 원천 차단:** 배열 인덱싱은 런타임에러(IndexError) 위험이 큽니다. Python의 Dictionary `get()` 메서드를 통해 해당 관절이 가려져 추출되지 않았을 때 우아하게 `None`을 반환하도록 처리하기 위함입니다.

---

## 2. 관절 추출 축 및 각도 판단의 기준 자료

본 파이프라인은 측정된 관절 각도가 "정상인가 비정상인가"를 판단하기 위해 외부 하드코딩 수치를 쓰지 않고, 별도의 외부 설정 파일인 **`normal_rom.json`**을 단일 진실 공급원으로 사용합니다. (디렉토리 위치: `mediapipe_rom_webcam_pipeline/normal_rom.json`)

### 사용된 레퍼런스 자료
- **AMA(미국의학협회) 장애평가 기준**
- **이재학(1996) 국민연금공단 관절운동범위 표준각도**

### 어떻게 구현되었고, 왜 이렇게 했는가?
```json
// normal_rom.json 예시
"shoulder_abduction": {
  "normal_deg": 150,
  "landmarks": ["hip", "shoulder", "elbow"]
}
```
**구현 방식:**
1. 파이프라인 실행 시 `shoulder_abduction` 동작을 분석한다고 하면, 이 JSON 파일을 로드합니다.
2. `landmarks` 키에 적힌 `hip`, `shoulder`, `elbow`를 읽어들여, "아, 어깨 외전을 측정하려면 이 세 점을 이어서 벡터 내적 각도를 구해야 하는구나"라고 프로그램이 **동적으로 판단**합니다.
3. 계산된 각도를 `normal_deg`인 150도와 비교하여 달성 비율(%)을 구합니다.

**이유:** 기준 각도나 측정 방식을 파이썬 소스코드 내부에 하드코딩(`if movement == "abduction": return 150`)해버리면, 추후 재활의학과 전문의 피드백을 받아 기준각을 160도로 수정해야 할 때 파이썬 코드를 뜯어고치고 서버를 재배포해야 합니다. 이를 JSON 파일로 외부화(Externalization)함으로써, **비개발자(기획자나 의료진)도 JSON 수치만 바꾸면 즉시 전체 AI 판단 기준이 변경되도록** 아키텍처를 설계한 것입니다.

---

## 3. 최종 출력 JSON (`rom_result.json`) 완벽 해석 코멘트

파이프라인 실행이 완료되면 DB 연동을 위해 생성되는 최종 결과물입니다. 조영진 팀원님은 이 JSON을 파싱하여 DB 쿼리를 작성하시면 됩니다.

```json
{
  "joint": "shoulder",                  // [DB 매핑] 상위 부위 카테고리 (예: category_cd)
  "movement": "abduction",              // [DB 매핑] 세부 동작 카테고리 (예: movement_cd)
  "video_file": "shoulder_test.mp4",    // 원본 비디오 파일명 (이력 추적용)
  "video_info": {
    "fps": 30.0,
    "duration_s": 4.133
  },
  "measurement": {
    // 알고리즘이 분석을 위해 비디오에서 추출한 프레임 위치 정보
    "neutral_captured_at": "0.500s (frame #15)",
    "max_selected": "3.633s (frame #109)",
    "use_world_landmarks": true,        // true면 3D 미터 좌표 기반 측정 (신뢰도 높음)
    "visibility_threshold": 0.65
  },
  
  // ▼ 핵심 1: 절대 측정 수치 (실제 측정된 각도들) ▼
  "rom_results": {
    "left_shoulder": {
      "neutral_angle": 175.3,           // [DB 매핑] 시작 시점(중립)의 관절 각도 (Float)
      "max_angle": 62.7,                // [DB 매핑] 최대 수축 시점의 관절 각도 (Float)
      "rom": 112.6,                     // [DB 매핑] 최대각 - 중립각 차이의 절대값 = 실제 움직인 범위 (Float)
      "reliable": true                  // 측정 신뢰도 (만약 false라면 해당 관절이 화면 밖으로 벗어난 것임. DB Insert 시 제외 권장)
    }
  },
  
  // ▼ 핵심 2: 의료적 평가 수치 (normal_rom.json 기반 분석치) ▼
  "rom_ratio": {
    "left_shoulder": {
      "rom_deg": 112.6,                 // 위에서 계산된 실제 움직임 범위
      "normal_deg": 150.0,              // 일반인 정상 가동 범위 기준치
      "rom_ratio_pct": 75.1,            // [DB 매핑] 정상 범위 대비 현재 가동 능력 백분율 (112.6 / 150.0 * 100)
      "grade": "정상 범위"              // [DB 매핑] 환자에게 보여줄 등급 텍스트 (예: 정상/경도/중등도/고도 제한)
    }
  },
  "confidence": "HIGH"                  // 프레임 전반적인 측정 신뢰도 총평
}
```

### DB 연동을 위한 조언 (Action Item)
1. `rom_results` 배열 하위의 `reliable` 값이 `false`인 경우, 해당 관절 데이터는 신뢰할 수 없는 쓰레기값이므로 DB 저장을 스킵하시거나 `NULL` 처리하시는 것을 강력히 권장합니다.
2. 앱/웹 프론트엔드에 "고객님의 현재 어깨 가동범위는 정상인의 75% 수준입니다"라고 띄워주기 위해, `rom_ratio_pct` 컬럼과 `grade` 컬럼을 DB에 만들어 두시면 프론트엔드 연동이 매우 수월해질 것입니다.
