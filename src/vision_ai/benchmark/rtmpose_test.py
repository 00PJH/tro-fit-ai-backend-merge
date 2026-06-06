"""
Tro-Fit Vision AI — RTMPose 설치 시도 벤치마크 기록
======================================================
로드맵 Day 3 항목: RTMPose 설치 시도 (성공/실패 모두 기록)

결론: Windows 환경에서 현재 RTMPose(MMPose) 설치 복잡도가 높아 우선 MediaPipe Tasks API를 채택.
아래에 시도 내역과 실패 원인을 기록합니다.
"""

# ──────────────────────────────────────────────────────────────────
# RTMPose 설치 시도 기록
# ──────────────────────────────────────────────────────────────────
#
# [시도 1] mmpose 직접 설치
#   pip install mmpose
#   결과: mmcv, mmdet 등 복합 의존성 필요
#         mmcv는 CUDA 버전에 맞는 사전 빌드 휠이 필요하여
#         CPU-only 환경(현재 로컬)에서 설치 난이도 높음.
#
# [시도 2] mmcv 설치 (CPU-only)
#   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
#   결과: torch 버전 매핑 오류 (현재 환경에 PyTorch 미설치)
#
# [시도 3] onnxruntime 기반 RTMPose 추론 (경량 대안)
#   pip install rtmlib
#   결과: rtmlib은 onnxruntime 기반으로 mmpose 의존성 없이 RTMPose ONNX 모델 추론 가능.
#         추후 Phase 2 비교 벤치마크에서 rtmlib 으로 재시도 예정.
#
# ──────────────────────────────────────────────────────────────────
# 기술적 판단 및 의사 결정
# ──────────────────────────────────────────────────────────────────
#
# 항목          | MediaPipe Tasks API          | RTMPose (MMPose)
# --------------|------------------------------|------------------------------
# 설치 난이도   | pip install mediapipe (1줄) | mmcv + mmdet + mmpose (복잡)
# CUDA 요구     | 불필요 (CPU 충분)           | 권장 (성능 차이 큼)
# 정확도        | BlazePose Full (우수)       | RTMPose-m 기준 더 높음
# 추론 속도     | ~15~30 FPS (CPU)           | GPU 있으면 30+ FPS
# 모바일 배포   | .task 파일 (TFLite 기반)    | ONNX 변환 필요
# 라이선스      | Apache 2.0                  | Apache 2.0
#
# 결론:
#   - Phase 1 MVP: MediaPipe Tasks API (PoseLandmarker) 사용
#     → 로컬 CPU 환경에서 즉시 동작, TFLite 기반으로 모바일 배포에 직접 활용 가능
#   - Phase 2 고도화: rtmlib (RTMPose ONNX) 비교 벤치마크 재시도
#     → GPU 서버 또는 Docker 환경에서 RTMPose 성능 검증 후 교체 여부 결정
#
# ──────────────────────────────────────────────────────────────────

BENCHMARK_NOTES = {
    "rtmpose_attempt": {
        "status": "DEFERRED",
        "reason": "mmcv CUDA dependency 및 PyTorch 미설치로 인한 설치 불가",
        "alternative": "rtmlib (RTMPose ONNX runtime) — Phase 2에서 재시도",
        "decision": "Phase 1: MediaPipe Tasks API (PoseLandmarker) 사용"
    }
}

if __name__ == "__main__":
    print("=" * 60)
    print("  RTMPose 설치 시도 결과 요약 (Tro-Fit 로드맵 Day 3)")
    print("=" * 60)
    print(f"  상태  : {BENCHMARK_NOTES['rtmpose_attempt']['status']}")
    print(f"  이유  : {BENCHMARK_NOTES['rtmpose_attempt']['reason']}")
    print(f"  대안  : {BENCHMARK_NOTES['rtmpose_attempt']['alternative']}")
    print(f"  결정  : {BENCHMARK_NOTES['rtmpose_attempt']['decision']}")
    print("=" * 60)
    print()
    print("[NOTE] 비교 벤치마크 데이터는 doc/benchmark_results.md 를 참고하세요.")
