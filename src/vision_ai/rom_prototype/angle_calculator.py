"""
Tro-Fit Vision AI — 관절 각도 계산 핵심 모듈
=============================================
MediaPipe Tasks API (PoseLandmarker) 기반 좌표 데이터와 호환됩니다.
순수 수학 연산(NumPy)만 사용하므로 API 버전에 무관하게 동작합니다.
"""
import numpy as np


def calculate_angle_3d(point_a, point_b, point_c) -> float:
    """
    3D 공간에서 세 점 A, B, C가 이루는 사잇각(도, Degree)을 계산합니다.
    기준점은 B(point_b)입니다. (예: 어깨 각도를 구하려면 A: 손목, B: 어깨, C: 골반)

    Args:
        point_a (list | tuple | np.ndarray): [x, y, z] 형태의 좌표 (상위 관절)
        point_b (list | tuple | np.ndarray): [x, y, z] 형태의 좌표 (기준 관절, 꼭짓점)
        point_c (list | tuple | np.ndarray): [x, y, z] 형태의 좌표 (하위 관절)

    Returns:
        float: B점을 기준으로 이루는 각도 (0 ~ 180도)
    """
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    c = np.array(point_c, dtype=np.float64)

    # 두 벡터 생성 (BA, BC)
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # 0 분모 에러 방지 (랜드마크 좌표가 동일한 경우)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0

    # 내적 계산 → 코사인 → 아크코사인
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # 부동소수점 오차로 [-1, 1] 범위를 벗어나는 경우 클리핑
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))


if __name__ == "__main__":
    # ── 단위 테스트 ────────────────────────────────────────────────
    test_cases = [
        {
            "name": "직각(90도) 테스트",
            "a": [0, 1, 0], "b": [0, 0, 0], "c": [1, 0, 0],
            "expected": 90.0,
        },
        {
            "name": "일직선(180도) 테스트",
            "a": [0, 1, 0], "b": [0, 0, 0], "c": [0, -1, 0],
            "expected": 180.0,
        },
        {
            "name": "동일점(0도) 에러 방지 테스트",
            "a": [0, 0, 0], "b": [0, 0, 0], "c": [1, 0, 0],
            "expected": 0.0,
        },
        {
            "name": "45도 테스트",
            "a": [1, 1, 0], "b": [0, 0, 0], "c": [1, 0, 0],
            "expected": 45.0,
        },
    ]

    print("=" * 50)
    print("  calculate_angle_3d() 단위 테스트")
    print("=" * 50)
    all_pass = True
    for tc in test_cases:
        result = calculate_angle_3d(tc["a"], tc["b"], tc["c"])
        passed = np.isclose(result, tc["expected"], atol=0.1)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} | {tc['name']}: expected={tc['expected']} deg, got={result:.4f} deg")
        if not passed:
            all_pass = False

    print("=" * 50)
    print(f"  Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 50)
