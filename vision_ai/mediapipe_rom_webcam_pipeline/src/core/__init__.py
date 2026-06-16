"""
core/ — ROM 파이프라인 핵심 모듈 패키지

공개 API:
  from core.landmarks         import BlazePoseLandmark, BODY_CONNECTIONS, LEFT_LANDMARKS, RIGHT_LANDMARKS
  from core.landmark_extractor import create_landmarker, extract_frame_record, draw_landmarks_on_frame
  from core.angle_engine       import calculate_angle_3d, analyze_pose, VISIBILITY_THRESHOLD
  from core.visualizer         import build_angle_canvas, generate_pictographic_svg
"""
