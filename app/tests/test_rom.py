from app.schemas.rom import RomAnalyzeRequest, FrameData, PoseData, Landmark, WorldLandmark
from app.services.rom_analysis import analyze_rom_from_frames

def test_analyze_rom_empty_frames():
    # Test that empty frames list raises ValueError
    request_data = RomAnalyzeRequest(
        session_id="test_session",
        measurement_type="shoulder",
        joint="left_shoulder",
        movement="abduction",
        frames=[]
    )
    try:
        analyze_rom_from_frames(request_data)
        assert False, "Should raise ValueError on empty frames"
    except ValueError as e:
        assert "측정 가능한 프레임 정보가 없습니다" in str(e)

def test_analyze_rom_simple():
    # Test with minimal valid pose landmarks
    landmark_dict = {
        "left_shoulder": Landmark(x=0.1, y=0.2, z=0.3, visibility=0.9),
        "left_elbow": Landmark(x=0.15, y=0.25, z=0.35, visibility=0.9),
        "left_hip": Landmark(x=0.1, y=0.5, z=0.3, visibility=0.9),
        "left_knee": Landmark(x=0.1, y=0.7, z=0.3, visibility=0.9),
        "left_ankle": Landmark(x=0.1, y=0.9, z=0.3, visibility=0.9),
        "left_foot_index": Landmark(x=0.12, y=0.92, z=0.3, visibility=0.9),
        "right_shoulder": Landmark(x=-0.1, y=0.2, z=0.3, visibility=0.9),
        "right_elbow": Landmark(x=-0.15, y=0.25, z=0.35, visibility=0.9),
        "right_hip": Landmark(x=-0.1, y=0.5, z=0.3, visibility=0.9),
        "right_knee": Landmark(x=-0.1, y=0.7, z=0.3, visibility=0.9),
        "right_ankle": Landmark(x=-0.1, y=0.9, z=0.3, visibility=0.9),
        "right_foot_index": Landmark(x=-0.12, y=0.92, z=0.3, visibility=0.9),
    }
    
    world_landmark_dict = {
        "left_shoulder": WorldLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9),
        "left_elbow": WorldLandmark(x=0.15, y=0.25, z=0.35, visibility=0.9),
        "left_hip": WorldLandmark(x=0.1, y=0.5, z=0.3, visibility=0.9),
        "left_knee": WorldLandmark(x=0.1, y=0.7, z=0.3, visibility=0.9),
        "left_ankle": WorldLandmark(x=0.1, y=0.9, z=0.3, visibility=0.9),
        "left_foot_index": WorldLandmark(x=0.12, y=0.92, z=0.3, visibility=0.9),
        "right_shoulder": WorldLandmark(x=-0.1, y=0.2, z=0.3, visibility=0.9),
        "right_elbow": WorldLandmark(x=-0.15, y=0.25, z=0.35, visibility=0.9),
        "right_hip": WorldLandmark(x=-0.1, y=0.5, z=0.3, visibility=0.9),
        "right_knee": WorldLandmark(x=-0.1, y=0.7, z=0.3, visibility=0.9),
        "right_ankle": WorldLandmark(x=-0.1, y=0.9, z=0.3, visibility=0.9),
        "right_foot_index": WorldLandmark(x=-0.12, y=0.92, z=0.3, visibility=0.9),
    }

    pose = PoseData(
        pose_index=0,
        landmarks=landmark_dict,
        world_landmarks=world_landmark_dict
    )

    frames = [
        FrameData(
            frame_index=0,
            timestamp_ms=500, # Neutral frame candidate
            detected=True,
            num_poses=1,
            image_width=640,
            image_height=480,
            poses=[pose]
        ),
        FrameData(
            frame_index=1,
            timestamp_ms=1200, # Max frame candidate
            detected=True,
            num_poses=1,
            image_width=640,
            image_height=480,
            poses=[pose]
        )
    ]

    request_data = RomAnalyzeRequest(
        session_id="test_session",
        measurement_type="shoulder",
        joint="left_shoulder",
        movement="abduction",
        frames=frames
    )

    result = analyze_rom_from_frames(request_data)
    assert result["joint"] == "left_shoulder"
    assert "rom_results" in result
    assert "left_shoulder" in result["rom_results"]
