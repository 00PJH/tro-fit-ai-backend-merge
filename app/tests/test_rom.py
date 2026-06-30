from app.schemas.rom import RomAnalyzeRequest, FrameData, PoseData, Landmark, WorldLandmark
from app.services.rom_analysis import analyze_rom_from_frames
from app.models.rom import RomHistory

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
    assert "mobility_analysis" in result
    assert len(result["mobility_analysis"]) == 1
    assert result["mobility_analysis"][0]["side"] == "left"
    assert "mobility_score" in result["mobility_analysis"][0]


# ==============================================================================
# API End-to-End Tests
# ==============================================================================
from fastapi.testclient import TestClient
from app.main import app
from app.api.v1.endpoints.rom import get_current_user_optional
from app.models.user import User

client = TestClient(app)

def test_api_analyze_rom_endpoint_anonymous():
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
            timestamp_ms=500,
            detected=True,
            num_poses=1,
            image_width=640,
            image_height=480,
            poses=[pose]
        ),
        FrameData(
            frame_index=1,
            timestamp_ms=1200,
            detected=True,
            num_poses=1,
            image_width=640,
            image_height=480,
            poses=[pose]
        )
    ]
    payload = {
        "session_id": "test_endpoint_session",
        "measurement_type": "shoulder",
        "joint": "left_shoulder",
        "movement": "abduction",
        "frames": [f.model_dump() for f in frames]
    }

    app.dependency_overrides.clear()
    
    response = client.post("/api/v1/analyze/rom", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_endpoint_session"
    assert data["week_number"] == 1
    assert "measured_at" in data
    assert data["joint"] == "left_shoulder"
    assert data["movement"] == "abduction"
    assert "mobility_analysis" in data


def test_api_analyze_rom_endpoint_authenticated_history():
    from app.db.session import SessionLocal
    db = SessionLocal()
    db.query(RomHistory).filter(RomHistory.user_id == "test_user_E2E").delete()
    db.query(User).filter(User.user_id == "test_user_E2E").delete()
    db.commit()

    mock_user = User(
        user_id="test_user_E2E",
        user_email="e2e@example.com",
        password_hash="dummy_hash",
        salt="dummy_salt",
        user_name="Test User",
        user_age=25,
        user_height=175.0,
        user_weight=70.0
    )
    db.add(mock_user)
    db.commit()

    app.dependency_overrides[get_current_user_optional] = lambda: mock_user

    try:
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
                timestamp_ms=500,
                detected=True,
                num_poses=1,
                image_width=640,
                image_height=480,
                poses=[pose]
            ),
            FrameData(
                frame_index=1,
                timestamp_ms=1200,
                detected=True,
                num_poses=1,
                image_width=640,
                image_height=480,
                poses=[pose]
            )
        ]
        payload1 = {
            "session_id": "session_week_1",
            "measurement_type": "shoulder",
            "joint": "left_shoulder",
            "movement": "abduction",
            "frames": [f.model_dump() for f in frames]
        }

        # First request should save history and return week_number = 1
        response1 = client.post("/api/v1/analyze/rom", json=payload1)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["week_number"] == 1

        # Get history
        response_hist = client.get("/api/v1/analyze/rom/history")
        assert response_hist.status_code == 200
        hist_data = response_hist.json()
        assert "results" in hist_data
        assert len(hist_data["results"]) >= 1
    finally:
        app.dependency_overrides.clear()
        db.query(RomHistory).filter(RomHistory.user_id == "test_user_E2E").delete()
        db.query(User).filter(User.user_id == "test_user_E2E").delete()
        db.commit()
        db.close()


def test_api_analyze_rom_validation_failure_422():
    # High threshold visibility = 0.9, but landmark has 0.1 visibility.
    landmark_dict = {
        "left_shoulder": Landmark(x=0.1, y=0.2, z=0.3, visibility=0.1),
        "left_elbow": Landmark(x=0.15, y=0.25, z=0.35, visibility=0.1),
        "left_hip": Landmark(x=0.1, y=0.5, z=0.3, visibility=0.1),
        "left_knee": Landmark(x=0.1, y=0.7, z=0.3, visibility=0.1),
        "left_ankle": Landmark(x=0.1, y=0.9, z=0.3, visibility=0.1),
        "left_foot_index": Landmark(x=0.12, y=0.92, z=0.3, visibility=0.1),
        "right_shoulder": Landmark(x=-0.1, y=0.2, z=0.3, visibility=0.1),
        "right_elbow": Landmark(x=-0.15, y=0.25, z=0.35, visibility=0.1),
        "right_hip": Landmark(x=-0.1, y=0.5, z=0.3, visibility=0.1),
        "right_knee": Landmark(x=-0.1, y=0.7, z=0.3, visibility=0.1),
        "right_ankle": Landmark(x=-0.1, y=0.9, z=0.3, visibility=0.1),
        "right_foot_index": Landmark(x=-0.12, y=0.92, z=0.3, visibility=0.1),
    }
    world_landmark_dict = {
        "left_shoulder": WorldLandmark(x=0.1, y=0.2, z=0.3, visibility=0.1),
        "left_elbow": WorldLandmark(x=0.15, y=0.25, z=0.35, visibility=0.1),
        "left_hip": WorldLandmark(x=0.1, y=0.5, z=0.3, visibility=0.1),
        "left_knee": WorldLandmark(x=0.1, y=0.7, z=0.3, visibility=0.1),
        "left_ankle": WorldLandmark(x=0.1, y=0.9, z=0.3, visibility=0.1),
        "left_foot_index": WorldLandmark(x=0.12, y=0.92, z=0.3, visibility=0.1),
        "right_shoulder": WorldLandmark(x=-0.1, y=0.2, z=0.3, visibility=0.1),
        "right_elbow": WorldLandmark(x=-0.15, y=0.25, z=0.35, visibility=0.1),
        "right_hip": WorldLandmark(x=-0.1, y=0.5, z=0.3, visibility=0.1),
        "right_knee": WorldLandmark(x=-0.1, y=0.7, z=0.3, visibility=0.1),
        "right_ankle": WorldLandmark(x=-0.1, y=0.9, z=0.3, visibility=0.1),
        "right_foot_index": WorldLandmark(x=-0.12, y=0.92, z=0.3, visibility=0.1),
    }
    pose = PoseData(
        pose_index=0,
        landmarks=landmark_dict,
        world_landmarks=world_landmark_dict
    )
    frames = [
        FrameData(
            frame_index=0,
            timestamp_ms=500,
            detected=True,
            num_poses=1,
            image_width=640,
            image_height=480,
            poses=[pose]
        ),
        FrameData(
            frame_index=1,
            timestamp_ms=1200,
            detected=True,
            num_poses=1,
            image_width=640,
            image_height=480,
            poses=[pose]
        )
    ]
    payload = {
        "session_id": "test_failure_session",
        "measurement_type": "shoulder",
        "joint": "left_shoulder",
        "movement": "abduction",
        "frames": [f.model_dump() for f in frames]
    }
    
    app.dependency_overrides.clear()
    response = client.post("/api/v1/analyze/rom", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "신뢰도가 낮아" in response.json()["detail"]
