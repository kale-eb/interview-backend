#!/usr/bin/env python3
"""
Interview Analysis API for ECS Deployment
Supports MediaPipe for advanced video analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Interview Analysis API",
    description="Advanced interview analysis with MediaPipe",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe (same as main.py)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Global MediaPipe instance (same as main.py)
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    refine_face_landmarks=True,  # This enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# AnalysisResult model (copied from main.py)
class AnalysisResult(BaseModel):
    posture: Dict[str, float]
    eye_contact: Dict[str, float]
    hand_gestures: Dict[str, float]
    overall_score: float
    recommendations: List[str]

# PostureAnalyzer, EyeContactAnalyzer, HandGestureAnalyzer (copied from main.py)
class PostureAnalyzer:
    @staticmethod
    def analyze_posture(pose_landmarks, image_width: int, image_height: int) -> Dict[str, float]:
        if not pose_landmarks:
            return {"alignment": 0.0, "straightness": 0.0, "confidence": 0.0, "head_tilt": 0.0}
        left_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
        nose = pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        left_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        shoulder_alignment = max(0, 1 - (shoulder_diff * 10))
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        frame_center = 0.5
        centering_offset = abs(center_x - frame_center)
        centering_score = max(0, 1 - (centering_offset * 4))
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        spine_deviation = abs(shoulder_center_x - hip_center_x)
        straightness_score = max(0, 1 - (spine_deviation * 8))
        ear_diff = abs(left_ear.y - right_ear.y)
        head_tilt_score = max(0, 1 - (ear_diff * 15))
        ear_center_x = (left_ear.x + right_ear.x) / 2
        nose_ear_deviation = abs(nose.x - ear_center_x)
        head_forward_score = max(0, 1 - (nose_ear_deviation * 8))
        combined_head_tilt = (head_tilt_score * 0.6 + head_forward_score * 0.4)
        confidence = (shoulder_alignment + centering_score + straightness_score + combined_head_tilt) / 4
        return {
            "alignment": round(shoulder_alignment, 3),
            "centering": round(centering_score, 3),
            "straightness": round(straightness_score, 3),
            "head_tilt": round(combined_head_tilt, 3),
            "confidence": round(confidence, 3)
        }

class EyeContactAnalyzer:
    @staticmethod
    def analyze_eye_contact(face_landmarks) -> Dict[str, float]:
        if not face_landmarks:
            return {"eye_contact_score": 0.0, "gaze_deviation": 1.0, "confidence": 0.0}
        try:
            left_eye_corner_left = face_landmarks.landmark[33]
            left_eye_corner_right = face_landmarks.landmark[133]
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            left_eye_top_inner = face_landmarks.landmark[158]
            left_eye_bottom_inner = face_landmarks.landmark[153]
            left_iris_landmarks = [468, 469, 470, 471, 472] if len(face_landmarks.landmark) > 472 else []
            right_iris_landmarks = [473, 474, 475, 476, 477] if len(face_landmarks.landmark) > 477 else []
            right_eye_corner_left = face_landmarks.landmark[362]
            right_eye_corner_right = face_landmarks.landmark[263]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            right_eye_top_inner = face_landmarks.landmark[385]
            right_eye_bottom_inner = face_landmarks.landmark[380]
            left_eye_width = abs(left_eye_corner_right.x - left_eye_corner_left.x)
            left_eye_height = max(
                abs(left_eye_top.y - left_eye_bottom.y),
                abs(left_eye_top_inner.y - left_eye_bottom_inner.y)
            ) * 1.5
            left_eye_geometric_center_x = (left_eye_corner_left.x + left_eye_corner_right.x) / 2
            left_eye_geometric_center_y = (left_eye_top.y + left_eye_bottom.y + left_eye_top_inner.y + left_eye_bottom_inner.y) / 4
            right_eye_width = abs(right_eye_corner_right.x - right_eye_corner_left.x)
            right_eye_height = max(
                abs(right_eye_top.y - right_eye_bottom.y),
                abs(right_eye_top_inner.y - right_eye_bottom_inner.y)
            ) * 1.5
            right_eye_geometric_center_x = (right_eye_corner_left.x + right_eye_corner_right.x) / 2
            right_eye_geometric_center_y = (right_eye_top.y + right_eye_bottom.y + right_eye_top_inner.y + right_eye_bottom_inner.y) / 4
            if left_iris_landmarks and right_iris_landmarks:
                left_iris_x = np.mean([face_landmarks.landmark[i].x for i in left_iris_landmarks])
                left_iris_y = np.mean([face_landmarks.landmark[i].y for i in left_iris_landmarks])
                right_iris_x = np.mean([face_landmarks.landmark[i].x for i in right_iris_landmarks])
                right_iris_y = np.mean([face_landmarks.landmark[i].y for i in right_iris_landmarks])
                iris_available = True
            else:
                left_iris_x = left_eye_geometric_center_x
                left_iris_y = left_eye_geometric_center_y
                right_iris_x = right_eye_geometric_center_x
                right_iris_y = right_eye_geometric_center_y
                iris_available = False
            left_iris_offset_x = (left_iris_x - left_eye_geometric_center_x) / (left_eye_width / 2) if left_eye_width > 0 else 0
            left_iris_offset_y = (left_iris_y - left_eye_geometric_center_y) / (left_eye_height / 2) if left_eye_height > 0 else 0
            right_iris_offset_x = (right_iris_x - right_eye_geometric_center_x) / (right_eye_width / 2) if right_eye_width > 0 else 0
            right_iris_offset_y = (right_iris_y - right_eye_geometric_center_y) / (right_eye_height / 2) if right_eye_height > 0 else 0
            avg_gaze_offset_x = (abs(left_iris_offset_x) + abs(right_iris_offset_x)) / 2
            avg_gaze_offset_y = (abs(left_iris_offset_y) + abs(right_iris_offset_y)) / 2
            nose_tip = face_landmarks.landmark[1]
            nose_bridge = face_landmarks.landmark[6]
            chin = face_landmarks.landmark[18]
            face_center_x = (left_eye_geometric_center_x + right_eye_geometric_center_x) / 2
            face_center_y = (left_eye_geometric_center_y + right_eye_geometric_center_y) / 2
            head_centering_deviation = abs(face_center_x - 0.5)
            eye_level_difference = abs(left_eye_geometric_center_y - right_eye_geometric_center_y)
            gaze_score = max(0, 1 - (avg_gaze_offset_x * 2.5 + avg_gaze_offset_y * 1.2))
            head_position_score = max(0, 1 - (head_centering_deviation * 3.5))
            head_tilt_score = max(0, 1 - (eye_level_difference * 15))
            eye_contact_score = (gaze_score * 0.7 + head_position_score * 0.2 + head_tilt_score * 0.1)
            eye_detection_quality = min(left_eye_width, right_eye_width) * min(left_eye_height, right_eye_height)
            confidence = min(1.0, eye_detection_quality * 1000)
            if iris_available:
                confidence = min(1.0, confidence * 1.2)
            overall_gaze_deviation = (avg_gaze_offset_x + avg_gaze_offset_y) / 2
            return {
                "eye_contact_score": round(max(0, min(1, eye_contact_score)), 3),
                "gaze_deviation": round(overall_gaze_deviation, 3),
                "confidence": round(max(0, min(1, confidence)), 3),
                "head_centering": round(head_position_score, 3),
                "gaze_horizontal": round(avg_gaze_offset_x, 3),
                "gaze_vertical": round(avg_gaze_offset_y, 3)
            }
        except Exception as e:
            logger.warning(f"Advanced eye tracking failed, using basic method: {e}")
            left_eye_x = (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2
            right_eye_x = (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2
            eye_center_x = (left_eye_x + right_eye_x) / 2
            center_deviation = abs(eye_center_x - 0.5)
            basic_score = max(0, 1 - center_deviation * 3.5)
            return {
                "eye_contact_score": round(basic_score, 3),
                "gaze_deviation": round(center_deviation, 3),
                "confidence": 0.5,
                "head_centering": round(basic_score, 3),
                "gaze_horizontal": round(center_deviation, 3),
                "gaze_vertical": 0.0
            }

class HandGestureAnalyzer:
    @staticmethod
    def analyze_hand_gestures(left_hand_landmarks, right_hand_landmarks, face_landmarks) -> Dict[str, float]:
        result = {
            "face_touching": 0.0,
            "nervous_gestures": 0.0,
            "appropriate_gestures": 0.0,
            "confidence": 0.0
        }
        if not face_landmarks:
            return result
        face_points = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        face_x_min = min(p[0] for p in face_points)
        face_x_max = max(p[0] for p in face_points)
        face_y_min = min(p[1] for p in face_points)
        face_y_max = max(p[1] for p in face_points)
        face_width = face_x_max - face_x_min
        face_height = face_y_max - face_y_min
        expansion_x = face_width * 0.3
        expansion_y = face_height * 0.3
        expanded_x_min = max(0, face_x_min - expansion_x)
        expanded_x_max = min(1, face_x_max + expansion_x)
        expanded_y_min = max(0, face_y_min - expansion_y)
        expanded_y_max = min(1, face_y_max + expansion_y)
        hands_detected = 0
        face_touching_score = 0.0
        nervous_score = 0.0
        appropriate_score = 0.0
        confidence = 0.0
        for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
            if hand_landmarks:
                hands_detected += 1
                hand_points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                for point in hand_points:
                    if (face_x_min <= point[0] <= face_x_max and 
                        face_y_min <= point[1] <= face_y_max):
                        face_touching_score += 0.5
                    elif (expanded_x_min <= point[0] <= expanded_x_max and 
                          expanded_y_min <= point[1] <= expanded_y_max):
                        face_touching_score += 0.2
                wrist = hand_landmarks.landmark[0]
                middle_finger_tip = hand_landmarks.landmark[12]
                if wrist.y < 0.6 and abs(wrist.x - 0.5) < 0.3:
                    nervous_score += 0.3
                if 0.3 <= wrist.y <= 0.8 and 0.2 <= wrist.x <= 0.8:
                    appropriate_score += 0.5
        if hands_detected > 0:
            confidence = min(1.0, hands_detected / 2.0)
            face_touching_score = min(1.0, face_touching_score)
            nervous_score = min(1.0, nervous_score)
            appropriate_score = min(1.0, appropriate_score)
        result.update({
            "face_touching": round(face_touching_score, 3),
            "nervous_gestures": round(nervous_score, 3),
            "appropriate_gestures": round(appropriate_score, 3),
            "confidence": round(confidence, 3)
        })
        return result

# Unified process_frame (copied from main.py)
def process_frame(image: np.ndarray) -> AnalysisResult:
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)
        height, width = image.shape[:2]
        posture_analysis = PostureAnalyzer.analyze_posture(results.pose_landmarks, width, height)
        eye_contact_analysis = EyeContactAnalyzer.analyze_eye_contact(results.face_landmarks)
        hand_gesture_analysis = HandGestureAnalyzer.analyze_hand_gestures(
            results.left_hand_landmarks, 
            results.right_hand_landmarks, 
            results.face_landmarks
        )
        overall_score = (
            posture_analysis["confidence"] * 0.4 +
            eye_contact_analysis["confidence"] * 0.35 +
            hand_gesture_analysis["confidence"] * 0.25
        )
        recommendations = []
        if posture_analysis["confidence"] < 0.7:
            recommendations.append("Try to sit up straighter and center yourself in the frame")
        if eye_contact_analysis["eye_contact_score"] < 0.6:
            recommendations.append("Maintain more eye contact with the camera")
        if hand_gesture_analysis["face_touching"] > 0.3:
            recommendations.append("Avoid touching your face during the interview")
        if hand_gesture_analysis["nervous_gestures"] > 0.5:
            recommendations.append("Try to keep your hands in a more natural, relaxed position")
        if not recommendations:
            recommendations.append("Great job! Keep up the good body language")
        return AnalysisResult(
            posture=posture_analysis,
            eye_contact=eye_contact_analysis,
            hand_gestures=hand_gesture_analysis,
            overall_score=round(overall_score, 3),
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Frame processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze/frame")
async def analyze_frame(
    frame_data: str = Form(...),
    session_id: str = Form(""),
    user_id: str = Form("")
):
    try:
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        image_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        result = process_frame(frame)
        return result.dict()
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/session")
async def analyze_session(
    session_data: str = Form(...),
    session_id: str = Form(...)
):
    """Analyze a complete session"""
    
    try:
        # Parse session data
        session_info = json.loads(session_data)
        
        # For now, return session summary
        # In production, you'd analyze all frames
        result = {
            "session_id": session_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "frames_analyzed": len(session_info.get("frames", [])),
            "summary": "Session analysis completed"
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error analyzing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Interview Analysis API",
        "version": "2.0.0",
        "features": ["MediaPipe", "Face Analysis", "Pose Analysis", "Hand Analysis"],
        "endpoints": ["/analyze/frame", "/analyze/session", "/health"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 