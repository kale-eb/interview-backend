#!/usr/bin/env python3
"""
Complete Video Analysis Service
Analyzes entire video files and returns session logs with ratings
"""

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import math
from typing import Dict, List, Optional, Tuple
import logging
import os
import sys
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Complete Video Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

class VideoAnalysisRequest(BaseModel):
    session_id: str
    user_id: str
    video_data: str  # base64 encoded video
    video_format: str = "mp4"

class VideoAnalysisResponse(BaseModel):
    session_id: str
    user_id: str
    session_log: Dict
    final_rating: Dict
    processing_time: float
    total_frames: int

class PostureAnalyzer:
    @staticmethod
    def analyze_posture(pose_landmarks, image_width: int, image_height: int) -> Dict[str, float]:
        """Analyze posture based on pose landmarks"""
        if not pose_landmarks:
            return {"alignment": 0.0, "straightness": 0.0, "confidence": 0.0, "head_tilt": 0.0}
        
        # Get key landmarks
        left_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
        nose = pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        
        # Additional landmarks for head tilt analysis
        left_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]
        
        # Calculate shoulder alignment (how level the shoulders are)
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        shoulder_alignment = max(0, 1 - (shoulder_diff * 10))  # Penalize shoulder tilt
        
        # Calculate centering (how centered the person is in frame)
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        frame_center = 0.5
        centering_offset = abs(center_x - frame_center)
        centering_score = max(0, 1 - (centering_offset * 4))
        
        # Calculate straightness (spine alignment)
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        spine_deviation = abs(shoulder_center_x - hip_center_x)
        straightness_score = max(0, 1 - (spine_deviation * 8))
        
        # Calculate head tilt (ears should be level for straight forward facing)
        ear_diff = abs(left_ear.y - right_ear.y)
        head_tilt_score = max(0, 1 - (ear_diff * 15))  # More sensitive to head tilt
        
        # Additional head orientation check using nose and ear positions
        # For facing forward, nose should be centered between ears horizontally
        ear_center_x = (left_ear.x + right_ear.x) / 2
        nose_ear_deviation = abs(nose.x - ear_center_x)
        head_forward_score = max(0, 1 - (nose_ear_deviation * 8))
        
        # Combined head tilt score (both level and forward-facing)
        combined_head_tilt = (head_tilt_score * 0.6 + head_forward_score * 0.4)
        
        # Overall posture confidence (including head tilt)
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
        """Analyze eye contact and gaze direction using iris tracking and head pose"""
        if not face_landmarks:
            return {"eye_contact_score": 0.0, "gaze_deviation": 1.0, "confidence": 0.0}
        
        try:
            # Key eye landmarks for better analysis
            # Left eye corners and key points
            left_eye_corner_left = face_landmarks.landmark[33]   # Left corner of left eye
            left_eye_corner_right = face_landmarks.landmark[133] # Right corner of left eye
            left_eye_top = face_landmarks.landmark[159]          # Top of left eye
            left_eye_bottom = face_landmarks.landmark[145]       # Bottom of left eye
            # Additional landmarks for better vertical detection
            left_eye_top_inner = face_landmarks.landmark[158]    # Inner top
            left_eye_bottom_inner = face_landmarks.landmark[153] # Inner bottom
            # Try to get iris landmarks (available with refine_face_landmarks=True)
            # Left iris: landmarks 468-472, Right iris: landmarks 473-477
            left_iris_landmarks = [468, 469, 470, 471, 472] if len(face_landmarks.landmark) > 472 else []
            right_iris_landmarks = [473, 474, 475, 476, 477] if len(face_landmarks.landmark) > 477 else []
            
            # Right eye corners and key points  
            right_eye_corner_left = face_landmarks.landmark[362]  # Left corner of right eye
            right_eye_corner_right = face_landmarks.landmark[263] # Right corner of right eye
            right_eye_top = face_landmarks.landmark[386]          # Top of right eye
            right_eye_bottom = face_landmarks.landmark[374]       # Bottom of right eye
            # Additional landmarks for better vertical detection
            right_eye_top_inner = face_landmarks.landmark[385]    # Inner top
            right_eye_bottom_inner = face_landmarks.landmark[380] # Inner bottom
            
            # Calculate eye dimensions and centers (using multiple landmarks for better accuracy)
            left_eye_width = abs(left_eye_corner_right.x - left_eye_corner_left.x)
            # Use both outer and inner vertical landmarks for better height estimation
            left_eye_height = max(
                abs(left_eye_top.y - left_eye_bottom.y),
                abs(left_eye_top_inner.y - left_eye_bottom_inner.y)
            ) * 1.5  # Scale up to account for visible eye opening vs. actual eye socket
            left_eye_geometric_center_x = (left_eye_corner_left.x + left_eye_corner_right.x) / 2
            left_eye_geometric_center_y = (left_eye_top.y + left_eye_bottom.y + left_eye_top_inner.y + left_eye_bottom_inner.y) / 4
            
            right_eye_width = abs(right_eye_corner_right.x - right_eye_corner_left.x)
            # Use both outer and inner vertical landmarks for better height estimation
            right_eye_height = max(
                abs(right_eye_top.y - right_eye_bottom.y),
                abs(right_eye_top_inner.y - right_eye_bottom_inner.y)
            ) * 1.5  # Scale up to account for visible eye opening vs. actual eye socket
            right_eye_geometric_center_x = (right_eye_corner_left.x + right_eye_corner_right.x) / 2
            right_eye_geometric_center_y = (right_eye_top.y + right_eye_bottom.y + right_eye_top_inner.y + right_eye_bottom_inner.y) / 4
            
            # Try to use iris landmarks if available, otherwise use geometric center
            if left_iris_landmarks and right_iris_landmarks:
                # Calculate iris centers from available iris landmarks
                left_iris_x = np.mean([face_landmarks.landmark[i].x for i in left_iris_landmarks])
                left_iris_y = np.mean([face_landmarks.landmark[i].y for i in left_iris_landmarks])
                right_iris_x = np.mean([face_landmarks.landmark[i].x for i in right_iris_landmarks])
                right_iris_y = np.mean([face_landmarks.landmark[i].y for i in right_iris_landmarks])
                iris_available = True
            else:
                # Fallback to geometric centers if iris landmarks not available
                left_iris_x = left_eye_geometric_center_x
                left_iris_y = left_eye_geometric_center_y
                right_iris_x = right_eye_geometric_center_x
                right_iris_y = right_eye_geometric_center_y
                iris_available = False
            
            # Calculate iris position relative to eye boundaries (gaze direction)
            # For good eye contact, iris should be centered in the eye
            left_iris_offset_x = (left_iris_x - left_eye_geometric_center_x) / (left_eye_width / 2) if left_eye_width > 0 else 0
            left_iris_offset_y = (left_iris_y - left_eye_geometric_center_y) / (left_eye_height / 2) if left_eye_height > 0 else 0
            
            right_iris_offset_x = (right_iris_x - right_eye_geometric_center_x) / (right_eye_width / 2) if right_eye_width > 0 else 0
            right_iris_offset_y = (right_iris_y - right_eye_geometric_center_y) / (right_eye_height / 2) if right_eye_height > 0 else 0
            
            # Average gaze deviation from center
            avg_gaze_offset_x = (abs(left_iris_offset_x) + abs(right_iris_offset_x)) / 2
            avg_gaze_offset_y = (abs(left_iris_offset_y) + abs(right_iris_offset_y)) / 2
            
            # Head pose estimation using facial landmarks
            nose_tip = face_landmarks.landmark[1]
            nose_bridge = face_landmarks.landmark[6]
            chin = face_landmarks.landmark[18]
            
            # Calculate head orientation
            face_center_x = (left_eye_geometric_center_x + right_eye_geometric_center_x) / 2
            face_center_y = (left_eye_geometric_center_y + right_eye_geometric_center_y) / 2
            
            # Head centering in frame (should be around 0.5 for good positioning)
            head_centering = 1 - abs(face_center_x - 0.5) * 2  # Penalize being off-center
            
            # Calculate eye contact score based on gaze direction and head orientation
            # Lower gaze deviation = better eye contact
            gaze_score = max(0, 1 - (avg_gaze_offset_x + avg_gaze_offset_y) / 2)
            
            # Combined eye contact score (gaze + head orientation)
            eye_contact_score = (gaze_score * 0.7 + head_centering * 0.3)
            
            # Confidence based on iris availability and overall face detection quality
            confidence = eye_contact_score * (1.0 if iris_available else 0.8)
            
            return {
                "eye_contact_score": round(eye_contact_score, 3),
                "gaze_deviation": round((avg_gaze_offset_x + avg_gaze_offset_y) / 2, 3),
                "head_centering": round(head_centering, 3),
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Eye contact analysis error: {str(e)}")
            return {"eye_contact_score": 0.0, "gaze_deviation": 1.0, "confidence": 0.0}

class HandGestureAnalyzer:
    @staticmethod
    def analyze_hand_gestures(left_hand_landmarks, right_hand_landmarks, face_landmarks) -> Dict[str, float]:
        """Analyze hand gestures and detect face touching"""
        if not face_landmarks:
            return {"face_touching": 0.0, "nervous_gestures": 0.0, "confidence": 0.0}
        
        try:
            # Get face landmarks for face touching detection
            nose_tip = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[18]
            left_ear = face_landmarks.landmark[234]
            right_ear = face_landmarks.landmark[454]
            
            # Define face region boundaries
            face_left = min(left_ear.x, nose_tip.x) - 0.05
            face_right = max(right_ear.x, nose_tip.x) + 0.05
            face_top = min(nose_tip.y, chin.y) - 0.1
            face_bottom = max(nose_tip.y, chin.y) + 0.1
            
            face_touching_score = 0.0
            nervous_gestures_score = 0.0
            total_hands = 0
            
            # Analyze left hand
            if left_hand_landmarks:
                total_hands += 1
                hand_points = []
                for landmark in left_hand_landmarks.landmark:
                    hand_points.append((landmark.x, landmark.y))
                
                # Check if any hand points are near the face
                face_touch_count = 0
                for x, y in hand_points:
                    if (face_left <= x <= face_right and face_top <= y <= face_bottom):
                        face_touch_count += 1
                
                if face_touch_count > 0:
                    face_touching_score += face_touch_count / len(hand_points)
                
                # Detect nervous gestures (rapid movements, fidgeting)
                # This is a simplified version - in practice you'd track movement over time
                hand_center_x = np.mean([p[0] for p in hand_points])
                hand_center_y = np.mean([p[1] for p in hand_points])
                
                # Simple heuristic: hands near face or rapid movement patterns
                if face_touching_score > 0.1:
                    nervous_gestures_score += 0.5
            
            # Analyze right hand
            if right_hand_landmarks:
                total_hands += 1
                hand_points = []
                for landmark in right_hand_landmarks.landmark:
                    hand_points.append((landmark.x, landmark.y))
                
                # Check if any hand points are near the face
                face_touch_count = 0
                for x, y in hand_points:
                    if (face_left <= x <= face_right and face_top <= y <= face_bottom):
                        face_touch_count += 1
                
                if face_touch_count > 0:
                    face_touching_score += face_touch_count / len(hand_points)
                
                # Detect nervous gestures
                hand_center_x = np.mean([p[0] for p in hand_points])
                hand_center_y = np.mean([p[1] for p in hand_points])
                
                if face_touching_score > 0.1:
                    nervous_gestures_score += 0.5
            
            # Normalize scores
            if total_hands > 0:
                face_touching_score /= total_hands
                nervous_gestures_score /= total_hands
            
            # Confidence based on hand detection
            confidence = 1.0 if total_hands > 0 else 0.5
            
            return {
                "face_touching": round(face_touching_score, 3),
                "nervous_gestures": round(nervous_gestures_score, 3),
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Hand gesture analysis error: {str(e)}")
            return {"face_touching": 0.0, "nervous_gestures": 0.0, "confidence": 0.0}

def process_frame(image: np.ndarray) -> AnalysisResult:
    """Process a single frame and return analysis results"""
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe Holistic
    results = holistic.process(rgb_image)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Analyze posture
    posture_analysis = PostureAnalyzer.analyze_posture(results.pose_landmarks, width, height)
    
    # Analyze eye contact
    eye_contact_analysis = EyeContactAnalyzer.analyze_eye_contact(results.face_landmarks)
    
    # Analyze hand gestures
    hand_gesture_analysis = HandGestureAnalyzer.analyze_hand_gestures(
        results.left_hand_landmarks, 
        results.right_hand_landmarks, 
        results.face_landmarks
    )
    
    # Calculate overall score
    overall_score = (
        posture_analysis["confidence"] * 0.4 +
        eye_contact_analysis["confidence"] * 0.35 +
        hand_gesture_analysis["confidence"] * 0.25
    )
    
    # Generate recommendations
    recommendations = []
    if posture_analysis["confidence"] < 0.7:
        recommendations.append("Try to sit up straighter and center yourself in the frame")
    if eye_contact_analysis["eye_contact_score"] < 0.6:
        recommendations.append("Maintain more eye contact with the camera")
    if hand_gesture_analysis["face_touching"] > 0.3:
        recommendations.append("Avoid touching your face during the interview")
    if hand_gesture_analysis["nervous_gestures"] > 0.5:
        recommendations.append("Try to keep your hands in a more natural, relaxed position")
    
    return AnalysisResult(
        posture=posture_analysis,
        eye_contact=eye_contact_analysis,
        hand_gestures=hand_gesture_analysis,
        overall_score=round(overall_score, 3),
        recommendations=recommendations
    )

@app.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze a complete video and return session log and final rating"""
    start_time = time.time()
    
    try:
        # Decode base64 video data
        video_data = base64.b64decode(request.video_data)
        video_bytes = io.BytesIO(video_data)
        
        # Initialize MediaPipe Holistic for video processing
        holistic_video = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize session log
        session_log = {
            "session_id": request.session_id,
            "user_id": request.user_id,
            "start_time": datetime.now().isoformat(),
            "total_frames": 0,
            "frame_rate": 0,
            "duration_seconds": 0,
            "posture_data": [],
            "eye_contact_data": [],
            "hand_gesture_data": [],
            "recommendations": [],
            "events": []
        }
        
        # Process video frame by frame
        cap = cv2.VideoCapture()
        cap.open(video_bytes)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / frame_rate if frame_rate > 0 else 0
        
        session_log["total_frames"] = total_frames
        session_log["frame_rate"] = frame_rate
        session_log["duration_seconds"] = duration_seconds
        
        logger.info(f"Processing video: {total_frames} frames, {frame_rate:.2f} fps, {duration_seconds:.2f}s")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = holistic_video.process(rgb_frame)
            
            # Get image dimensions
            height, width = frame.shape[:2]
            
            # Analyze posture
            posture_analysis = PostureAnalyzer.analyze_posture(results.pose_landmarks, width, height)
            session_log["posture_data"].append({
                "frame_number": frame_count,
                "timestamp": frame_count / frame_rate if frame_rate > 0 else 0,
                "posture": posture_analysis
            })
            
            # Analyze eye contact
            eye_contact_analysis = EyeContactAnalyzer.analyze_eye_contact(results.face_landmarks)
            session_log["eye_contact_data"].append({
                "frame_number": frame_count,
                "timestamp": frame_count / frame_rate if frame_rate > 0 else 0,
                "eye_contact": eye_contact_analysis
            })
            
            # Analyze hand gestures
            hand_gesture_analysis = HandGestureAnalyzer.analyze_hand_gestures(
                results.left_hand_landmarks, 
                results.right_hand_landmarks, 
                results.face_landmarks
            )
            session_log["hand_gesture_data"].append({
                "frame_number": frame_count,
                "timestamp": frame_count / frame_rate if frame_rate > 0 else 0,
                "hand_gestures": hand_gesture_analysis
            })
            
            # Log behavioral events
            if posture_analysis["confidence"] < 0.6:
                session_log["events"].append({
                    "frame_number": frame_count,
                    "timestamp": frame_count / frame_rate if frame_rate > 0 else 0,
                    "event_type": "poor_posture",
                    "severity": "medium",
                    "details": {"confidence": posture_analysis["confidence"]}
                })
            
            if eye_contact_analysis["eye_contact_score"] < 0.5:
                session_log["events"].append({
                    "frame_number": frame_count,
                    "timestamp": frame_count / frame_rate if frame_rate > 0 else 0,
                    "event_type": "eye_contact_break",
                    "severity": "medium",
                    "details": {"score": eye_contact_analysis["eye_contact_score"]}
                })
            
            if hand_gesture_analysis["face_touching"] > 0.3:
                session_log["events"].append({
                    "frame_number": frame_count,
                    "timestamp": frame_count / frame_rate if frame_rate > 0 else 0,
                    "event_type": "face_touching",
                    "severity": "high" if hand_gesture_analysis["face_touching"] > 0.7 else "medium",
                    "details": {"score": hand_gesture_analysis["face_touching"]}
                })
            
            frame_count += 1
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        # Calculate final ratings
        posture_scores = [p["posture"]["confidence"] for p in session_log["posture_data"]]
        eye_contact_scores = [e["eye_contact"]["eye_contact_score"] for e in session_log["eye_contact_data"]]
        hand_gesture_scores = [h["hand_gestures"]["confidence"] for h in session_log["hand_gesture_data"]]
        
        # Calculate averages (excluding zero scores)
        avg_posture = np.mean([s for s in posture_scores if s > 0]) if any(s > 0 for s in posture_scores) else 0
        avg_eye_contact = np.mean([s for s in eye_contact_scores if s > 0]) if any(s > 0 for s in eye_contact_scores) else 0
        avg_hand_gesture = np.mean([s for s in hand_gesture_scores if s > 0]) if any(s > 0 for s in hand_gesture_scores) else 0
        
        # Calculate overall rating
        overall_rating = (
            avg_posture * 0.4 +
            avg_eye_contact * 0.35 +
            avg_hand_gesture * 0.25
        )
        
        # Generate final recommendations
        final_recommendations = []
        if avg_posture < 0.7:
            final_recommendations.append("Work on maintaining better posture throughout the interview")
        if avg_eye_contact < 0.6:
            final_recommendations.append("Practice maintaining consistent eye contact with the camera")
        if any(h["hand_gestures"]["face_touching"] > 0.3 for h in session_log["hand_gesture_data"]):
            final_recommendations.append("Avoid touching your face during interviews")
        
        # Create final rating
        final_rating = {
            "session_id": request.session_id,
            "user_id": request.user_id,
            "overall_rating": round(overall_rating, 3),
            "posture_rating": round(avg_posture, 3),
            "eye_contact_rating": round(avg_eye_contact, 3),
            "hand_gesture_rating": round(avg_hand_gesture, 3),
            "total_events": len(session_log["events"]),
            "events_per_minute": round(len(session_log["events"]) / (duration_seconds / 60), 2) if duration_seconds > 0 else 0,
            "final_recommendations": final_recommendations
        }
        
        # Close video capture
        cap.release()
        
        processing_time = time.time() - start_time
        
        logger.info(f"Video analysis completed in {processing_time:.2f}s")
        
        return VideoAnalysisResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            session_log=session_log,
            final_rating=final_rating,
            processing_time=processing_time,
            total_frames=total_frames
        )
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "video-analysis",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Complete Video Analysis API",
        "version": "1.0.0",
        "endpoints": [
            "POST /analyze_video - Analyze complete video and return session log with rating",
            "GET /health - Health check",
            "GET /docs - API documentation"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 