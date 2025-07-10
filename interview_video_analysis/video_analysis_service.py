#!/usr/bin/env python3
"""
Complete Video Analysis Service
Analyzes entire video files and returns session logs with ratings
"""

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import tempfile
import os
import time
import json
from datetime import datetime
from typing import Dict, List
import logging
import io
import requests

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

class VideoAnalysisRequest(BaseModel):
    video_data: str  # base64 encoded video

class VideoAnalysisResponse(BaseModel):
    session_log: Dict  # Full session log for logging
    final_rating: Dict  # Clean rating format like terminal output
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
            
            # Right eye corners and key points  
            right_eye_corner_left = face_landmarks.landmark[362]  # Left corner of right eye
            right_eye_corner_right = face_landmarks.landmark[263] # Right corner of right eye
            right_eye_top = face_landmarks.landmark[386]          # Top of right eye
            right_eye_bottom = face_landmarks.landmark[374]       # Bottom of right eye
            
            # Calculate eye dimensions and centers
            left_eye_width = abs(left_eye_corner_right.x - left_eye_corner_left.x)
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y) * 1.5
            left_eye_center_x = (left_eye_corner_left.x + left_eye_corner_right.x) / 2
            left_eye_center_y = (left_eye_top.y + left_eye_bottom.y) / 2
            
            right_eye_width = abs(right_eye_corner_right.x - right_eye_corner_left.x)
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y) * 1.5
            right_eye_center_x = (right_eye_corner_left.x + right_eye_corner_right.x) / 2
            right_eye_center_y = (right_eye_top.y + right_eye_bottom.y) / 2
            
            # Calculate face center and head positioning
            face_center_x = (left_eye_center_x + right_eye_center_x) / 2
            face_center_y = (left_eye_center_y + right_eye_center_y) / 2
            
            # Head centering in frame (should be around 0.5 for good positioning)
            head_centering_deviation = abs(face_center_x - 0.5)
            
            # Head tilt detection (eyes should be level)
            eye_level_difference = abs(left_eye_center_y - right_eye_center_y)
            
            # Calculate eye contact score
            head_position_score = max(0, 1 - (head_centering_deviation * 3.5))
            head_tilt_score = max(0, 1 - (eye_level_difference * 15))
            
            # Combined eye contact score
            eye_contact_score = (head_position_score * 0.7 + head_tilt_score * 0.3)
            
            # Calculate confidence based on eye detection quality
            eye_detection_quality = min(left_eye_width, right_eye_width) * min(left_eye_height, right_eye_height)
            confidence = min(1.0, eye_detection_quality * 1000)
            
            return {
                "eye_contact_score": round(max(0, min(1, eye_contact_score)), 3),
                "gaze_deviation": round(head_centering_deviation, 3),
                "confidence": round(max(0, min(1, confidence)), 3),
                "head_centering": round(head_position_score, 3)
            }
            
        except Exception as e:
            logger.warning(f"Eye tracking failed, using basic method: {e}")
            
            # Simple fallback method
            left_eye_x = (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2
            right_eye_x = (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2
            eye_center_x = (left_eye_x + right_eye_x) / 2
            
            # Basic centering score
            center_deviation = abs(eye_center_x - 0.5)
            basic_score = max(0, 1 - center_deviation * 3.5)
            
            return {
                "eye_contact_score": round(basic_score, 3),
                "gaze_deviation": round(center_deviation, 3),
                "confidence": 0.5,
                "head_centering": round(basic_score, 3)
            }

class HandGestureAnalyzer:
    @staticmethod
    def analyze_hand_gestures(left_hand_landmarks, right_hand_landmarks, face_landmarks) -> Dict[str, float]:
        """Analyze hand gestures and detect nervous behaviors"""
        result = {
            "face_touching": 0.0,
            "nervous_gestures": 0.0,
            "appropriate_gestures": 0.0,
            "confidence": 0.0
        }
        
        if not face_landmarks:
            return result
        
        # Get face boundaries
        face_points = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        face_x_min = min(p[0] for p in face_points)
        face_x_max = max(p[0] for p in face_points)
        face_y_min = min(p[1] for p in face_points)
        face_y_max = max(p[1] for p in face_points)
        
        # Expand face region by 30% to catch approaching hands
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
        confidence = 0.0  # Initialize confidence
        
        for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
            if hand_landmarks:
                hands_detected += 1
                
                # Get hand landmarks
                hand_points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                
                # Check for face touching
                for point in hand_points:
                    # Direct face contact
                    if (face_x_min <= point[0] <= face_x_max and 
                        face_y_min <= point[1] <= face_y_max):
                        face_touching_score += 0.5
                    # Approaching face
                    elif (expanded_x_min <= point[0] <= expanded_x_max and 
                          expanded_y_min <= point[1] <= expanded_y_max):
                        face_touching_score += 0.2
                
                # Analyze hand position for nervous gestures
                wrist = hand_landmarks.landmark[0]
                
                # Check if hands are in "nervous" positions
                if wrist.y < 0.6 and abs(wrist.x - 0.5) < 0.3:
                    nervous_score += 0.3
                
                # Check for appropriate gesture range
                if 0.3 <= wrist.y <= 0.8 and 0.2 <= wrist.x <= 0.8:
                    appropriate_score += 0.5
        
        # Normalize scores
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

@app.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze a complete video and return session log with rating"""
    tmpfile_path = None
    try:
        start_time = time.time()
        # Generate session_id and user_id automatically
        session_id = f"session_{int(time.time())}"
        user_id = "auto_user"
        logger.info(f"Starting video analysis for session {session_id}")
        
        # Decode base64 video
        video_data = base64.b64decode(request.video_data)
        
        # Auto-detect format by trying common extensions
        video_formats = ['mp4', 'mov', 'avi', 'mkv', 'webm']
        tmpfile_path = None
        
        for fmt in video_formats:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp:
                    tmp.write(video_data)
                    tmpfile_path = tmp.name
                
                # Test if the file can be opened
                test_cap = cv2.VideoCapture(tmpfile_path)
                if test_cap.isOpened():
                    test_cap.release()
                    logger.info(f"Auto-detected video format: {fmt}")
                    break
                else:
                    # Clean up failed attempt
                    os.unlink(tmpfile_path)
                    tmpfile_path = None
            except Exception:
                if tmpfile_path and os.path.exists(tmpfile_path):
                    os.unlink(tmpfile_path)
                tmpfile_path = None
                continue
        
        if tmpfile_path is None:
            raise HTTPException(status_code=400, detail="Could not detect video format or video is invalid")
        
        # Initialize MediaPipe
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
        
        # Initialize video capture
        cap = cv2.VideoCapture(tmpfile_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / frame_rate if frame_rate > 0 else 0
        
        logger.info(f"Video info: {total_frames} frames, {frame_rate:.2f} fps, {duration_seconds:.2f}s")
        
        # Initialize session tracking with duration-based event logging
        session_start_time = time.time()
        session_events = []
        session_stats = {
            "eye_contact_breaks": [],
            "face_touch_incidents": [],
            "bad_posture_periods": [],
            "flicker_events": []
        }
        
        # Duration tracking for sustained events
        eye_contact_break_start = None
        face_touch_start = None
        bad_posture_start = None
        
        frame_count = 0
        # Remove frame_data storage - we only need events, not every frame
        
        # For calculating averages at the end
        posture_scores = []
        eye_contact_scores = []
        
        # Process at 10 FPS max for faster analysis
        analysis_fps = 10
        frame_skip = max(1, int(frame_rate / analysis_fps)) if frame_rate > 0 else 1
        logger.info(f"Processing every {frame_skip} frames (target: {analysis_fps} FPS)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / frame_rate if frame_rate > 0 else 0
            
            # Skip frames to achieve target FPS
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = holistic_video.process(rgb_frame)
            
            # Get image dimensions
            height, width = frame.shape[:2]
            
            # Analyze components
            posture_analysis = PostureAnalyzer.analyze_posture(results.pose_landmarks, width, height)
            eye_contact_analysis = EyeContactAnalyzer.analyze_eye_contact(results.face_landmarks)
            hand_gesture_analysis = HandGestureAnalyzer.analyze_hand_gestures(
                results.left_hand_landmarks, 
                results.right_hand_landmarks, 
                results.face_landmarks
            )
            
            # Store scores for calculating averages (not full frame data)
            posture_scores.append(posture_analysis["confidence"])
            eye_contact_scores.append(eye_contact_analysis["eye_contact_score"])
            
            # Duration-based event logging (like original live_ui.py)
            # Only log sustained behaviors that meet minimum duration thresholds
            
            # Eye contact break tracking (1.5+ seconds like original)
            if eye_contact_analysis["eye_contact_score"] < 0.6:
                if eye_contact_break_start is None:
                    eye_contact_break_start = current_time
            else:
                if eye_contact_break_start is not None:
                    duration = current_time - eye_contact_break_start
                    if duration >= 1.5:  # Only log breaks 1.5+ seconds
                        start_rel_time = eye_contact_break_start
                        end_rel_time = current_time
                        
                        # Add to session_stats (for rating service compatibility)
                        stats_data = {
                            "start_time_relative": start_rel_time,
                            "end_time_relative": end_rel_time,
                            "duration": duration,
                            "severity": "high" if duration > 3 else "medium",
                            "score_start": eye_contact_analysis["eye_contact_score"]
                        }
                        session_stats["eye_contact_breaks"].append(stats_data)
                        
                        # Add to events (for logging)
                        session_events.append({
                            "timestamp_relative": end_rel_time,
                            "event_type": "eye_contact_break",
                            "details": stats_data
                        })
                        
                        logger.info(f"🔴 Eye contact break logged: {duration:.1f}s at {start_rel_time:.1f}s")
                    eye_contact_break_start = None
            
            # Face touching tracking (0.5+ seconds like original)
            if hand_gesture_analysis["face_touching"] > 0.25:  # Match original threshold
                if face_touch_start is None:
                    face_touch_start = current_time
            else:
                if face_touch_start is not None:
                    duration = current_time - face_touch_start
                    if duration >= 0.5:  # Only log sustained face touching
                        start_rel_time = face_touch_start
                        end_rel_time = current_time
                        
                        # Add to session_stats (for rating service compatibility)
                        stats_data = {
                            "start_time_relative": start_rel_time,
                            "end_time_relative": end_rel_time,
                            "duration": duration,
                            "severity": "high" if duration > 2 else "medium",
                            "max_score": hand_gesture_analysis["face_touching"],
                            "total_count": len(session_stats["face_touch_incidents"]) + 1
                        }
                        session_stats["face_touch_incidents"].append(stats_data)
                        
                        # Add to events (for logging)
                        session_events.append({
                            "timestamp_relative": end_rel_time,
                            "event_type": "face_touch",
                            "details": stats_data
                        })
                        
                        logger.info(f"🔴 Face touch logged: {duration:.1f}s at {start_rel_time:.1f}s")
                    face_touch_start = None
            
            # Bad posture tracking (3+ seconds like original)
            if posture_analysis["confidence"] < 0.7:
                if bad_posture_start is None:
                    bad_posture_start = current_time
            else:
                if bad_posture_start is not None:
                    duration = current_time - bad_posture_start
                    if duration >= 3.0:  # Only log sustained bad posture
                        start_rel_time = bad_posture_start
                        end_rel_time = current_time
                        
                        # Add to session_stats (for rating service compatibility)
                        stats_data = {
                            "start_time_relative": start_rel_time,
                            "end_time_relative": end_rel_time,
                            "duration": duration,
                            "severity": "medium"
                        }
                        session_stats["bad_posture_periods"].append(stats_data)
                        
                        # Add to events (for logging)
                        session_events.append({
                            "timestamp_relative": end_rel_time,
                            "event_type": "bad_posture_start",
                            "details": stats_data
                        })
                        
                        logger.info(f"🔴 Bad posture logged: {duration:.1f}s at {start_rel_time:.1f}s")
                    bad_posture_start = None
            
            frame_count += 1
            
            # Log progress (adjusted for frame skipping)
            if frame_count % (100 * frame_skip) == 0:
                analyzed_frames = frame_count // frame_skip
                logger.info(f"Analyzed {analyzed_frames} frames (frame {frame_count}/{total_frames})")
        
        # Handle ongoing events at end of video (same format as during video)
        if eye_contact_break_start is not None:
            duration = duration_seconds - eye_contact_break_start
            if duration >= 1.5:
                start_rel_time = eye_contact_break_start
                end_rel_time = duration_seconds
                
                # Add to session_stats (for rating service compatibility)
                stats_data = {
                    "start_time_relative": start_rel_time,
                    "end_time_relative": end_rel_time,
                    "duration": duration,
                    "severity": "high" if duration > 3 else "medium",
                    "score_start": 0.0  # Unknown score at end
                }
                session_stats["eye_contact_breaks"].append(stats_data)
                
                # Add to events (for logging)
                session_events.append({
                    "timestamp_relative": end_rel_time,
                    "event_type": "eye_contact_break",
                    "details": stats_data
                })
                
                logger.info(f"🔴 Eye contact break logged (end): {duration:.1f}s at {start_rel_time:.1f}s")
        
        if face_touch_start is not None:
            duration = duration_seconds - face_touch_start
            if duration >= 0.5:
                start_rel_time = face_touch_start
                end_rel_time = duration_seconds
                
                # Add to session_stats (for rating service compatibility)
                stats_data = {
                    "start_time_relative": start_rel_time,
                    "end_time_relative": end_rel_time,
                    "duration": duration,
                    "severity": "high" if duration > 2 else "medium",
                    "max_score": 0.0,  # Unknown score at end
                    "total_count": len(session_stats["face_touch_incidents"]) + 1
                }
                session_stats["face_touch_incidents"].append(stats_data)
                
                # Add to events (for logging)
                session_events.append({
                    "timestamp_relative": end_rel_time,
                    "event_type": "face_touch",
                    "details": stats_data
                })
                
                logger.info(f"🔴 Face touch logged (end): {duration:.1f}s at {start_rel_time:.1f}s")
        
        if bad_posture_start is not None:
            duration = duration_seconds - bad_posture_start
            if duration >= 3.0:
                start_rel_time = bad_posture_start
                end_rel_time = duration_seconds
                
                # Add to session_stats (for rating service compatibility)
                stats_data = {
                    "start_time_relative": start_rel_time,
                    "end_time_relative": end_rel_time,
                    "duration": duration,
                    "severity": "medium"
                }
                session_stats["bad_posture_periods"].append(stats_data)
                
                # Add to events (for logging)
                session_events.append({
                    "timestamp_relative": end_rel_time,
                    "event_type": "bad_posture_start",
                    "details": stats_data
                })
                
                logger.info(f"🔴 Bad posture logged (end): {duration:.1f}s at {start_rel_time:.1f}s")
        
        # Close video capture
        cap.release()
        
        # Create session log for rating service (compatible with existing format)
        session_stats.update({
            "session_id": session_id,
            "start_time": datetime.fromtimestamp(session_start_time).isoformat(),
            "session_start_timestamp": session_start_time,
            "end_time": datetime.now().isoformat(),
            "total_duration": duration_seconds,
            "total_face_touches": len(session_stats["face_touch_incidents"]),
            "avg_eye_contact_score": np.mean(eye_contact_scores) if eye_contact_scores else 0,
            "avg_posture_score": np.mean(posture_scores) if posture_scores else 0
        })
        
        session_data = {
            "session_stats": session_stats,
            "events": session_events
        }
        
        # Send to rating service using localhost (since both services are running in Docker)
        rating_result = None
        try:
            rating_endpoint = "http://localhost:8001/rate/session"
            response = requests.post(
                rating_endpoint,
                json=session_data,
                timeout=15
            )
            
            if response.status_code == 200:
                rating_response = response.json()
                if rating_response.get("success"):
                    rating_result = rating_response["rating"]
                    logger.info(f"📊 Session rated successfully: {rating_result.get('final_score', 'N/A')}/100")
                else:
                    logger.error(f"Rating service error: {rating_response.get('error', 'Unknown error')}")
            else:
                logger.error(f"Rating service returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️  Rating service not available")
        except Exception as e:
            logger.error(f"Error rating session: {str(e)}")
        
        # Create session log (same format as original - no frame_data)
        session_log = {
            "session_stats": session_stats,
            "events": session_events
        }
        
        # Create clean final rating (like terminal output)
        final_rating = rating_result if rating_result else {
            "final_score": 0.0,
            "grade": "N/A",
            "eye_contact_score": 0.0,
            "eye_contact_grade": "N/A",
            "face_touch_score": 0.0,
            "face_touch_grade": "N/A",
            "posture_score": 0.0,
            "posture_grade": "N/A",
            "frequency_penalty": 0.0,
            "events_per_minute": len(session_events) / (duration_seconds / 60) if duration_seconds > 0 else 0
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Video analysis completed in {processing_time:.2f}s")
        
        return VideoAnalysisResponse(
            session_log=session_log,
            final_rating=final_rating,
            processing_time=processing_time,
            total_frames=total_frames
        )
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")
    finally:
        # Clean up temporary file
        if tmpfile_path and os.path.exists(tmpfile_path):
            os.unlink(tmpfile_path)

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