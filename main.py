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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Interview Analysis API", version="1.0.0")

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

# Global MediaPipe instance
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

class AnalysisResult(BaseModel):
    posture: Dict[str, float]
    eye_contact: Dict[str, float]
    hand_gestures: Dict[str, float]
    overall_score: float
    recommendations: List[str]

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
            head_centering_deviation = abs(face_center_x - 0.5)
            
            # Head tilt detection (eyes should be level)
            eye_level_difference = abs(left_eye_geometric_center_y - right_eye_geometric_center_y)
            
            # Calculate eye contact score (IMPROVED VERTICAL DETECTION)
            # Good eye contact = low gaze deviation + centered head + level eyes
            # Reduced vertical penalty to make vertical detection more sensitive
            gaze_score = max(0, 1 - (avg_gaze_offset_x * 2.5 + avg_gaze_offset_y * 1.2))
            head_position_score = max(0, 1 - (head_centering_deviation * 3.5))
            head_tilt_score = max(0, 1 - (eye_level_difference * 15))
            
            # Combined eye contact score (more weight on direct gaze)
            eye_contact_score = (gaze_score * 0.7 + head_position_score * 0.2 + head_tilt_score * 0.1)
            
            # Calculate confidence based on eye detection quality
            eye_detection_quality = min(left_eye_width, right_eye_width) * min(left_eye_height, right_eye_height)
            confidence = min(1.0, eye_detection_quality * 1000)  # Scale appropriately
            
            # Bonus confidence if iris tracking is available
            if iris_available:
                confidence = min(1.0, confidence * 1.2)
                
            # Overall gaze deviation for reporting
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
            # Fallback to basic analysis if advanced method fails
            logger.warning(f"Advanced eye tracking failed, using basic method: {e}")
            
            # Simple fallback method
            left_eye_x = (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2
            right_eye_x = (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2
            eye_center_x = (left_eye_x + right_eye_x) / 2
            
            # Basic centering score (STRICT FALLBACK)
            center_deviation = abs(eye_center_x - 0.5)
            basic_score = max(0, 1 - center_deviation * 3.5)
            
            return {
                "eye_contact_score": round(basic_score, 3),
                "gaze_deviation": round(center_deviation, 3),
                "confidence": 0.5,  # Lower confidence for fallback method
                "head_centering": round(basic_score, 3),
                "gaze_horizontal": round(center_deviation, 3),
                "gaze_vertical": 0.0
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
        
        # Get face boundaries with expanded region for early detection
        face_points = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        face_x_min = min(p[0] for p in face_points)
        face_x_max = max(p[0] for p in face_points)
        face_y_min = min(p[1] for p in face_points)
        face_y_max = max(p[1] for p in face_points)
        
        # Expand face region by 30% to catch approaching hands earlier
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
                
                # Check for face touching (using both exact and expanded regions)
                for point in hand_points:
                    # Direct face contact (higher score)
                    if (face_x_min <= point[0] <= face_x_max and 
                        face_y_min <= point[1] <= face_y_max):
                        face_touching_score += 0.5
                    # Approaching face (lower score for early detection)
                    elif (expanded_x_min <= point[0] <= expanded_x_max and 
                          expanded_y_min <= point[1] <= expanded_y_max):
                        face_touching_score += 0.2
                
                # Analyze hand position for nervous gestures
                wrist = hand_landmarks.landmark[0]
                middle_finger_tip = hand_landmarks.landmark[12]
                
                # Check if hands are in "nervous" positions (too close to face/neck area)
                if wrist.y < 0.6 and abs(wrist.x - 0.5) < 0.3:  # Near face/neck region
                    nervous_score += 0.3
                
                # Check for appropriate gesture range (natural speaking position)
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

def process_frame(image: np.ndarray) -> AnalysisResult:
    """Process a single frame and return analysis results"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = holistic.process(rgb_image)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Analyze each component
        posture_analysis = PostureAnalyzer.analyze_posture(results.pose_landmarks, width, height)
        eye_contact_analysis = EyeContactAnalyzer.analyze_eye_contact(results.face_landmarks)
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

@app.post("/analyze_frame", response_model=AnalysisResult)
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze a single frame for interview behavior"""
    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process the frame
        result = process_frame(image)
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_base64", response_model=AnalysisResult)
async def analyze_base64_frame(frame_data: Dict[str, str]):
    """Analyze a base64 encoded frame"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process the frame
        result = process_frame(image)
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_base64_frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Interview Analysis API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Interview Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze_frame": "POST - Upload image file for analysis",
            "/analyze_base64": "POST - Send base64 encoded image for analysis",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

def kill_process_on_port(port):
    try:
        import psutil
    except ImportError:
        print("psutil not installed. Please add it to requirements.txt.")
        return
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        for conn in proc.info.get('connections', []):
            if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                print(f"Killing process {proc.info['pid']} ({proc.info['name']}) on port {port}")
                try:
                    proc.kill()
                except Exception as e:
                    print(f"Could not kill process {proc.info['pid']}: {e}")

# Kill processes on ports 8000 and 8001 before starting the app
kill_process_on_port(8000)
kill_process_on_port(8001)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
