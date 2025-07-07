import cv2
import numpy as np
import mediapipe as mp
import time
import json
import os
from datetime import datetime
from collections import deque
import math
import threading
from pathlib import Path

# Import analysis classes from main.py
from main import PostureAnalyzer, EyeContactAnalyzer, HandGestureAnalyzer

class SessionRecorder:
    """Records video sessions and creates clips of behavioral events"""
    
    def __init__(self, session_id=None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording = False
        self.video_writer = None
        self.video_frames = []  # Store frames for clipping
        self.fps = 20  # Recording FPS
        
        # Create recordings directory
        self.recordings_dir = Path("interview_recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
        self.session_dir = self.recordings_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.full_video_path = self.session_dir / "full_session.mp4"
        self.clips_dir = self.session_dir / "clips"
        self.clips_dir.mkdir(exist_ok=True)
        
        print(f"📹 Session recorder initialized: {self.session_id}")
    
    def start_recording(self, frame_width, frame_height):
        """Start recording the session"""
        if self.recording:
            return False
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.full_video_path), 
            fourcc, 
            self.fps, 
            (frame_width, frame_height)
        )
        
        self.recording = True
        self.video_frames = []
        print(f"🔴 Recording started: {self.full_video_path}")
        return True
    
    def stop_recording(self):
        """Stop recording the session"""
        if not self.recording:
            return False
            
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        print(f"⏹️ Recording stopped. Video saved: {self.full_video_path}")
        return True
    
    def add_frame(self, frame):
        """Add a frame to the recording"""
        if not self.recording or self.video_writer is None:
            return
            
        # Write to video file
        self.video_writer.write(frame)
        
        # Store frame with timestamp for clipping (keep last 500 frames = ~25 seconds at 20fps)  
        # This ensures we have enough buffer for clips with up to 6s before + 2s after
        current_time = time.time()
        self.video_frames.append({
            'frame': frame.copy(),
            'timestamp': current_time
        })
        
        # Keep only recent frames to manage memory (increased for longer clips)
        if len(self.video_frames) > 500:
            self.video_frames.pop(0)
    
    def create_clip(self, event_name, start_time, end_time, session_start_time, buffer_before=3.0, buffer_after=2.0):
        """Create a video clip for a specific event with configurable buffers"""
        if not self.video_frames:
            return None
            
        # Convert absolute times to relative times
        rel_start = start_time
        rel_end = end_time
        
        # Find frames within the time range (with configurable buffer before/after)
        clip_start = max(0, rel_start - buffer_before)  # Don't go negative
        clip_end = rel_end + buffer_after
        
        # Find matching frames
        clip_frames = []
        current_time = time.time()
        
        for frame_data in self.video_frames:
            frame_rel_time = frame_data['timestamp'] - session_start_time
            if clip_start <= frame_rel_time <= clip_end:
                clip_frames.append(frame_data['frame'])
        
        if not clip_frames:
            print(f"⚠️ No frames found for clip: {event_name}")
            return None
        
        # Create clip video with buffer info in filename
        clip_filename = f"{event_name}_{clip_start:.1f}s_to_{clip_end:.1f}s_buffered.mp4"
        clip_path = self.clips_dir / clip_filename
        
        if clip_frames:
            height, width = clip_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            clip_writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (width, height))
            
            for frame in clip_frames:
                clip_writer.write(frame)
            
            clip_writer.release()
            print(f"🎬 Created clip: {clip_path}")
            return str(clip_path)
        
        return None

class BehaviorLogger:
    """Logs behavioral patterns and interview metrics over time"""
    
    def __init__(self, session_id=None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()  # Use time.time() for relative calculations
        self.session_start_datetime = datetime.now()  # Keep datetime for display
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "interview_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize session recorder
        self.recorder = SessionRecorder(self.session_id)
        
        # Initialize tracking variables
        self.eye_contact_history = deque(maxlen=50)  # Last 5 seconds at 10 FPS
        self.last_eye_contact_good = True
        self.eye_contact_break_start = None
        self.flicker_count = 0
        self.last_flicker_check = time.time()
        
        self.face_touch_count = 0
        self.face_touch_start = None
        self.last_face_touch_state = False
        
        self.bad_posture_start = None
        self.posture_history = deque(maxlen=50)
        
        # Event logs (now using relative timestamps)
        self.events = []
        self.session_stats = {
            'session_id': self.session_id,
            'start_time': self.session_start_datetime.isoformat(),
            'session_start_timestamp': self.session_start_time,
            'eye_contact_breaks': [],
            'face_touch_incidents': [],
            'bad_posture_periods': [],
            'flicker_events': [],
            'total_duration': 0,
            'avg_eye_contact_score': 0,
            'avg_posture_score': 0,
            'total_face_touches': 0
        }
        
        print(f"📊 Behavior logging started - Session ID: {self.session_id}")
    
    def log_analysis(self, posture_analysis, eye_contact_analysis, hand_gesture_analysis):
        """Log current analysis and detect behavioral patterns"""
        current_time = time.time()
        relative_time = current_time - self.session_start_time  # Time since session start
        timestamp = datetime.now()
        
        # Use calibrated thresholds for logging decisions
        cal_thresholds = self.eye_contact_calibrator.get_calibrated_thresholds() if hasattr(self, 'eye_contact_calibrator') else {'good_threshold': 0.60}
        eye_contact_good = eye_contact_analysis.get('eye_contact_score', 0) >= cal_thresholds['good_threshold']
        self.eye_contact_history.append(eye_contact_good)
        
        # Detect eye contact breaks (only log sustained breaks > 1.5 seconds)
        if self.last_eye_contact_good and not eye_contact_good:
            # Eye contact just broke - start tracking but don't log yet
            if self.eye_contact_break_start is None:
                self.eye_contact_break_start = current_time
        elif not self.last_eye_contact_good and eye_contact_good:
            # Eye contact restored
            if self.eye_contact_break_start:
                break_duration = current_time - self.eye_contact_break_start
                # Only log breaks longer than 1.5 seconds (more conservative)
                if break_duration > 1.5:
                    start_rel_time = self.eye_contact_break_start - self.session_start_time
                    end_rel_time = current_time - self.session_start_time
                    
                    event_data = {
                        'start_time_relative': start_rel_time,
                        'end_time_relative': end_rel_time,
                        'duration': break_duration,
                        'severity': 'high' if break_duration > 3.0 else 'medium',
                        'score_start': eye_contact_analysis.get('eye_contact_score', 0)
                    }
                    
                    self.session_stats['eye_contact_breaks'].append(event_data)
                    self.log_event("eye_contact_break", relative_time, event_data)
                    
                    # Create video clip for this event (3s before, 2s after)
                    if self.recorder.recording:
                        threading.Thread(
                            target=self.recorder.create_clip,
                            args=("eye_contact_break", start_rel_time, end_rel_time, self.session_start_time, 3.0, 2.0),
                            daemon=True
                        ).start()
                    
                    print(f"🔴 Logged eye contact break: {break_duration:.1f}s at {start_rel_time:.1f}s")
                else:
                    print(f"⚪ Brief eye contact break ignored: {break_duration:.1f}s")
                self.eye_contact_break_start = None
        
        # Detect eye contact flickering (rapid on/off pattern, excluding normal blinks)
        if current_time - self.last_flicker_check > 3.0:  # Check every 3 seconds
            if len(self.eye_contact_history) >= 30:  # Need at least 3 seconds of data
                recent_changes = sum(1 for i in range(1, len(self.eye_contact_history)) 
                                   if self.eye_contact_history[i] != self.eye_contact_history[i-1])
                # Increased threshold to avoid normal blinks being counted as flicker
                if recent_changes > 10:  # More than ~3 state changes per second = problematic flickering
                    self.flicker_count += 1
                    event_data = {
                        'timestamp_relative': relative_time,
                        'changes_per_second': recent_changes / 3.0,
                        'changes_detected': recent_changes,
                        'severity': 'high' if recent_changes > 15 else 'medium'
                    }
                    
                    self.session_stats['flicker_events'].append(event_data)
                    self.log_event("eye_contact_flicker", relative_time, event_data)
            self.last_flicker_check = current_time
        
        self.last_eye_contact_good = eye_contact_good
        
        # Track face touching patterns (with duration tracking like other behaviors)
        face_touching = hand_gesture_analysis.get('face_touching', 0) > 0.25  # Lowered from 0.3 to 0.25
        touch_score = hand_gesture_analysis.get('face_touching', 0)
        
        # Debug: Show face touching score for debugging
        if touch_score > 0.1:  # Show when hands are getting close
            print(f"🤏 Hand proximity score: {touch_score:.2f} at {relative_time:.1f}s")
        
        # Track face touching duration (similar to eye contact breaks and bad posture)
        if not self.last_face_touch_state and face_touching:
            # Face touching just started - begin tracking
            if self.face_touch_start is None:
                self.face_touch_start = current_time
                print(f"👋 Face touch started at {relative_time:.1f}s (score: {touch_score:.2f})")
        elif self.last_face_touch_state and not face_touching:
            # Face touching stopped
            if self.face_touch_start:
                touch_duration = current_time - self.face_touch_start
                # Only log sustained face touching (>0.5 seconds to avoid quick gestures)
                if touch_duration > 0.5:
                    self.face_touch_count += 1
                    start_rel_time = self.face_touch_start - self.session_start_time
                    end_rel_time = current_time - self.session_start_time
                    
                    event_data = {
                        'start_time_relative': start_rel_time,
                        'end_time_relative': end_rel_time,
                        'duration': touch_duration,
                        'severity': 'high' if touch_duration > 2.0 else 'medium',
                        'max_score': touch_score,
                        'total_count': self.face_touch_count
                    }
                    
                    self.session_stats['face_touch_incidents'].append(event_data)
                    self.log_event("face_touch", end_rel_time, event_data)
                    
                    # Create video clip using actual event duration (1s before start, 1s after end)
                    if self.recorder.recording:
                        print(f"🎬 Creating face touch clip: {touch_duration:.1f}s event from {start_rel_time:.1f}s to {end_rel_time:.1f}s")
                        
                        threading.Thread(
                            target=self.recorder.create_clip,
                            args=("face_touch", start_rel_time, end_rel_time, self.session_start_time, 1.0, 1.0),
                            daemon=True
                        ).start()
                    
                    print(f"🔴 Logged face touch: {touch_duration:.1f}s from {start_rel_time:.1f}s to {end_rel_time:.1f}s")
                else:
                    print(f"⚪ Brief face touch ignored: {touch_duration:.1f}s")
                self.face_touch_start = None
        
        self.last_face_touch_state = face_touching
        
        # Track posture patterns
        posture_good = posture_analysis.get('confidence', 0) > 0.7
        self.posture_history.append(posture_good)
        
        if not posture_good and self.bad_posture_start is None:
            # Bad posture period started
            self.bad_posture_start = current_time
            self.log_event("bad_posture_start", relative_time, {
                'confidence': posture_analysis.get('confidence', 0),
                'alignment': posture_analysis.get('alignment', 0),
                'centering': posture_analysis.get('centering', 0),
                'straightness': posture_analysis.get('straightness', 0)
            })
        elif posture_good and self.bad_posture_start is not None:
            # Bad posture period ended
            duration = current_time - self.bad_posture_start
            if duration > 3.0:  # Only log sustained bad posture
                start_rel_time = self.bad_posture_start - self.session_start_time
                end_rel_time = current_time - self.session_start_time
                
                event_data = {
                    'start_time_relative': start_rel_time,
                    'end_time_relative': end_rel_time,
                    'duration': duration,
                    'severity': 'high' if duration > 10.0 else 'medium'
                }
                
                self.session_stats['bad_posture_periods'].append(event_data)
                self.log_event("bad_posture_end", relative_time, event_data)
                
                # Create video clip for bad posture (2s before, 2s after)
                if self.recorder.recording:
                    threading.Thread(
                        target=self.recorder.create_clip,
                        args=("bad_posture", start_rel_time, end_rel_time, self.session_start_time, 2.0, 2.0),
                        daemon=True
                    ).start()
                    
            self.bad_posture_start = None
    
    def log_event(self, event_type, relative_time, details):
        """Log a specific behavioral event"""
        self.events.append({
            'timestamp_relative': relative_time,
            'event_type': event_type,
            'details': details
        })
    

    
    def get_session_summary(self):
        """Get current session statistics"""
        duration = time.time() - self.session_start_time
        self.session_stats['total_duration'] = duration
        self.session_stats['total_face_touches'] = self.face_touch_count
        
        return {
            'session_id': self.session_id,
            'session_duration': duration,
            'eye_contact_breaks': self.session_stats['eye_contact_breaks'],
            'total_break_time': sum(b['duration'] for b in self.session_stats['eye_contact_breaks']),
            'face_touch_incidents': self.session_stats['face_touch_incidents'],
            'total_face_touch_time': sum(f.get('duration', 0) for f in self.session_stats['face_touch_incidents']),
            'face_touches_count': self.face_touch_count,
            'bad_posture_periods': self.session_stats['bad_posture_periods'],
            'total_bad_posture_time': sum(p['duration'] for p in self.session_stats['bad_posture_periods']),
            'flicker_events': len(self.session_stats['flicker_events']),
            'total_events': len(self.events)
        }
    
    def save_session_log(self):
        """Save complete session log to file"""
        # Update session_stats with final values
        self.session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats['total_duration'] = time.time() - self.session_start_time
        self.session_stats['total_face_touches'] = self.face_touch_count
        
        # Save detailed log - session_stats should be the root object for rating API compatibility
        log_file = os.path.join(self.logs_dir, f"session_{self.session_id}.json")
        with open(log_file, 'w') as f:
            json.dump({
                'session_stats': self.session_stats,
                'events': self.events
            }, f, indent=2)
        
        # Save summary
        summary_file = os.path.join(self.logs_dir, f"summary_{self.session_id}.json")
        with open(summary_file, 'w') as f:
            json.dump(self.get_session_summary(), f, indent=2)
        
        print(f"📁 Session logs saved: {log_file}")
        return log_file

class EyeContactCalibrator:
    """Calibrates eye contact detection based on user's natural camera-looking position"""
    
    def __init__(self):
        self.calibration_samples = []
        self.baseline_gaze_horizontal = 0.0
        self.baseline_gaze_vertical = 0.0
        self.baseline_head_centering = 0.5
        self.baseline_eye_contact_score = 0.0
        self.calibrated = False
        
    def add_calibration_sample(self, eye_analysis):
        """Add a sample during calibration phase"""
        if eye_analysis['confidence'] > 0.5:  # Only use good quality samples
            self.calibration_samples.append({
                'gaze_horizontal': eye_analysis.get('gaze_horizontal', 0),
                'gaze_vertical': eye_analysis.get('gaze_vertical', 0),
                'head_centering': eye_analysis.get('head_centering', 0.5),
                'eye_contact_score': eye_analysis.get('eye_contact_score', 0)
            })
    
    def finalize_calibration(self):
        """Calculate baseline values from collected samples"""
        if len(self.calibration_samples) < 10:  # Need at least 10 good samples
            print("⚠️ Not enough calibration samples, using default thresholds")
            return False
        
        # Calculate average baseline values
        self.baseline_gaze_horizontal = np.mean([s['gaze_horizontal'] for s in self.calibration_samples])
        self.baseline_gaze_vertical = np.mean([s['gaze_vertical'] for s in self.calibration_samples])
        self.baseline_head_centering = np.mean([s['head_centering'] for s in self.calibration_samples])
        self.baseline_eye_contact_score = np.mean([s['eye_contact_score'] for s in self.calibration_samples])
        
        self.calibrated = True
        
        print(f"✅ Calibration complete!")
        print(f"   Baseline gaze horizontal: {self.baseline_gaze_horizontal:.3f}")
        print(f"   Baseline gaze vertical: {self.baseline_gaze_vertical:.3f}")
        print(f"   Baseline head centering: {self.baseline_head_centering:.3f}")
        print(f"   Baseline eye contact score: {self.baseline_eye_contact_score:.3f}")
        
        return True
    
    def get_calibrated_thresholds(self):
        """Get dynamic thresholds based on calibration"""
        if not self.calibrated:
            # Return default thresholds if not calibrated
            return {
                'good_threshold': 0.60,
                'bad_threshold': 0.50,
                'gaze_tolerance_h': 0.25,
                'gaze_tolerance_v': 0.25,
                'head_centering_tolerance': 0.25
            }
        
        # Dynamic thresholds based on baseline
        # Good eye contact = within reasonable deviation from baseline
        good_threshold = max(0.5, self.baseline_eye_contact_score - 0.15)
        bad_threshold = max(0.3, self.baseline_eye_contact_score - 0.30)
        
        return {
            'good_threshold': good_threshold,
            'bad_threshold': bad_threshold,
            'gaze_tolerance_h': self.baseline_gaze_horizontal + 0.15,
            'gaze_tolerance_v': self.baseline_gaze_vertical + 0.15,
            'head_centering_tolerance': 0.25
        }
    
    def evaluate_eye_contact(self, eye_analysis):
        """Evaluate eye contact quality based on calibrated baseline"""
        if not self.calibrated or eye_analysis['confidence'] < 0.5:
            return self._default_evaluation(eye_analysis)
        
        thresholds = self.get_calibrated_thresholds()
        score = eye_analysis['eye_contact_score']
        
        # Simple binary: good or bad (no neutral/yellow state)
        if score >= thresholds['good_threshold']:
            return 'good'
        else:
            return 'bad'
    
    def _default_evaluation(self, eye_analysis):
        """Fallback evaluation when not calibrated"""
        score = eye_analysis['eye_contact_score']
        if score >= 0.60:
            return 'good'
        else:
            return 'bad'

class LiveAnalysisUI:
    def __init__(self, camera_index=0):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Analysis components
        self.posture_analyzer = PostureAnalyzer()
        self.eye_contact_analyzer = EyeContactAnalyzer()
        self.hand_gesture_analyzer = HandGestureAnalyzer()
        
        # Eye contact calibration
        self.eye_contact_calibrator = EyeContactCalibrator()
        self.calibration_phase = True
        self.calibration_start_time = None
        self.calibration_duration = 5.0  # 5 seconds
        
        # Session recording
        self.session_active = False
        self.session_start_time = None
        
        # UI state
        self.message_queue = deque(maxlen=10)  # Keep last 10 messages
        self.last_analysis_time = 0
        self.analysis_interval = 0.1  # Analyze every 100ms (10 FPS)
        
        # Colors for drawing
        self.colors = {
            'good': (0, 255, 0),      # Green
            'warning': (0, 165, 255),  # Orange
            'bad': (0, 0, 255),       # Red
            'neutral': (255, 255, 255), # White
            'text_bg': (0, 0, 0),     # Black
            'calibration': (255, 255, 0), # Yellow for calibration
        }
        
        # Warning thresholds (will be updated after calibration)
        self.thresholds = {
            'posture_confidence': 0.7,
            'eye_contact_score': 0.60,  # Will be updated after calibration
            'face_touching': 0.3,
            'nervous_gestures': 0.5,
            'head_tilt': 0.7
        }
        
        # Visual indicator state
        self.indicators = {
            'head_tilt': {'active': False, 'message': '', 'color': 'good'},
            'posture': {'active': False, 'message': '', 'color': 'good'},
            'eye_contact': {'active': False, 'message': '', 'color': 'good'},
            'hand_gestures': {'active': False, 'message': '', 'color': 'good'}
        }
        
        # Initialize behavior logger (will be created when session starts)
        self.behavior_logger = None
        
        # Separate timing for UI vs logging
        self.ui_responsiveness = {
            'eye_contact_ui_delay': 0.3,  # 300ms delay for UI (reduces blink flicker)
            'last_ui_change': 0
        }
        
        self.logging_criteria = {
            'eye_contact_log_threshold': 1.5,  # Only log issues lasting 1.5+ seconds
            'last_bad_start': 0,
            'current_issue_logged': False
        }
        
    def start_calibration(self):
        """Start the calibration phase"""
        print("\n" + "="*60)
        print("👁️  EYE CONTACT CALIBRATION")
        print("="*60)
        print("Please look directly at your camera for 5 seconds.")
        print("Keep your head in a natural position and maintain steady eye contact.")
        print("This will help customize the detection for your setup.")
        print("Calibration will begin in 3 seconds...")
        print("="*60)
        
        self.calibration_phase = True
        self.calibration_start_time = None
        
    def update_calibration(self, results):
        """Update calibration during the calibration phase"""
        if not self.calibration_phase:
            return
            
        current_time = time.time()
        
        if self.calibration_start_time is None:
            self.calibration_start_time = current_time
            return
            
        elapsed = current_time - self.calibration_start_time
        
        if elapsed >= self.calibration_duration:
            # Calibration complete
            if self.eye_contact_calibrator.finalize_calibration():
                # Update thresholds based on calibration
                cal_thresholds = self.eye_contact_calibrator.get_calibrated_thresholds()
                self.thresholds['eye_contact_score'] = cal_thresholds['good_threshold']
                print(f"Updated eye contact threshold to: {cal_thresholds['good_threshold']:.2f}")
            
            self.calibration_phase = False
            print("\n🚀 Calibration complete! Press SPACE to start recording session...")
            return
        
        # Collect calibration samples
        if results.face_landmarks:
            eye_analysis = self.eye_contact_analyzer.analyze_eye_contact(results.face_landmarks)
            self.eye_contact_calibrator.add_calibration_sample(eye_analysis)
    
    def draw_calibration_overlay(self, image):
        """Draw calibration UI overlay"""
        h, w = image.shape[:2]
        
        if self.calibration_start_time is None:
            # Show countdown
            cv2.putText(image, "CALIBRATION STARTING...", (w//2 - 150, h//2 - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['calibration'], 3)
            cv2.putText(image, "Look at the camera", (w//2 - 100, h//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['calibration'], 2)
            return
        
        elapsed = time.time() - self.calibration_start_time
        remaining = max(0, self.calibration_duration - elapsed)
        
        # Semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Calibration instructions
        cv2.putText(image, "CALIBRATING EYE CONTACT", (w//2 - 200, h//2 - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['calibration'], 3)
        
        cv2.putText(image, "Keep looking at the camera", (w//2 - 150, h//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['calibration'], 2)
        
        # Countdown timer
        cv2.putText(image, f"Time remaining: {remaining:.1f}s", (w//2 - 120, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['calibration'], 2)
        
        # Sample count
        sample_count = len(self.eye_contact_calibrator.calibration_samples)
        cv2.putText(image, f"Samples collected: {sample_count}", (w//2 - 100, h//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['calibration'], 2)
        
        # Draw target circle at center
        center = (w//2, h//2 + 100)
        cv2.circle(image, center, 30, self.colors['calibration'], 3)
        cv2.circle(image, center, 10, self.colors['calibration'], -1)

    def analyze_and_generate_messages(self, results, image_width, image_height):
        """Analyze results and generate appropriate messages"""
        current_time = time.time()
        
        # Handle calibration phase
        if self.calibration_phase:
            self.update_calibration(results)
            return
        
        # Skip analysis if no active session
        if not self.session_active or not self.behavior_logger:
            return
        
        # Only analyze every 100ms to match 10 FPS requirement
        if current_time - self.last_analysis_time < self.analysis_interval:
            return

        self.last_analysis_time = current_time
        
        # Perform analysis
        posture_analysis = self.posture_analyzer.analyze_posture(results.pose_landmarks, image_width, image_height)
        eye_contact_analysis = self.eye_contact_analyzer.analyze_eye_contact(results.face_landmarks)
        hand_gesture_analysis = self.hand_gesture_analyzer.analyze_hand_gestures(
            results.left_hand_landmarks, 
            results.right_hand_landmarks, 
            results.face_landmarks
        )
        
        # Store latest analysis for UI display
        self._last_analysis = {
            'posture': posture_analysis,
            'eye_contact': eye_contact_analysis,
            'hand_gestures': hand_gesture_analysis
        }
        
        # Log behavioral patterns (use calibrated thresholds)
        cal_thresholds = self.eye_contact_calibrator.get_calibrated_thresholds()
        
        # Update behavior logger with calibrated thresholds
        original_threshold = self.behavior_logger.__dict__.get('eye_contact_threshold', 0.60)
        self.behavior_logger.__dict__['eye_contact_threshold'] = cal_thresholds['good_threshold']
        
        self.behavior_logger.log_analysis(posture_analysis, eye_contact_analysis, hand_gesture_analysis)
        
        # Update visual indicators based on analysis
        # Reset all indicators
        for indicator in self.indicators.values():
            indicator['active'] = False
            indicator['message'] = ''
            indicator['color'] = 'good'
        
        # Head tilt indicator
        if posture_analysis['confidence'] > 0 and 'head_tilt' in posture_analysis:
            if posture_analysis['head_tilt'] < self.thresholds['head_tilt']:
                self.indicators['head_tilt']['active'] = True
                self.indicators['head_tilt']['message'] = 'HEAD TILTED'
                self.indicators['head_tilt']['color'] = 'bad'
        
        # Posture indicator (excluding head tilt)
        if posture_analysis['confidence'] > 0:
            posture_issues = []
            if posture_analysis['alignment'] < 0.6:
                posture_issues.append('SHOULDERS')
            if posture_analysis['centering'] < 0.6:
                posture_issues.append('CENTERING')
            if posture_analysis['straightness'] < 0.6:
                posture_issues.append('SPINE')
            
            if posture_issues:
                self.indicators['posture']['active'] = True
                self.indicators['posture']['message'] = ' & '.join(posture_issues)
                self.indicators['posture']['color'] = 'bad' if len(posture_issues) > 1 else 'warning'
        
        # Eye contact indicator (immediate UI response with minimal delay)
        if eye_contact_analysis['confidence'] > 0:
            current_time = time.time()
            eye_contact_quality = self.eye_contact_calibrator.evaluate_eye_contact(eye_contact_analysis)
            eye_contact_poor = eye_contact_quality == 'bad'
            
            if eye_contact_poor:
                # Record when poor eye contact started for UI
                if self.ui_responsiveness['last_ui_change'] == 0:
                    self.ui_responsiveness['last_ui_change'] = current_time
                
                # Show indicator after minimal delay (300ms to reduce blink flicker)
                time_poor = current_time - self.ui_responsiveness['last_ui_change']
                if time_poor >= self.ui_responsiveness['eye_contact_ui_delay']:
                    self.indicators['eye_contact']['active'] = True
                    
                    # Determine specific issue using calibrated thresholds
                    cal_thresholds = self.eye_contact_calibrator.get_calibrated_thresholds()
                    
                    # More sensitive vertical detection
                    vertical_threshold = cal_thresholds.get('gaze_tolerance_v', 0.25) * 0.7  # Lower threshold for vertical
                    horizontal_threshold = cal_thresholds.get('gaze_tolerance_h', 0.25)
                    
                    if 'gaze_horizontal' in eye_contact_analysis and eye_contact_analysis['gaze_horizontal'] > horizontal_threshold:
                        self.indicators['eye_contact']['message'] = 'LOOKING L/R'
                    elif 'gaze_vertical' in eye_contact_analysis and eye_contact_analysis['gaze_vertical'] > vertical_threshold:
                        self.indicators['eye_contact']['message'] = 'LOOKING U/D'
                    elif 'head_centering' in eye_contact_analysis and eye_contact_analysis['head_centering'] < (self.eye_contact_calibrator.baseline_head_centering - cal_thresholds['head_centering_tolerance']):
                        self.indicators['eye_contact']['message'] = 'OFF-CENTER'
                    else:
                        self.indicators['eye_contact']['message'] = 'POOR CONTACT'
                    
                    # Simple binary color: only good or bad (no yellow/warning)
                    self.indicators['eye_contact']['color'] = 'bad'
                
                # Separate logging logic - track sustained issues
                if self.logging_criteria['last_bad_start'] == 0:
                    self.logging_criteria['last_bad_start'] = current_time
                    self.logging_criteria['current_issue_logged'] = False
                
                # Check if this issue should be logged (sustained for 1.5+ seconds)
                sustained_duration = current_time - self.logging_criteria['last_bad_start']
                if sustained_duration >= self.logging_criteria['eye_contact_log_threshold'] and not self.logging_criteria['current_issue_logged']:
                    # This is now a significant issue worth logging
                    print(f"📊 Eye contact issue now qualifies for logging: {sustained_duration:.1f}s")
                    self.logging_criteria['current_issue_logged'] = True
                    
            else:
                # Reset UI timer when eye contact is good
                self.ui_responsiveness['last_ui_change'] = 0
                
                # Reset logging timer and log resolution if there was a logged issue
                if self.logging_criteria['last_bad_start'] > 0:
                    issue_duration = current_time - self.logging_criteria['last_bad_start']
                    if self.logging_criteria['current_issue_logged']:
                        print(f"✅ Logged eye contact issue resolved after: {issue_duration:.1f}s")
                    else:
                        print(f"⚪ Brief eye contact issue (not logged): {issue_duration:.1f}s")
                    
                    self.logging_criteria['last_bad_start'] = 0
                    self.logging_criteria['current_issue_logged'] = False
        
        # Hand gesture indicator (updated threshold to match detection)
        if hand_gesture_analysis['confidence'] > 0:
            hand_issues = []
            if hand_gesture_analysis['face_touching'] > 0.25:  # Match the lowered detection threshold
                hand_issues.append('FACE TOUCH')
            if hand_gesture_analysis['nervous_gestures'] > self.thresholds['nervous_gestures']:
                hand_issues.append('NERVOUS')
            
            if hand_issues:
                self.indicators['hand_gestures']['active'] = True
                self.indicators['hand_gestures']['message'] = ' & '.join(hand_issues)
                self.indicators['hand_gestures']['color'] = 'bad' if 'FACE TOUCH' in hand_issues else 'warning'
    
    def draw_ui_overlay(self, image):
        """Draw UI overlay with visual indicator boxes"""
        # Handle calibration phase
        if self.calibration_phase:
            self.draw_calibration_overlay(image)
            return
            
        h, w = image.shape[:2]
        
        # Draw title with session status
        title = "INTERVIEW ANALYSIS"
        if self.eye_contact_calibrator.calibrated:
            title += " (CALIBRATED)"
        if self.session_active:
            title += " - RECORDING 🔴"
            color = self.colors['bad']  # Red when recording
        else:
            title += " - READY (Press SPACE to start)"
            color = self.colors['good']
            
        cv2.putText(image, title, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw visual indicator boxes at the top
        box_width = 150
        box_height = 80
        box_margin = 10
        start_x = 20
        start_y = 50
        
        indicators_info = [
            ('head_tilt', 'HEAD TILT'),
            ('posture', 'POSTURE'),
            ('eye_contact', 'EYE CONTACT'),
            ('hand_gestures', 'HANDS')
        ]
        
        for i, (key, title) in enumerate(indicators_info):
            # Calculate box position
            box_x = start_x + i * (box_width + box_margin)
            box_y = start_y
            
            # Get indicator state
            indicator = self.indicators[key]
            
            # Determine box color based on state
            if indicator['active']:
                if indicator['color'] == 'bad':
                    box_color = (0, 0, 255)  # Red
                    text_color = (255, 255, 255)  # White text
                elif indicator['color'] == 'warning':
                    box_color = (0, 165, 255)  # Orange
                    text_color = (255, 255, 255)  # White text
                else:
                    box_color = (0, 255, 0)  # Green
                    text_color = (0, 0, 0)  # Black text
            else:
                box_color = (0, 200, 0)  # Light green (good/inactive)
                text_color = (0, 0, 0)  # Black text
            
            # Draw box with rounded corners effect
            cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), box_color, -1)
            cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)
            
            # Draw title
            cv2.putText(image, title, (box_x + 5, box_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
            
            # Draw status
            if indicator['active'] and indicator['message']:
                # Split long messages to fit in box
                message = indicator['message']
                if len(message) > 12:
                    # Split message into two lines
                    words = message.split()
                    if len(words) > 1:
                        mid = len(words) // 2
                        line1 = ' '.join(words[:mid])
                        line2 = ' '.join(words[mid:])
                        cv2.putText(image, line1, (box_x + 5, box_y + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
                        cv2.putText(image, line2, (box_x + 5, box_y + 55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
                    else:
                        cv2.putText(image, message[:12], (box_x + 5, box_y + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
                else:
                    cv2.putText(image, message, (box_x + 5, box_y + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
                
                # Add pulsing effect for active warnings
                if indicator['color'] in ['bad', 'warning']:
                    pulse = int(abs(np.sin(time.time() * 3)) * 20)  # Pulsing border
                    cv2.rectangle(image, (box_x - 2, box_y - 2), 
                                 (box_x + box_width + 2, box_y + box_height + 2), 
                                 box_color, pulse // 10)
            else:
                cv2.putText(image, "OK", (box_x + 5, box_y + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Draw real-time metrics (bottom right)
        metrics_x = w - 200
        metrics_y = h - 120
        
        # Semi-transparent background for metrics
        overlay = image.copy()
        cv2.rectangle(overlay, (metrics_x - 10, metrics_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        cv2.putText(image, "LIVE METRICS:", (metrics_x, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['neutral'], 1)
        
        # Display current analysis scores
        if hasattr(self, '_last_analysis'):
            analysis = self._last_analysis
            y_offset = metrics_y + 20
            
            # Posture score
            if 'posture' in analysis:
                score = analysis['posture']['confidence']
                color = self.colors['good'] if score > 0.7 else self.colors['warning'] if score > 0.4 else self.colors['bad']
                cv2.putText(image, f"Posture: {score:.1%}", 
                           (metrics_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                y_offset += 15
            
            # Eye contact score (using calibrated thresholds)
            if 'eye_contact' in analysis:
                score = analysis['eye_contact']['eye_contact_score']
                cal_thresholds = self.eye_contact_calibrator.get_calibrated_thresholds()
                # Simple binary: green or red only
                if score >= cal_thresholds['good_threshold']:
                    color = self.colors['good']
                else:
                    color = self.colors['bad']
                    
                cv2.putText(image, f"Eye Contact: {score:.1%}", 
                           (metrics_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                y_offset += 15
                
                # Show calibrated threshold
                if self.eye_contact_calibrator.calibrated:
                    cv2.putText(image, f"(Threshold: {cal_thresholds['good_threshold']:.1%})", 
                               (metrics_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.25, self.colors['neutral'], 1)
                    y_offset += 12
            
            # Hand gesture score (updated threshold)
            if 'hand_gestures' in analysis:
                confidence = analysis['hand_gestures']['confidence']
                face_touch = analysis['hand_gestures']['face_touching']
                color = self.colors['bad'] if face_touch > 0.25 else self.colors['good']  # Match detection threshold
                cv2.putText(image, f"Hands: {confidence:.1%}", 
                           (metrics_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                y_offset += 15
        
        # Draw FPS info (top right)
        cv2.putText(image, f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))} | Analysis: 10Hz", 
                   (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['neutral'], 1)
        
        # Draw session stats (bottom left) 
        if self.session_active and self.behavior_logger:
            session_summary = self.behavior_logger.get_session_summary()
            stats_y = h - 60
            cv2.putText(image, f"SESSION: {session_summary['session_duration']:.1f}s | Breaks: {len(session_summary['eye_contact_breaks'])} | Touches: {session_summary['face_touches_count']}", 
                       (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['neutral'], 1)
        
        # Draw instructions (bottom left)
        if self.session_active:
            instructions = [
                "CONTROLS: SPACE=Stop Session | R=Recalibrate | S=Save Log | Q=Quit"
            ]
        else:
            instructions = [
                "CONTROLS: SPACE=Start Session | R=Recalibrate | Q=Quit"
            ]
        
        cv2.putText(image, instructions[0], (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['neutral'], 1)
    
    def run(self):
        """Main loop for live analysis"""
        # Start with calibration
        self.start_calibration()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.holistic.process(rgb_frame)
                
                # Draw MediaPipe landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                if results.face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())
                
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())
                
                # Perform analysis and update UI
                self.analyze_and_generate_messages(results, frame.shape[1], frame.shape[0])
                
                # Draw UI overlay
                self.draw_ui_overlay(frame)
                
                # Display frame
                cv2.imshow('Interview Analysis - Live UI', frame)
                
                # Add frame to recording if session is active
                if self.session_active and self.behavior_logger:
                    self.behavior_logger.recorder.add_frame(frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # SPACE key
                    if self.calibration_phase:
                        continue
                    elif not self.session_active:
                        # Start session
                        self.start_session(frame.shape[1], frame.shape[0])
                    else:
                        # Stop session
                        self.stop_session()
                elif key == ord('r') and not self.calibration_phase:
                    # Recalibrate
                    print("\n🔄 Recalibrating eye contact...")
                    self.start_calibration()
                elif key == ord('s') and not self.calibration_phase and self.behavior_logger:
                    # Save session summary
                    self.behavior_logger.save_session_log()
                    print("📊 Session summary saved!")
                
        except KeyboardInterrupt:
            print("\nStopping analysis...")
        
        finally:
            # Cleanup
            if self.session_active:
                self.stop_session()
            
            self.cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
    
    def start_session(self, frame_width, frame_height):
        """Start a new recording session"""
        if self.session_active:
            return
            
        print("\n🚀 Starting new session...")
        self.session_active = True
        self.session_start_time = time.time()
        
        # Create new behavior logger for this session
        self.behavior_logger = BehaviorLogger()
        
        # Start video recording
        if self.behavior_logger.recorder.start_recording(frame_width, frame_height):
            print(f"📹 Recording started - Session: {self.behavior_logger.session_id}")
        else:
            print("❌ Failed to start recording")
    
    def stop_session(self):
        """Stop the current recording session"""
        if not self.session_active or not self.behavior_logger:
            return
            
        print("\n⏹️ Stopping session...")
        self.session_active = False
        
        # Stop video recording
        self.behavior_logger.recorder.stop_recording()
        
        # Save session data
        log_file = self.behavior_logger.save_session_log()
        summary = self.behavior_logger.get_session_summary()
        
        print(f"📊 Session Summary:")
        print(f"   Duration: {summary['session_duration']:.1f} seconds")
        print(f"   Eye contact breaks: {len(summary['eye_contact_breaks'])}")
        print(f"   Face touches: {summary['face_touches_count']}")
        print(f"   Bad posture periods: {len(summary['bad_posture_periods'])}")
        print(f"   Video clips created: Check {self.behavior_logger.recorder.clips_dir}")
        print(f"   Full session video: {self.behavior_logger.recorder.full_video_path}")
        
        self.behavior_logger = None

def main():
    """Run the live analysis UI"""
    print("🎥 INTERVIEW ANALYSIS WITH VIDEO RECORDING")
    print("="*50)
    print("Controls:")
    print("  SPACE - Start/Stop recording session")
    print("  R - Recalibrate eye contact")
    print("  S - Save session log manually")
    print("  Q - Quit application")
    print("\nFeatures:")
    print("✅ Eye contact calibration for personalized thresholds")
    print("📹 Full session video recording") 
    print("🎬 Automatic video clips of behavioral events")
    print("📊 Detailed behavioral analysis and logging")
    print("⚡ Real-time visual feedback")
    print("\nMake sure you're well-lit and centered in the camera frame.")
    print("="*50)
    
    try:
        ui = LiveAnalysisUI(camera_index=0)
        ui.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is available and MediaPipe is properly installed.")

if __name__ == "__main__":
    main() 