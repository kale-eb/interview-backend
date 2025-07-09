#!/usr/bin/env python3
"""
Live Interview Analysis UI for ECS Deployment
Uses MediaPipe for advanced video analysis
"""

import cv2
import numpy as np
import requests
import json
import base64
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import os
from PIL import Image, ImageTk

class ECSInterviewUI:
    def __init__(self, api_endpoint, rating_endpoint="http://localhost:8001"):
        self.api_endpoint = api_endpoint
        self.rating_endpoint = rating_endpoint
        self.session_id = f"session_{int(time.time())}"
        self.user_id = "user_001"
        
        # Video capture
        self.cap = None
        self.is_recording = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.analysis_queue = queue.Queue()
        
        # Analysis results
        self.current_analysis = {
            "eye_contact_score": 0.0,
            "posture_score": 0.0,
            "face_touching": False,
            "overall_score": 0.0
        }
        
        # Session logging (similar to local version)
        self.session_start_time = None
        self.session_events = []
        self.session_stats = {
            "eye_contact_breaks": [],
            "face_touch_incidents": [],
            "bad_posture_periods": [],
            "flicker_events": []
        }
        
        # Statistics
        self.frame_count = 0
        self.analysis_count = 0
        self.start_time = None
        
        # Create logs directory
        os.makedirs("interview_logs", exist_ok=True)
        os.makedirs("interview_recordings", exist_ok=True)
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        self.root = tk.Tk()
        self.root.title("Interview Analysis - ECS Version")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_label = ttk.Label(main_frame, text="Video Feed")
        self.video_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Analysis", command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Analysis", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Analysis results
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Eye contact
        ttk.Label(results_frame, text="Eye Contact:").grid(row=0, column=0, sticky=tk.W)
        self.eye_contact_label = ttk.Label(results_frame, text="0.0")
        self.eye_contact_label.grid(row=0, column=1, padx=10)
        
        # Posture
        ttk.Label(results_frame, text="Posture:").grid(row=1, column=0, sticky=tk.W)
        self.posture_label = ttk.Label(results_frame, text="0.0")
        self.posture_label.grid(row=1, column=1, padx=10)
        
        # Face touching
        ttk.Label(results_frame, text="Face Touching:").grid(row=2, column=0, sticky=tk.W)
        self.face_touching_label = ttk.Label(results_frame, text="No")
        self.face_touching_label.grid(row=2, column=1, padx=10)
        
        # Overall score
        ttk.Label(results_frame, text="Overall Score:").grid(row=3, column=0, sticky=tk.W)
        self.overall_score_label = ttk.Label(results_frame, text="0.0", font=("Arial", 12, "bold"))
        self.overall_score_label.grid(row=3, column=1, padx=10)
        
        # Statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(stats_frame, text="Frames Processed:").grid(row=0, column=0, sticky=tk.W)
        self.frames_label = ttk.Label(stats_frame, text="0")
        self.frames_label.grid(row=0, column=1, padx=10)
        
        ttk.Label(stats_frame, text="Analysis Count:").grid(row=1, column=0, sticky=tk.W)
        self.analysis_label = ttk.Label(stats_frame, text="0")
        self.analysis_label.grid(row=1, column=1, padx=10)
        
        ttk.Label(stats_frame, text="Session Time:").grid(row=2, column=0, sticky=tk.W)
        self.time_label = ttk.Label(stats_frame, text="00:00")
        self.time_label.grid(row=2, column=1, padx=10)
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready to start analysis", foreground="blue")
        self.status_label.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def start_analysis(self):
        """Start video analysis"""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.is_recording = True
            self.start_time = time.time()
            self.session_start_time = time.time()
            self.frame_count = 0
            self.analysis_count = 0
            
            # Reset session data
            self.session_events = []
            self.session_stats = {
                "eye_contact_breaks": [],
                "face_touch_incidents": [],
                "bad_posture_periods": [],
                "flicker_events": []
            }
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Analysis running...", foreground="green")
            
            # Start threads
            self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
            self.update_thread = threading.Thread(target=self.update_ui_loop, daemon=True)
            
            self.video_thread.start()
            self.analysis_thread.start()
            self.update_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start analysis: {str(e)}")
    
    def stop_analysis(self):
        """Stop video analysis"""
        self.is_recording = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Save session log
        self.save_session_log()
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Analysis stopped", foreground="red")
        
        # Clear video display
        self.video_label.config(text="Video Feed")
    
    def video_capture_loop(self):
        """Capture video frames and update video feed display"""
        while self.is_recording and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Add frame to queue (remove old frames if queue is full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
                self.frame_count += 1

                # Show the current frame in the UI (no overlays)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                img = img.resize((640, 360))  # Resize for UI
                imgtk = ImageTk.PhotoImage(image=img)
                def update_label():
                    self.video_label.imgtk = imgtk  # Keep reference
                    self.video_label.config(image=imgtk, text="")
                self.video_label.after(0, update_label)
            else:
                break
    
    def analysis_loop(self):
        """Process frames for analysis"""
        while self.is_recording:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1)
                
                # Analyze frame
                analysis_result = self.analyze_frame(frame)
                
                # Add result to analysis queue
                if self.analysis_queue.full():
                    try:
                        self.analysis_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.analysis_queue.put(analysis_result)
                self.analysis_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {str(e)}")
                continue
    
    def analyze_frame(self, frame):
        """Send frame to ECS API for analysis"""
        try:
            # Encode frame to base64 (use PNG to avoid JPEG encoder issues)
            _, buffer = cv2.imencode('.png', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare request data
            data = {
                'frame_data': frame_base64,
                'session_id': self.session_id,
                'user_id': self.user_id
            }
            
            # Send request to ECS API
            response = requests.post(
                f"{self.api_endpoint}/analyze/frame",
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Frame analysis error: {str(e)}")
            return None
    
    def update_ui_loop(self):
        """Update UI with analysis results"""
        while self.is_recording:
            try:
                # Get latest analysis result
                analysis_result = self.analysis_queue.get(timeout=1)
                
                if analysis_result:
                    # Update current analysis using correct keys from backend response
                    if "eye_contact" in analysis_result:
                        self.current_analysis["eye_contact_score"] = analysis_result["eye_contact"].get("eye_contact_score", 0.0)
                    
                    if "posture" in analysis_result:
                        self.current_analysis["posture_score"] = analysis_result["posture"].get("confidence", 0.0)
                    
                    if "hand_gestures" in analysis_result:
                        self.current_analysis["face_touching"] = analysis_result["hand_gestures"].get("face_touching", 0.0) > 0.3
                    
                    self.current_analysis["overall_score"] = analysis_result.get("overall_score", 0.0)
                    
                    # Log behavioral events (similar to local version)
                    self.log_behavioral_events(analysis_result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"UI update error: {str(e)}")
                continue
    
    def log_behavioral_events(self, analysis_result):
        """Log behavioral events for session analysis"""
        current_time = time.time()
        relative_time = current_time - self.session_start_time
        
        # Eye contact analysis
        if "eye_contact" in analysis_result:
            eye_contact_score = analysis_result["eye_contact"].get("eye_contact_score", 0.0)
            if eye_contact_score < 0.6:  # Threshold for poor eye contact
                event_data = {
                    "timestamp_relative": relative_time,
                    "event_type": "eye_contact_break",
                    "details": {
                        "score": eye_contact_score,
                        "duration": 0.1,  # Approximate frame duration
                        "severity": "medium"
                    }
                }
                self.session_events.append(event_data)
                
                # Also add to session_stats for rating API compatibility
                stats_data = {
                    "start_time_relative": relative_time,
                    "end_time_relative": relative_time + 0.1,
                    "duration": 0.1,
                    "severity": "medium",
                    "score_start": eye_contact_score
                }
                self.session_stats["eye_contact_breaks"].append(stats_data)
        
        # Face touching analysis
        if "hand_gestures" in analysis_result:
            face_touching_score = analysis_result["hand_gestures"].get("face_touching", 0.0)
            if face_touching_score > 0.3:  # Threshold for face touching
                event_data = {
                    "timestamp_relative": relative_time,
                    "event_type": "face_touch",
                    "details": {
                        "score": face_touching_score,
                        "severity": "high" if face_touching_score > 0.7 else "medium"
                    }
                }
                self.session_events.append(event_data)
                
                # Also add to session_stats for rating API compatibility
                stats_data = {
                    "start_time_relative": relative_time,
                    "end_time_relative": relative_time + 0.1,
                    "duration": 0.1,
                    "severity": "high" if face_touching_score > 0.7 else "medium",
                    "max_score": face_touching_score,
                    "total_count": len([e for e in self.session_events if e["event_type"] == "face_touch"]) + 1
                }
                self.session_stats["face_touch_incidents"].append(stats_data)
        
        # Posture analysis
        if "posture" in analysis_result:
            posture_score = analysis_result["posture"].get("confidence", 0.0)
            if posture_score < 0.7:  # Threshold for poor posture
                event_data = {
                    "timestamp_relative": relative_time,
                    "event_type": "bad_posture_start",
                    "details": {
                        "confidence": posture_score,
                        "severity": "medium"
                    }
                }
                self.session_events.append(event_data)
                
                # Also add to session_stats for rating API compatibility
                stats_data = {
                    "start_time_relative": relative_time,
                    "end_time_relative": relative_time + 0.1,
                    "duration": 0.1,
                    "severity": "medium"
                }
                self.session_stats["bad_posture_periods"].append(stats_data)
    
    def save_session_log(self):
        """Save session log to file"""
        if not self.session_start_time:
            return
        
        duration = time.time() - self.session_start_time
        
        # Update session_stats with final values for rating API compatibility
        self.session_stats.update({
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
            "session_start_timestamp": self.session_start_time,
            "end_time": datetime.now().isoformat(),
            "total_duration": duration,
            "total_face_touches": len([e for e in self.session_events if e["event_type"] == "face_touch"]),
            "avg_eye_contact_score": 0,  # Could be calculated from analysis results
            "avg_posture_score": 0       # Could be calculated from analysis results
        })
        
        session_data = {
            "session_stats": self.session_stats,
            "events": self.session_events
        }
        
        # Save to file
        log_file = f"interview_logs/session_{self.session_id}.json"
        try:
            with open(log_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"📁 Session log saved: {log_file}")
            
            # Send to rating service
            self.rate_session(log_file)
            
        except Exception as e:
            print(f"Error saving session log: {str(e)}")
    
    def rate_session(self, log_file):
        """Send session log to rating service"""
        try:
            # Read the log file
            with open(log_file, 'r') as f:
                session_data = json.load(f)
            
            # Send to rating service
            response = requests.post(
                f"{self.rating_endpoint}/rate/session",
                json=session_data,
                timeout=10
            )
            
            if response.status_code == 200:
                rating_result = response.json()
                if rating_result.get("success"):
                    rating = rating_result["rating"]
                    print(f"\n📊 SESSION RATING:")
                    print(f"   Final Score: {rating['final_score']}/100 ({rating['grade']})")
                    print(f"   Eye Contact: {rating['eye_contact_score']}% ({rating['eye_contact_grade']})")
                    print(f"   Face Touch: {rating['face_touch_score']}% ({rating['face_touch_grade']})")
                    print(f"   Posture: {rating['posture_score']}% ({rating['posture_grade']})")
                    print(f"   Events/min: {rating['events_per_minute']}")
                else:
                    print(f"Rating service error: {rating_result.get('error', 'Unknown error')}")
            else:
                print(f"Rating service returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("⚠️  Rating service not available (run Docker container)")
        except Exception as e:
            print(f"Error rating session: {str(e)}")
    
    def update_ui(self):
        """Update UI elements"""
        # Update analysis results
        self.eye_contact_label.config(text=f"{self.current_analysis['eye_contact_score']:.3f}")
        self.posture_label.config(text=f"{self.current_analysis['posture_score']:.3f}")
        self.face_touching_label.config(text="Yes" if self.current_analysis['face_touching'] else "No")
        self.overall_score_label.config(text=f"{self.current_analysis['overall_score']:.3f}")
        
        # Update statistics
        self.frames_label.config(text=str(self.frame_count))
        self.analysis_label.config(text=str(self.analysis_count))
        
        # Update session time
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.time_label.config(text=f"{minutes:02d}:{seconds:02d}")
        
        # Schedule next update
        self.root.after(100, self.update_ui)
    
    def run(self):
        """Start the UI"""
        self.update_ui()
        self.root.mainloop()

def main():
    """Main function"""
    # Load ECS configuration
    try:
        with open('ecs_config.json', 'r') as f:
            config = json.load(f)
        api_endpoint = config['api_endpoint']
        rating_endpoint = config.get('rating_endpoint', 'http://localhost:8001')
    except FileNotFoundError:
        # Use default endpoints if config not found
        api_endpoint = "http://localhost:8000"
        rating_endpoint = "http://localhost:8001"
        print(f"ECS config not found, using default endpoints:")
        print(f"  Analysis API: {api_endpoint}")
        print(f"  Rating API: {rating_endpoint}")
    
    print(f"Starting Interview Analysis UI")
    print(f"Analysis API: {api_endpoint}")
    print(f"Rating API: {rating_endpoint}")
    
    # Test API connections
    try:
        response = requests.get(f"{api_endpoint}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Analysis API connection successful")
        else:
            print(f"⚠️ Analysis API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Analysis API connection failed: {str(e)}")
        print("Make sure your ECS analysis service is running")
    
    try:
        response = requests.get(f"{rating_endpoint}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Rating API connection successful")
        else:
            print(f"⚠️ Rating API returned status {response.status_code}")
    except Exception as e:
        print(f"⚠️ Rating API not available: {str(e)}")
        print("Run 'docker build -t interview-rating . -f Dockerfile.rate-interview' to build rating service")
        print("Run 'docker run --rm -p 8001:8001 -v $(pwd)/interview_logs:/app/logs interview-rating' to start rating service")
    
    # Start UI
    ui = ECSInterviewUI(api_endpoint, rating_endpoint)
    ui.run()

if __name__ == "__main__":
    main() 