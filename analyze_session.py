#!/usr/bin/env python3
"""
Interview Session Analysis Tool
Analyzes saved session logs and provides detailed insights
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

class SessionAnalyzer:
    def __init__(self, logs_dir="interview_logs", recordings_dir="interview_recordings"):
        self.logs_dir = logs_dir
        self.recordings_dir = recordings_dir
        
    def list_sessions(self) -> List[str]:
        """List all available sessions"""
        session_files = glob.glob(os.path.join(self.logs_dir, "session_*.json"))
        sessions = []
        for file_path in session_files:
            session_id = os.path.basename(file_path).replace("session_", "").replace(".json", "")
            sessions.append(session_id)
        return sorted(sessions)
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Load session data from log file"""
        log_file = os.path.join(self.logs_dir, f"session_{session_id}.json")
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Session log not found: {log_file}")
        
        with open(log_file, 'r') as f:
            return json.load(f)
    
    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """Analyze a specific session"""
        session_data = self.load_session(session_id)
        
        # Extract key metrics
        duration = session_data.get('duration', 0)
        events = session_data.get('events', [])
        
        # Count different types of events
        event_counts = {}
        for event in events:
            event_type = event.get('event_type', 'unknown')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculate behavioral scores
        eye_contact_breaks = len([e for e in events if e.get('event_type') == 'eye_contact_break'])
        face_touches = len([e for e in events if e.get('event_type') == 'face_touch'])
        bad_posture_periods = len([e for e in events if e.get('event_type') == 'bad_posture_start'])
        
        # Calculate total time in problematic behaviors
        problematic_time = 0
        for event in events:
            if event.get('event_type') in ['eye_contact_break', 'bad_posture_start']:
                details = event.get('details', {})
                problematic_time += details.get('duration', 0)
        
        # Calculate percentage of time with issues
        issue_percentage = (problematic_time / duration * 100) if duration > 0 else 0
        
        # Generate recommendations
        recommendations = []
        if eye_contact_breaks > 3:
            recommendations.append("Work on maintaining consistent eye contact")
        if face_touches > 2:
            recommendations.append("Practice keeping hands away from face")
        if bad_posture_periods > 2:
            recommendations.append("Focus on maintaining good posture")
        if issue_percentage > 20:
            recommendations.append("Overall: Practice maintaining professional body language")
        if not recommendations:
            recommendations.append("Excellent performance! Keep up the good work")
        
        return {
            'session_id': session_id,
            'duration': duration,
            'eye_contact_breaks': eye_contact_breaks,
            'face_touches': face_touches,
            'bad_posture_periods': bad_posture_periods,
            'problematic_time': problematic_time,
            'issue_percentage': issue_percentage,
            'event_counts': event_counts,
            'recommendations': recommendations,
            'raw_data': session_data
        }
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """Print formatted analysis results"""
        print(f"\n{'='*60}")
        print(f"📊 SESSION ANALYSIS: {analysis['session_id']}")
        print(f"{'='*60}")
        
        # Basic stats
        print(f"⏱️  Duration: {analysis['duration']:.1f} seconds ({analysis['duration']/60:.1f} minutes)")
        print(f"👁️  Eye contact breaks: {analysis['eye_contact_breaks']}")
        print(f"🤏 Face touches: {analysis['face_touches']}")
        print(f"🧍 Bad posture periods: {analysis['bad_posture_periods']}")
        print(f"⚠️  Time with issues: {analysis['problematic_time']:.1f}s ({analysis['issue_percentage']:.1f}%)")
        
        # Event breakdown
        if analysis['event_counts']:
            print(f"\n📈 Event Breakdown:")
            for event_type, count in analysis['event_counts'].items():
                print(f"   {event_type}: {count}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        # Video files
        recording_dir = os.path.join(self.recordings_dir, analysis['session_id'])
        if os.path.exists(recording_dir):
            print(f"\n📹 Video Files:")
            full_video = os.path.join(recording_dir, "full_session.mp4")
            if os.path.exists(full_video):
                print(f"   • Full session: {full_video}")
            
            clips_dir = os.path.join(recording_dir, "clips")
            if os.path.exists(clips_dir):
                clips = glob.glob(os.path.join(clips_dir, "*.mp4"))
                if clips:
                    print(f"   • Event clips: {len(clips)} clips available")
                    for clip in clips[:3]:  # Show first 3 clips
                        print(f"     - {os.path.basename(clip)}")
                    if len(clips) > 3:
                        print(f"     ... and {len(clips) - 3} more")
        
        print(f"{'='*60}\n")
    
    def compare_sessions(self, session_ids: List[str]):
        """Compare multiple sessions"""
        if len(session_ids) < 2:
            print("Need at least 2 sessions to compare")
            return
        
        analyses = []
        for session_id in session_ids:
            try:
                analysis = self.analyze_session(session_id)
                analyses.append(analysis)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        if len(analyses) < 2:
            print("Not enough valid sessions to compare")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 SESSION COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison table
        print(f"{'Session':<15} {'Duration':<10} {'Eye Breaks':<12} {'Face Touches':<13} {'Posture':<10} {'Issues %':<10}")
        print("-" * 80)
        
        for analysis in analyses:
            print(f"{analysis['session_id']:<15} "
                  f"{analysis['duration']:<10.1f} "
                  f"{analysis['eye_contact_breaks']:<12} "
                  f"{analysis['face_touches']:<13} "
                  f"{analysis['bad_posture_periods']:<10} "
                  f"{analysis['issue_percentage']:<10.1f}")
        
        # Find best and worst sessions
        best_session = min(analyses, key=lambda x: x['issue_percentage'])
        worst_session = max(analyses, key=lambda x: x['issue_percentage'])
        
        print(f"\n🏆 Best Session: {best_session['session_id']} ({best_session['issue_percentage']:.1f}% issues)")
        print(f"📉 Needs Improvement: {worst_session['session_id']} ({worst_session['issue_percentage']:.1f}% issues)")
        
        print(f"{'='*80}\n")

def main():
    """Main function for session analysis"""
    analyzer = SessionAnalyzer()
    
    # List available sessions
    sessions = analyzer.list_sessions()
    
    if not sessions:
        print("No session logs found. Run some interview sessions first!")
        return
    
    print(f"📁 Available Sessions ({len(sessions)}):")
    for i, session_id in enumerate(sessions, 1):
        print(f"  {i}. {session_id}")
    
    # Interactive menu
    while True:
        print(f"\n🔍 Session Analysis Options:")
        print(f"  1. Analyze specific session")
        print(f"  2. Compare multiple sessions")
        print(f"  3. List all sessions")
        print(f"  4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            session_num = input(f"Enter session number (1-{len(sessions)}): ").strip()
            try:
                session_idx = int(session_num) - 1
                if 0 <= session_idx < len(sessions):
                    session_id = sessions[session_idx]
                    analysis = analyzer.analyze_session(session_id)
                    analyzer.print_analysis(analysis)
                else:
                    print("Invalid session number")
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == '2':
            print("Enter session numbers to compare (e.g., 1,3,5):")
            session_nums = input().strip().split(',')
            try:
                session_ids = []
                for num in session_nums:
                    session_idx = int(num.strip()) - 1
                    if 0 <= session_idx < len(sessions):
                        session_ids.append(sessions[session_idx])
                    else:
                        print(f"Invalid session number: {num}")
                        break
                else:
                    if len(session_ids) >= 2:
                        analyzer.compare_sessions(session_ids)
                    else:
                        print("Need at least 2 valid sessions to compare")
            except ValueError:
                print("Please enter valid numbers separated by commas")
        
        elif choice == '3':
            print(f"\n📁 Available Sessions ({len(sessions)}):")
            for i, session_id in enumerate(sessions, 1):
                print(f"  {i}. {session_id}")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 