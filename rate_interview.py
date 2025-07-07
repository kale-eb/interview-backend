#!/usr/bin/env python3
"""
Interview Rating System

Core functions for rating user body behavior out of 100 from interview log data.
Designed for both local use and AWS Lambda deployment.

Returns clean JSON with:
- final_score: Overall score out of 100
- grade: Overall letter grade (A+ to F)
- eye_contact_score/grade: Eye contact percentage and grade
- face_touch_score/grade: Face touch percentage and grade
- posture_score/grade: Posture percentage and grade
- frequency_penalty: Additional penalty for high event frequency
- events_per_minute: Rate of behavioral issues

Lambda usage:
    from rate_interview import rate_session_data
    result = rate_session_data(session_data)
    
Local usage:
    python rate_interview.py session_file.json
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional


def apply_logarithmic_curve(score: float, max_score: float = 100) -> float:
    """
    Apply logarithmic curve to boost lower scores while keeping high scores similar.
    Examples: 30 -> ~60, 50 -> ~70, 70 -> ~80, 90 -> ~92
    
    Args:
        score: The raw score to curve
        max_score: The maximum possible score for this category
    """
    if score <= 0:
        return 0
    if score >= max_score:
        return max_score
    
    # Logarithmic transformation that compresses the bottom end
    # Uses formula: new_score = max_score * (1 - exp(-k * old_score/max_score))
    # where k is chosen to give desired curve characteristics
    k = 2.3  # Tuned to give 30->60, 90->92 approximately
    
    normalized_score = score / max_score
    curved_score = max_score * (1 - math.exp(-k * normalized_score))
    
    return min(max_score, max(0, curved_score))


def calculate_rating(session_data: Dict) -> Dict:
    """
    Calculate behavior rating from session data.
    
    Args:
        session_data: Loaded JSON session data
        config: "default", "strict", or "lenient"
    
    Returns:
        Dict with rating results
    """
    
    # Configuration
    cfg = {
        "eye_contact_weight": 1.0,
        "posture_weight": 1.0,      
        "face_touch_weight": 1.0,   
        "duration_scaling": 0.4,    
        "min_session_for_scaling": 60.0,  
        "severity_multipliers": {"low": 0.8, "medium": 1.2, "high": 1.6},
        "base_penalty_per_event": 0.0,  # No base penalty for having issues
        "frequency_penalty_factor": 1.5,
        # Max penalties per category (caps)
        "max_eye_contact_penalty": 35.0,
        "max_posture_penalty": 30.0, 
        "max_face_touch_penalty": 35.0,
        "max_flicker_penalty": 15.0,
        "max_frequency_penalty": 20.0,
        "max_base_events_penalty": 15.0
    }
    stats = session_data.get("session_stats", session_data)
    duration = stats.get("total_duration", 0)
    
    # Start with perfect score
    score = 100.0
    
    # Calculate time scaling factor (longer sessions get more forgiveness)
    if duration > cfg["min_session_for_scaling"]:
        time_factor = 1.0 - (cfg["duration_scaling"] * 
                           math.log(duration / cfg["min_session_for_scaling"]))
        time_factor = max(0.3, min(1.0, time_factor))  # Bound between 0.3 and 1.0
    else:
        time_factor = 1.0
    
    # Calculate total event count for frequency penalty
    total_events = len(session_data.get("events", []))
    
    # Frequency penalty: reasonable penalty for many events in short time
    events_per_minute = (total_events / duration) * 60 if duration > 0 else 0
    frequency_penalty = 0
    if events_per_minute > 20:  # More than 20 events per minute is severe
        frequency_penalty = (events_per_minute - 20) * cfg["frequency_penalty_factor"] * 2 + 15
    elif events_per_minute > 10:  # More than 10 events per minute is bad
        frequency_penalty = (events_per_minute - 10) * cfg["frequency_penalty_factor"] * 1.5
    elif events_per_minute > 5:  # More than 5 events per minute is concerning
        frequency_penalty = (events_per_minute - 5) * cfg["frequency_penalty_factor"] * 0.8
    
    # Base penalty just for having behavioral issues - more reasonable
    base_event_penalty = min(total_events * cfg["base_penalty_per_event"], cfg["max_base_events_penalty"])
    
    # Calculate penalties for each behavior type - Reasonable with caps
    penalties = {}
    
    # Eye contact penalties - reasonable scaling with cap
    eye_penalty = 0
    eye_breaks = stats.get("eye_contact_breaks", [])
    for break_event in eye_breaks:
        # Reasonable scaling: 3 points per second + base 8
        duration = break_event.get("duration", 0)
        duration_penalty = duration * 3 + 8  # Base 8 + 3 per second
        severity = break_event.get("severity", "medium")
        severity_mult = cfg["severity_multipliers"].get(severity, 1.0)
        eye_penalty += duration_penalty * severity_mult
    
    # Small penalty for multiple breaks
    if len(eye_breaks) > 1:
        eye_penalty += (len(eye_breaks) - 1) * 5  # 5 points per additional break
    
    # Add flicker penalty to eye contact (since flicker is eye movement instability)
    flicker_count = len(stats.get("flicker_events", []))
    flicker_penalty = min(flicker_count * 3.0, cfg["max_flicker_penalty"] / time_factor)
    eye_penalty += flicker_penalty
    
    # Apply cap and scaling
    eye_penalty = min(eye_penalty, cfg["max_eye_contact_penalty"] / time_factor)
    penalties["eye_contact"] = eye_penalty * time_factor * cfg["eye_contact_weight"]
    
    # Posture penalties - much better duration scaling per your request
    posture_penalty = 0
    posture_periods = stats.get("bad_posture_periods", [])
    for period in posture_periods:
        duration = period.get("duration", 0)
        # Better scaling: 2 seconds = ~1 point, 10 seconds = ~5 points
        # Using square root scaling: more forgiving for short durations
        if duration <= 2:
            duration_penalty = duration * 0.5  # 2 seconds = 1 point
        elif duration <= 10:
            duration_penalty = 1 + (duration - 2) * 0.5  # 10 seconds = 5 points total
        else:
            duration_penalty = 5 + (duration - 10) * 0.3  # Slower increase after 10s
        
        severity = period.get("severity", "medium")
        severity_mult = cfg["severity_multipliers"].get(severity, 1.0)
        posture_penalty += duration_penalty * severity_mult
    
    # Small penalty for multiple periods
    if len(posture_periods) > 1:
        posture_penalty += (len(posture_periods) - 1) * 3
    
    # Apply cap and scaling
    posture_penalty = min(posture_penalty, cfg["max_posture_penalty"] / time_factor)
    penalties["posture"] = posture_penalty * time_factor * cfg["posture_weight"]
    
    # Face touch penalties - strict but reasonable
    face_penalty = 0
    face_incidents = stats.get("face_touch_incidents", [])
    for incident in face_incidents:
        if "duration" in incident:
            # Reasonable: 5 points per second + base 10
            duration = incident.get("duration", 0)
            base_penalty = duration * 5 + 10
        else:
            # Single touch event
            base_penalty = 8
        
        severity = incident.get("severity", "medium")
        severity_mult = cfg["severity_multipliers"].get(severity, 1.0)
        face_penalty += base_penalty * severity_mult
    
    # Penalty for multiple face touches
    if len(face_incidents) > 1:
        face_penalty += (len(face_incidents) - 1) * 6
    
    # Apply cap and scaling
    face_penalty = min(face_penalty, cfg["max_face_touch_penalty"] / time_factor)
    penalties["face_touch"] = face_penalty * time_factor * cfg["face_touch_weight"]
    
    # Add the frequency and base event penalties with caps
    penalties["frequency"] = min(frequency_penalty, cfg["max_frequency_penalty"]) * time_factor
    penalties["base_events"] = base_event_penalty * time_factor
    
    # Define point allocations for each category (totaling 100 points)
    category_allocations = {
        "eye_contact": 35,  # Includes flicker penalties
        "face_touch": 35, 
        "posture": 30
    }
    
    # Calculate individual category scores and apply curves
    category_scores = {}
    curved_category_scores = {}
    
    for category, max_points in category_allocations.items():
        # Calculate raw category score (max_points - penalty, bounded at 0)
        penalty = penalties.get(category, 0)
        raw_category_score = max(0, max_points - penalty)
        
        # Apply logarithmic curve to this category
        curved_score = apply_logarithmic_curve(raw_category_score, max_points)
        
        category_scores[category] = raw_category_score
        curved_category_scores[category] = curved_score
    
    # Sum curved category scores for base final score
    base_final_score = sum(curved_category_scores.values())
    
    # Apply frequency penalty as a final reduction (not curved)
    frequency_penalty = penalties.get("frequency", 0)
    final_score = max(0, base_final_score - frequency_penalty)
    
    # Helper function to determine grade from score
    def get_grade(score):
        if score >= 95: return "A+"
        elif score >= 90: return "A"
        elif score >= 85: return "A-"
        elif score >= 80: return "B+"
        elif score >= 75: return "B"
        elif score >= 70: return "B-"
        elif score >= 65: return "C+"
        elif score >= 60: return "C"
        elif score >= 55: return "C-"
        elif score >= 50: return "D"
        else: return "F"
    
    # Calculate category percentages and grades
    eye_contact_percentage = round((curved_category_scores["eye_contact"] / category_allocations["eye_contact"]) * 100, 1)
    face_touch_percentage = round((curved_category_scores["face_touch"] / category_allocations["face_touch"]) * 100, 1)
    posture_percentage = round((curved_category_scores["posture"] / category_allocations["posture"]) * 100, 1)
    
    return {
        "final_score": round(final_score, 1),
        "grade": get_grade(final_score),
        "eye_contact_score": eye_contact_percentage,
        "eye_contact_grade": get_grade(eye_contact_percentage),
        "face_touch_score": face_touch_percentage,
        "face_touch_grade": get_grade(face_touch_percentage),
        "posture_score": posture_percentage,
        "posture_grade": get_grade(posture_percentage),
        "frequency_penalty": round(penalties["frequency"], 2),
        "events_per_minute": round(events_per_minute, 1)
    }


# =============================================================================
# LAMBDA-READY FUNCTIONS
# =============================================================================

def rate_session_data(session_data: Dict) -> Dict:
    """
    Main function for rating session data. Perfect for Lambda use.
    
    Args:
        session_data: Parsed JSON session data
    
    Returns:
        Dict with rating results
    """
    try:
        return calculate_rating(session_data)
    except Exception as e:
        return {"error": f"Failed to calculate rating: {str(e)}"}


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "session_data": {...}  # The session JSON data
    }
    
    Or for S3-triggered Lambda:
    {
        "Records": [{"s3": {"bucket": {"name": "..."}, "object": {"key": "..."}}}]
    }
    """
    try:
        # Direct session data provided
        if "session_data" in event:
            session_data = event["session_data"]
            return rate_session_data(session_data)
        
        # S3 event (for automatic processing of uploaded files)
        elif "Records" in event and event["Records"]:
            import boto3
            s3 = boto3.client('s3')
            
            results = {}
            for record in event["Records"]:
                bucket = record["s3"]["bucket"]["name"]
                key = record["s3"]["object"]["key"]
                
                # Download and process the file
                response = s3.get_object(Bucket=bucket, Key=key)
                session_data = json.loads(response["Body"].read())
                
                rating = rate_session_data(session_data)
                
                # Optionally save result back to S3
                if event.get("save_result", False):
                    result_key = key.replace(".json", "_rating.json")
                    s3.put_object(
                        Bucket=bucket,
                        Key=result_key,
                        Body=json.dumps(rating, indent=2)
                    )
                
                results[key] = rating
            
            return results if len(results) > 1 else list(results.values())[0]
        
        else:
            return {"error": "Invalid event format. Expected 'session_data' or S3 Records."}
    
    except Exception as e:
        return {"error": f"Lambda execution failed: {str(e)}"}


# =============================================================================
# LOCAL FILE FUNCTIONS
# =============================================================================

def rate_session_file(file_path: str) -> Dict:
    """Rate a single session file (for local use)."""
    try:
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        return rate_session_data(session_data)
    except Exception as e:
        return {"error": f"Failed to process {file_path}: {str(e)}"}


def rate_all_sessions(logs_dir: str = "interview_logs") -> Dict:
    """Rate all session files in the logs directory."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return {"error": f"Directory {logs_dir} does not exist"}
    
    session_files = list(logs_path.glob("session_*.json"))
    if not session_files:
        return {"error": f"No session files found in {logs_dir}"}
    
    results = {}
    for session_file in session_files:
        session_id = session_file.stem.replace("session_", "")
        rating = rate_session_file(str(session_file))
        results[session_id] = rating
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rate_interview.py <session_file.json>")

        sys.exit(1)
    
    result = rate_session_file(sys.argv[1])
    print(json.dumps(result, indent=2)) 