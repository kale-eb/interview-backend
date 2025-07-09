# Interview Video Analysis System

A comprehensive behavioral analysis system for interview recordings using computer vision and AI.

## Features

- **Real-time Video Analysis**: Uses MediaPipe for pose detection, eye contact analysis, and hand gesture recognition
- **Behavioral Scoring**: Rates interview performance based on eye contact, posture, and face touching
- **Session Logging**: Saves detailed logs of behavioral events and session statistics
- **Docker Deployment**: Ready for cloud deployment on DigitalOcean
- **Dual UI Options**: 
  - Local analysis (`live_ui.py`) - runs everything locally
  - API-based UI (`live_ui_ecs.py`) - connects to remote APIs

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run local analysis (everything runs on your machine)
python live_ui.py

# Run API-based UI (connects to remote services)
python live_ui_ecs.py
```

### Docker Deployment
```bash
# Build and run services
docker compose up -d --build

# Access APIs
# Analysis API: http://localhost:8000
# Rating API: http://localhost:8001
```

## Configuration

Update `ecs_config.json` with your API endpoints:
```json
{
  "api_endpoint": "http://your-server-ip:8000",
  "rating_endpoint": "http://your-server-ip:8001"
}
```

## File Structure

- `live_ui.py` - Main local analysis UI
- `live_ui_ecs.py` - API-based UI for remote deployment
- `main.py` - FastAPI analysis service
- `main_ecs.py` - ECS-optimized analysis service
- `rate_interview.py` - Interview rating logic
- `rate_interview_api.py` - Rating API service
- `analyze_session.py` - Session analysis tools
- `interview_logs/` - Session log files
- `interview_recordings/` - Video recordings and clips

## API Endpoints

### Analysis API (`main.py`/`main_ecs.py`)
- `POST /analyze/frame` - Analyze a single video frame
- `GET /health` - Health check

### Rating API (`rate_interview_api.py`)
- `POST /rate/session` - Rate a complete session
- `GET /rate/all-sessions` - Rate all sessions in a directory
- `GET /health` - Health check

## Behavioral Metrics

The system tracks and scores:
- **Eye Contact**: Duration and quality of eye contact
- **Posture**: Sitting position and body alignment
- **Face Touching**: Frequency and duration of face touching
- **Overall Score**: Composite score out of 100

## Deployment

### DigitalOcean Droplet
1. Clone repository to droplet
2. Run `docker compose up -d --build`
3. Update `ecs_config.json` with droplet IP
4. Run `python live_ui_ecs.py` locally to connect

### Local Development
1. Install Python dependencies
2. Run `python live_ui.py` for local analysis
3. Or run APIs separately and use `python live_ui_ecs.py` 