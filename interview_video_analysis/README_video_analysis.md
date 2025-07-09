# Video Analysis Service

A FastAPI service that analyzes complete video files and returns comprehensive session logs with behavioral ratings.

## Features

- **Complete Video Analysis**: Processes entire video files in one request
- **Behavioral Analysis**: Analyzes posture, eye contact, and hand gestures
- **Session Logging**: Generates detailed session logs with timestamps
- **Final Ratings**: Provides overall and component-specific ratings
- **Event Detection**: Identifies behavioral events throughout the session

## API Endpoints

### POST `/analyze_video`
Analyzes a complete video and returns session log with final rating.

**Request Body:**
```json
{
  "session_id": "session_123",
  "user_id": "user_456", 
  "video_data": "base64_encoded_video_data",
  "video_format": "mp4"
}
```

**Response:**
```json
{
  "session_id": "session_123",
  "user_id": "user_456",
  "session_log": {
    "total_frames": 1500,
    "frame_rate": 30.0,
    "duration_seconds": 50.0,
    "posture_data": [...],
    "eye_contact_data": [...],
    "hand_gesture_data": [...],
    "events": [...]
  },
  "final_rating": {
    "overall_rating": 0.85,
    "posture_rating": 0.90,
    "eye_contact_rating": 0.80,
    "hand_gesture_rating": 0.85,
    "total_events": 12,
    "events_per_minute": 14.4,
    "final_recommendations": [...]
  },
  "processing_time": 45.2,
  "total_frames": 1500
}
```

### GET `/health`
Health check endpoint.

## Deployment

### Using Docker

1. **Build the image:**
   ```bash
   docker build -f Dockerfile.video-analysis -t video-analysis-service .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8002:8002 video-analysis-service
   ```

### Using Docker Compose

1. **Start the service:**
   ```bash
   docker-compose -f docker-compose.video-analysis.yml up -d
   ```

2. **Check logs:**
   ```bash
   docker-compose -f docker-compose.video-analysis.yml logs -f
   ```

### Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_video_analysis.txt
   ```

2. **Run the service:**
   ```bash
   python video_analysis_service.py
   ```

## Testing

Run the test script to verify the service:

```bash
python test_video_analysis.py
```

## DigitalOcean Deployment

1. **SSH into your DigitalOcean droplet:**
   ```bash
   ssh root@your-droplet-ip
   ```

2. **Clone your repository:**
   ```bash
   git clone https://github.com/your-username/interviewVideoBackend.git
   cd interviewVideoBackend/interview_video_analysis
   ```

3. **Build and run:**
   ```bash
   docker build -f Dockerfile.video-analysis -t video-analysis-service .
   docker run -d -p 8002:8002 --name video-analysis video-analysis-service
   ```

4. **Set up reverse proxy (nginx):**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location /video-analysis/ {
           proxy_pass http://localhost:8002/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## Usage Example

```python
import requests
import base64

# Read video file
with open("interview_video.mp4", "rb") as f:
    video_data = f.read()

# Encode to base64
video_base64 = base64.b64encode(video_data).decode('utf-8')

# Send to analysis service
response = requests.post(
    "http://localhost:8002/analyze_video",
    json={
        "session_id": "interview_001",
        "user_id": "user_123",
        "video_data": video_base64,
        "video_format": "mp4"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Overall Rating: {result['final_rating']['overall_rating']}")
    print(f"Processing Time: {result['processing_time']:.2f}s")
```

## Architecture

The service processes videos by:

1. **Decoding** the base64 video data
2. **Processing** each frame with MediaPipe Holistic
3. **Analyzing** posture, eye contact, and hand gestures
4. **Aggregating** results into session statistics
5. **Generating** final ratings and recommendations

## Performance

- **Processing Speed**: ~30fps on modern hardware
- **Memory Usage**: ~2GB for typical interview videos
- **Response Time**: Depends on video length (typically 1-3x real-time)

## Monitoring

The service includes health checks and logging:

- Health endpoint: `GET /health`
- Logs: Check container logs with `docker logs video-analysis`
- Metrics: Processing time and frame count in responses 