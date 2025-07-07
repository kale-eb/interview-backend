"""
InterviewMaster Backend API
A comprehensive FastAPI application for streak-based behavioral interview training.

Features:
- User authentication via Supabase
- Daily prompt system with streak tracking
- Live AI interviews with real-time feedback
- Video/audio analysis via Lambda
- Progress tracking and gamification
- Real-time WebSocket connections
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Supabase and database
from supabase import create_client, Client
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# AI and ML services
import openai
import boto3
from elevenlabs import generate, set_api_key

# Video processing (existing functions)
from video_processing import (
    PostureAnalyzer, EyeContactAnalyzer, HandGestureAnalyzer,
    process_frame, analyze_audio_segment
)

# Configuration
from config import settings

# Initialize services
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
openai.api_key = settings.OPENAI_API_KEY
set_api_key(settings.ELEVENLABS_API_KEY)

# AWS services
s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
lambda_client = boto3.client('lambda', region_name=settings.AWS_REGION)

# Database setup
engine = create_async_engine(settings.DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Security
security = HTTPBearer()

# WebSocket manager for real-time feedback
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
    
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_sessions[user_id] = session_id
        
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
    
    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except:
                self.disconnect(user_id)

manager = ConnectionManager()

# Pydantic models
class UserProfile(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    current_streak: int = 0
    longest_streak: int = 0
    total_currency: int = 0
    streak_freeze_count: int = 0
    subscription_tier: str = "free"
    onboarding_completed: bool = False

class DailyPromptResponse(BaseModel):
    prompt_id: str
    response_text: str
    time_spent_minutes: Optional[int] = None

class SessionCreate(BaseModel):
    session_type: str = Field(..., regex="^(live_ai|practice_text|mock_interview)$")
    session_mode: str = Field(..., regex="^(real_time|recorded|hybrid)$")
    job_description_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

class RealTimeAnalysis(BaseModel):
    session_id: str
    timestamp_seconds: float
    analysis_type: str
    eye_contact_score: Optional[float] = None
    posture_score: Optional[float] = None
    hand_gesture_score: Optional[float] = None
    filler_words_count: Optional[int] = None
    speech_pace_wpm: Optional[int] = None
    confidence_score: Optional[float] = None
    feedback_message: Optional[str] = None

class FrameAnalysisRequest(BaseModel):
    session_id: str
    frame_data: str  # base64 encoded image
    timestamp: float
    analysis_type: str = "video"

class AudioAnalysisRequest(BaseModel):
    session_id: str
    audio_data: str  # base64 encoded audio
    timestamp: float
    duration: float

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Verify JWT token with Supabase
        user = supabase.auth.get_user(credentials.credentials)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return user.user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# Database dependency
async def get_db():
    async with async_session() as session:
        yield session

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting InterviewMaster Backend...")
    
    # Initialize background tasks
    asyncio.create_task(daily_streak_checker())
    asyncio.create_task(achievement_processor())
    
    yield
    
    # Shutdown
    print("🔄 Shutting down InterviewMaster Backend...")

# Initialize FastAPI app
app = FastAPI(
    title="InterviewMaster API",
    description="Streak-based behavioral interview training platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUTHENTICATION & USER MANAGEMENT
# ============================================================================

@app.post("/auth/register")
async def register_user(email: str, password: str, full_name: Optional[str] = None):
    """Register a new user with Supabase auth"""
    try:
        result = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {"full_name": full_name}
            }
        })
        return {"message": "User registered successfully", "user": result.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login_user(email: str, password: str):
    """Login user and return JWT token"""
    try:
        result = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return {
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
            "user": result.user
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/auth/profile", response_model=UserProfile)
async def get_user_profile(current_user=Depends(get_current_user), db=Depends(get_db)):
    """Get current user's profile with streak and currency info"""
    try:
        result = await db.execute(
            text("SELECT * FROM user_profiles WHERE id = :user_id"),
            {"user_id": current_user.id}
        )
        profile = result.fetchone()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return UserProfile(**dict(profile))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DAILY PROMPTS & STREAK SYSTEM
# ============================================================================

@app.get("/daily-prompt")
async def get_daily_prompt(current_user=Depends(get_current_user), db=Depends(get_db)):
    """Get today's daily prompt for the user"""
    try:
        # Check if user already answered today
        today = datetime.now().date()
        result = await db.execute(
            text("""
                SELECT COUNT(*) as count FROM daily_prompt_responses 
                WHERE user_id = :user_id AND response_date = :today
            """),
            {"user_id": current_user.id, "today": today}
        )
        
        if result.fetchone().count > 0:
            return {"message": "Already answered today's prompt", "completed": True}
        
        # Get a random prompt the user hasn't answered recently
        result = await db.execute(
            text("""
                SELECT * FROM daily_prompt_templates 
                WHERE is_active = true 
                AND id NOT IN (
                    SELECT prompt_id FROM daily_prompt_responses 
                    WHERE user_id = :user_id 
                    AND response_date > :cutoff_date
                )
                ORDER BY RANDOM() LIMIT 1
            """),
            {
                "user_id": current_user.id,
                "cutoff_date": today - timedelta(days=30)  # Don't repeat prompts from last 30 days
            }
        )
        
        prompt = result.fetchone()
        if not prompt:
            raise HTTPException(status_code=404, detail="No available prompts")
        
        return {
            "prompt": dict(prompt),
            "completed": False,
            "streak_reward": 10 + (await get_current_streak(current_user.id, db) * 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/daily-prompt/respond")
async def respond_to_daily_prompt(
    response: DailyPromptResponse,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Submit response to daily prompt and update streak"""
    try:
        # Check if already answered today
        today = datetime.now().date()
        result = await db.execute(
            text("""
                SELECT COUNT(*) as count FROM daily_prompt_responses 
                WHERE user_id = :user_id AND response_date = :today
            """),
            {"user_id": current_user.id, "today": today}
        )
        
        if result.fetchone().count > 0:
            raise HTTPException(status_code=400, detail="Already answered today")
        
        # Analyze response with AI
        ai_analysis = await analyze_story_with_ai(response.response_text)
        
        # Save response
        await db.execute(
            text("""
                INSERT INTO daily_prompt_responses 
                (user_id, prompt_id, response_text, response_date, time_spent_minutes, 
                 ai_analysis, story_quality_score, extracted_keywords, behavioral_category)
                VALUES (:user_id, :prompt_id, :response_text, :today, :time_spent, 
                        :ai_analysis, :quality_score, :keywords, :category)
            """),
            {
                "user_id": current_user.id,
                "prompt_id": response.prompt_id,
                "response_text": response.response_text,
                "today": today,
                "time_spent": response.time_spent_minutes,
                "ai_analysis": ai_analysis,
                "quality_score": ai_analysis.get("quality_score", 50),
                "keywords": ai_analysis.get("keywords", []),
                "category": ai_analysis.get("category", "general")
            }
        )
        
        # Update streak and currency
        await update_user_streak(current_user.id, db)
        
        # Update behavioral profile
        await update_behavioral_profile(current_user.id, response.response_text, ai_analysis, db)
        
        await db.commit()
        
        return {
            "message": "Response saved successfully",
            "ai_analysis": ai_analysis,
            "streak_updated": True
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# INTERVIEW SESSIONS
# ============================================================================

@app.post("/sessions/create")
async def create_interview_session(
    session_data: SessionCreate,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Create a new interview session"""
    try:
        session_id = f"usr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_data.session_type}"
        
        # Insert session
        result = await db.execute(
            text("""
                INSERT INTO interview_sessions 
                (user_id, job_description_id, session_type, session_mode, session_id, title, description)
                VALUES (:user_id, :job_desc_id, :session_type, :session_mode, :session_id, :title, :description)
                RETURNING id
            """),
            {
                "user_id": current_user.id,
                "job_desc_id": session_data.job_description_id,
                "session_type": session_data.session_type,
                "session_mode": session_data.session_mode,
                "session_id": session_id,
                "title": session_data.title,
                "description": session_data.description
            }
        )
        
        session_uuid = result.fetchone()[0]
        await db.commit()
        
        return {
            "session_id": session_uuid,
            "session_identifier": session_id,
            "websocket_url": f"/ws/{current_user.id}/{session_uuid}"
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    """WebSocket endpoint for real-time feedback during interviews"""
    await manager.connect(websocket, user_id, session_id)
    
    try:
        while True:
            # Receive frame or audio data
            data = await websocket.receive_json()
            
            if data["type"] == "frame":
                await process_video_frame(data, user_id, session_id)
            elif data["type"] == "audio":
                await process_audio_segment(data, user_id, session_id)
            elif data["type"] == "session_end":
                await end_session(user_id, session_id)
                break
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)
        await end_session(user_id, session_id)

@app.post("/sessions/{session_id}/analyze-frame")
async def analyze_video_frame(
    session_id: str,
    frame_request: FrameAnalysisRequest,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Analyze a video frame for behavioral cues"""
    try:
        # Decode frame and analyze
        analysis_result = await process_frame_analysis(
            frame_request.frame_data,
            frame_request.timestamp,
            session_id
        )
        
        # Store real-time analysis
        await db.execute(
            text("""
                INSERT INTO real_time_analysis 
                (session_id, timestamp_seconds, analysis_type, eye_contact_score, 
                 posture_score, hand_gesture_score, overall_confidence, feedback_message)
                VALUES (:session_id, :timestamp, :analysis_type, :eye_contact, 
                        :posture, :hand_gesture, :confidence, :feedback)
            """),
            {
                "session_id": session_id,
                "timestamp": frame_request.timestamp,
                "analysis_type": frame_request.analysis_type,
                "eye_contact": analysis_result.get("eye_contact_score"),
                "posture": analysis_result.get("posture_score"),
                "hand_gesture": analysis_result.get("hand_gesture_score"),
                "confidence": analysis_result.get("overall_confidence"),
                "feedback": analysis_result.get("feedback_message")
            }
        )
        
        await db.commit()
        
        # Send real-time feedback if needed
        if analysis_result.get("feedback_message"):
            await manager.send_personal_message(
                {
                    "type": "feedback",
                    "message": analysis_result["feedback_message"],
                    "category": analysis_result.get("feedback_category", "general"),
                    "timestamp": frame_request.timestamp
                },
                current_user.id
            )
        
        return analysis_result
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PROGRESS TRACKING & GAMIFICATION
# ============================================================================

@app.get("/progress")
async def get_user_progress(current_user=Depends(get_current_user), db=Depends(get_db)):
    """Get user's progress across all skill categories"""
    try:
        result = await db.execute(
            text("""
                SELECT skill_category, current_level, current_score, target_score,
                       scores_history, milestones_achieved, last_updated
                FROM user_progress 
                WHERE user_id = :user_id
                ORDER BY skill_category
            """),
            {"user_id": current_user.id}
        )
        
        progress_data = [dict(row) for row in result.fetchall()]
        
        return {
            "progress": progress_data,
            "overall_level": calculate_overall_level(progress_data),
            "next_milestone": get_next_milestone(progress_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/achievements")
async def get_user_achievements(current_user=Depends(get_current_user), db=Depends(get_db)):
    """Get user's earned achievements and available ones"""
    try:
        # Get earned achievements
        earned_result = await db.execute(
            text("""
                SELECT a.*, ua.earned_at, ua.progress_data
                FROM achievements a
                JOIN user_achievements ua ON a.id = ua.achievement_id
                WHERE ua.user_id = :user_id
                ORDER BY ua.earned_at DESC
            """),
            {"user_id": current_user.id}
        )
        
        # Get available achievements
        available_result = await db.execute(
            text("""
                SELECT a.* FROM achievements a
                WHERE a.is_active = true
                AND a.id NOT IN (
                    SELECT achievement_id FROM user_achievements WHERE user_id = :user_id
                )
                ORDER BY a.rarity, a.reward_currency DESC
            """),
            {"user_id": current_user.id}
        )
        
        return {
            "earned_achievements": [dict(row) for row in earned_result.fetchall()],
            "available_achievements": [dict(row) for row in available_result.fetchall()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
async def get_leaderboard(category: str = "streak", limit: int = 50, db=Depends(get_db)):
    """Get leaderboard for different categories"""
    try:
        if category == "streak":
            result = await db.execute(
                text("""
                    SELECT full_name, current_streak, longest_streak, total_currency,
                           RANK() OVER (ORDER BY current_streak DESC) as rank
                    FROM user_profiles 
                    WHERE is_active = true
                    ORDER BY current_streak DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
        elif category == "currency":
            result = await db.execute(
                text("""
                    SELECT full_name, current_streak, longest_streak, total_currency,
                           RANK() OVER (ORDER BY total_currency DESC) as rank
                    FROM user_profiles 
                    WHERE is_active = true
                    ORDER BY total_currency DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid category")
        
        return {"leaderboard": [dict(row) for row in result.fetchall()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STORE & CUSTOMIZATION
# ============================================================================

@app.get("/store")
async def get_store_items(current_user=Depends(get_current_user), db=Depends(get_db)):
    """Get available customization items in the store"""
    try:
        result = await db.execute(
            text("""
                SELECT ci.*, 
                       CASE WHEN uc.user_id IS NOT NULL THEN true ELSE false END as owned
                FROM customization_items ci
                LEFT JOIN user_customizations uc ON ci.id = uc.item_id AND uc.user_id = :user_id
                WHERE ci.is_available = true
                ORDER BY ci.rarity, ci.cost_currency
            """),
            {"user_id": current_user.id}
        )
        
        return {"items": [dict(row) for row in result.fetchall()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store/purchase/{item_id}")
async def purchase_item(
    item_id: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db)
):
    """Purchase a customization item"""
    try:
        # Check if item exists and user has enough currency
        result = await db.execute(
            text("""
                SELECT ci.cost_currency, up.total_currency
                FROM customization_items ci, user_profiles up
                WHERE ci.id = :item_id AND up.id = :user_id
                AND ci.is_available = true
            """),
            {"item_id": item_id, "user_id": current_user.id}
        )
        
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        
        cost, user_currency = row
        if user_currency < cost:
            raise HTTPException(status_code=400, detail="Insufficient currency")
        
        # Check if already owned
        result = await db.execute(
            text("""
                SELECT COUNT(*) as count FROM user_customizations 
                WHERE user_id = :user_id AND item_id = :item_id
            """),
            {"user_id": current_user.id, "item_id": item_id}
        )
        
        if result.fetchone().count > 0:
            raise HTTPException(status_code=400, detail="Item already owned")
        
        # Purchase item
        await db.execute(
            text("""
                INSERT INTO user_customizations (user_id, item_id, purchase_price)
                VALUES (:user_id, :item_id, :cost)
            """),
            {"user_id": current_user.id, "item_id": item_id, "cost": cost}
        )
        
        # Deduct currency
        await db.execute(
            text("""
                UPDATE user_profiles 
                SET total_currency = total_currency - :cost
                WHERE id = :user_id
            """),
            {"cost": cost, "user_id": current_user.id}
        )
        
        # Record transaction
        await db.execute(
            text("""
                INSERT INTO currency_transactions (user_id, amount, transaction_type, description, reference_id)
                VALUES (:user_id, :amount, 'purchase', 'Purchased customization item', :item_id)
            """),
            {"user_id": current_user.id, "amount": -cost, "item_id": item_id}
        )
        
        await db.commit()
        
        return {"message": "Item purchased successfully", "remaining_currency": user_currency - cost}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def analyze_story_with_ai(story_text: str) -> Dict[str, Any]:
    """Analyze user's story using OpenAI to extract behavioral insights"""
    try:
        prompt = f"""
        Analyze this behavioral story and provide insights:
        
        Story: {story_text}
        
        Please analyze and return JSON with:
        - quality_score (0-100): How well-developed is this story?
        - keywords: List of key skills/traits demonstrated
        - category: Primary behavioral category (leadership, teamwork, etc.)
        - strengths: What the person did well
        - improvements: How they could improve their storytelling
        - star_rating: How well does this follow the STAR method? (1-5)
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        # Parse AI response
        content = response.choices[0].message.content
        # Add JSON parsing logic here
        
        return {
            "quality_score": 75,  # Placeholder
            "keywords": ["leadership", "problem-solving"],
            "category": "leadership",
            "strengths": ["Clear situation description"],
            "improvements": ["Add more specific results"],
            "star_rating": 4
        }
    except Exception as e:
        print(f"AI analysis error: {e}")
        return {"quality_score": 50, "keywords": [], "category": "general"}

async def get_current_streak(user_id: str, db) -> int:
    """Get user's current streak"""
    result = await db.execute(
        text("SELECT current_streak FROM user_profiles WHERE id = :user_id"),
        {"user_id": user_id}
    )
    return result.fetchone()[0] if result.fetchone() else 0

async def update_user_streak(user_id: str, db):
    """Update user's streak and award currency"""
    await db.execute(
        text("SELECT update_user_streak(:user_id)"),
        {"user_id": user_id}
    )

async def update_behavioral_profile(user_id: str, story_text: str, ai_analysis: Dict, db):
    """Update user's behavioral profile with new story"""
    # Add logic to categorize and store the story in behavioral profile
    pass

async def process_frame_analysis(frame_data: str, timestamp: float, session_id: str):
    """Process video frame for behavioral analysis"""
    # Integrate with existing video processing functions
    # This would use your existing PostureAnalyzer, EyeContactAnalyzer, etc.
    return {
        "eye_contact_score": 0.85,
        "posture_score": 0.92,
        "hand_gesture_score": 0.78,
        "overall_confidence": 0.83,
        "feedback_message": "Great eye contact! Try to sit up a bit straighter."
    }

async def daily_streak_checker():
    """Background task to check and reset streaks"""
    while True:
        # Check for users who missed their daily prompt
        # Reset streaks as needed
        await asyncio.sleep(3600)  # Check every hour

async def achievement_processor():
    """Background task to check for new achievements"""
    while True:
        # Check user progress against achievement requirements
        # Award achievements and currency
        await asyncio.sleep(1800)  # Check every 30 minutes

def calculate_overall_level(progress_data: List[Dict]) -> int:
    """Calculate overall user level from progress data"""
    if not progress_data:
        return 1
    
    total_score = sum(p["current_score"] for p in progress_data)
    avg_score = total_score / len(progress_data)
    return max(1, int(avg_score / 10))

def get_next_milestone(progress_data: List[Dict]) -> Dict:
    """Get the next milestone user can achieve"""
    # Logic to determine next achievable milestone
    return {
        "category": "eye_contact",
        "target_score": 80,
        "current_score": 65,
        "reward": "50 currency + Eye Contact Badge"
    }

async def end_session(user_id: str, session_id: str):
    """End interview session and trigger Lambda analysis"""
    try:
        # Update session status
        async with async_session() as db:
            await db.execute(
                text("""
                    UPDATE interview_sessions 
                    SET status = 'completed', ended_at = NOW()
                    WHERE id = :session_id
                """),
                {"session_id": session_id}
            )
            await db.commit()
        
        # Trigger Lambda for comprehensive analysis
        lambda_client.invoke(
            FunctionName='interview-analysis-processor',
            InvocationType='Event',  # Async
            Payload=json.dumps({
                "session_id": session_id,
                "user_id": user_id,
                "analysis_type": "comprehensive"
            })
        )
        
        # Award session completion currency
        await award_session_completion(user_id, session_id)
        
    except Exception as e:
        print(f"Error ending session: {e}")

async def award_session_completion(user_id: str, session_id: str):
    """Award currency for completing a session"""
    try:
        async with async_session() as db:
            # Award currency
            await db.execute(
                text("""
                    UPDATE user_profiles 
                    SET total_currency = total_currency + 50
                    WHERE id = :user_id
                """),
                {"user_id": user_id}
            )
            
            # Record transaction
            await db.execute(
                text("""
                    INSERT INTO currency_transactions (user_id, amount, transaction_type, description, reference_id)
                    VALUES (:user_id, 50, 'session_completion', 'Completed interview session', :session_id)
                """),
                {"user_id": user_id, "session_id": session_id}
            )
            
            await db.commit()
    except Exception as e:
        print(f"Error awarding session completion: {e}")

# ============================================================================
# HEALTH CHECK & STARTUP
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "InterviewMaster API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 