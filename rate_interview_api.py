#!/usr/bin/env python3
"""
FastAPI wrapper for rate_interview.py
Provides HTTP endpoints for session analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
from datetime import datetime
from typing import Dict, List
import logging

from rate_interview import rate_session_data, rate_all_sessions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Interview Rating API",
    description="Rate interview sessions using behavioral analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "interview-rating"
    }

@app.post("/rate/session")
async def rate_session(session_data: Dict):
    """Rate a single session from JSON data"""
    try:
        logger.info("Rating session data")
        result = rate_session_data(session_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "rating": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error rating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rating failed: {str(e)}")

@app.post("/rate/session-file")
async def rate_session_file(file: UploadFile = File(...)):
    """Rate a session from uploaded JSON file"""
    try:
        # Read and parse the uploaded file
        content = await file.read()
        session_data = json.loads(content.decode('utf-8'))
        
        logger.info(f"Rating session file: {file.filename}")
        result = rate_session_data(session_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "filename": file.filename,
            "rating": result,
            "timestamp": datetime.now().isoformat()
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        logger.error(f"Error rating session file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rating failed: {str(e)}")

@app.get("/rate/all-sessions")
async def rate_all_sessions_endpoint(logs_dir: str = "logs"):
    """Rate all session files in the specified directory"""
    try:
        logger.info(f"Rating all sessions in {logs_dir}")
        results = rate_all_sessions(logs_dir)
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return {
            "success": True,
            "total_sessions": len(results),
            "ratings": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error rating all sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rating failed: {str(e)}")

@app.get("/sessions/list")
async def list_sessions(logs_dir: str = "logs"):
    """List all available session files"""
    try:
        import glob
        from pathlib import Path
        
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            return {"sessions": [], "message": f"Directory {logs_dir} does not exist"}
        
        session_files = list(logs_path.glob("session_*.json"))
        sessions = []
        
        for session_file in session_files:
            session_id = session_file.stem.replace("session_", "")
            sessions.append({
                "session_id": session_id,
                "filename": session_file.name,
                "size": session_file.stat().st_size,
                "modified": datetime.fromtimestamp(session_file.stat().st_mtime).isoformat()
            })
        
        return {
            "success": True,
            "sessions": sessions,
            "total_count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Interview Rating API",
        "version": "1.0.0",
        "endpoints": [
            "POST /rate/session - Rate session from JSON data",
            "POST /rate/session-file - Rate session from uploaded file",
            "GET /rate/all-sessions - Rate all sessions in directory",
            "GET /sessions/list - List available sessions",
            "GET /health - Health check"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 