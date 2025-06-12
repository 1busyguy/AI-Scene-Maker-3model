from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generator
import os
import tempfile
import json
import asyncio
import uuid
from datetime import datetime
import logging

# Import existing functionality (surgical precision - no modifications)
from ui.gradio_ui import start_chain_generation_with_updates
from utils import openai_client, fal_client, video_processing
from utils.langchain_prompts import generate_cinematic_prompt
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Scene Maker API",
    description="Professional API for AI-powered video generation and scene creation",
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

# Global storage for generation sessions
generation_sessions: Dict[str, Dict] = {}

# Pydantic Models
class ImageAnalysisResponse(BaseModel):
    theme: str
    background: str
    main_subject: str
    tone_and_color: str
    action_direction: str

class VideoGenerationRequest(BaseModel):
    action_direction: str
    theme: Optional[str] = None
    background: Optional[str] = None
    main_subject: Optional[str] = None
    tone_and_color: Optional[str] = None
    scene_vision: Optional[str] = None
    resolution: str = Field(default="720p", description="Video resolution")
    inference_steps: int = Field(default=40, ge=1, le=40)
    safety_checker: bool = False
    prompt_expansion: bool = True
    num_chains: int = Field(default=3, ge=1, le=10)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    model_type: str = Field(default="WAN (Default)", description="AI model to use")
    
    # Model-specific parameters
    pixverse_duration: str = Field(default="5", description="Duration for Pixverse (5 or 8)")
    pixverse_style: str = Field(default="None", description="Style for Pixverse")
    pixverse_negative_prompt: str = Field(default="", description="Negative prompt for Pixverse")
    luma_duration: str = Field(default="5", description="Duration for LUMA")
    luma_aspect_ratio: str = Field(default="16:9", description="Aspect ratio for LUMA")
    kling_duration: str = Field(default="5", description="Duration for Kling (5 or 10)")
    kling_aspect_ratio: str = Field(default="16:9", description="Aspect ratio for Kling")
    kling_negative_prompt: str = Field(default="", description="Negative prompt for Kling")
    kling_creativity: float = Field(default=0.5, ge=0.0, le=1.0, description="Creativity for Kling")
    
    # Enhancement options
    enable_character_consistency: bool = True
    enable_face_enhancement: bool = True
    enable_face_swapping: bool = False
    enable_quality_preservation: bool = True
    quality_vs_speed: str = Field(default="Maximum Quality", description="Quality vs Speed preference")

class VideoGenerationResponse(BaseModel):
    session_id: str
    status: str
    message: str

class GenerationStatus(BaseModel):
    session_id: str
    status: str
    progress: float
    current_operation: str
    completed_chains: int
    total_chains: int
    video_paths: List[str]
    final_video_path: Optional[str]
    error_message: Optional[str]

class ConfigResponse(BaseModel):
    output_dir: str
    default_resolution: str
    default_inference_steps: int
    api_keys_configured: bool
    available_models: List[str]

# Helper Functions
def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    file_extension = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        content = upload_file.file.read()
        tmp_file.write(content)
        return tmp_file.name

def create_model_params(request: VideoGenerationRequest) -> Dict[str, Any]:
    """Create model-specific parameters from request"""
    model_params = {}
    
    if request.model_type == "Pixverse v3.5":
        model_params.update({
            "duration": request.pixverse_duration,
            "style": request.pixverse_style if request.pixverse_style != "None" else None,
            "negative_prompt": request.pixverse_negative_prompt
        })
    elif request.model_type == "LUMA Ray2":
        model_params.update({
            "duration": request.luma_duration,
            "aspect_ratio": request.luma_aspect_ratio
        })
    elif request.model_type == "Kling 2.1 PRO":
        model_params.update({
            "duration": request.kling_duration,
            "aspect_ratio": request.kling_aspect_ratio,
            "negative_prompt": request.kling_negative_prompt,
            "creativity": request.kling_creativity
        })
    
    return model_params

async def run_generation_async(session_id: str, image_path: str, request: VideoGenerationRequest):
    """Run video generation in background and update session"""
    try:
        generation_sessions[session_id]["status"] = "running"
        generation_sessions[session_id]["start_time"] = datetime.now().isoformat()
        
        model_params = create_model_params(request)
        cancel_requested = lambda: generation_sessions[session_id].get("cancelled", False)
        
        # Run the existing generation function
        for update_type, data, message in start_chain_generation_with_updates(
            action_direction=request.action_direction,
            image=image_path,
            theme=request.theme,
            background=request.background,
            main_subject=request.main_subject,
            tone_and_color=request.tone_and_color,
            scene_vision=request.scene_vision,
            resolution=request.resolution,
            inference_steps=request.inference_steps,
            safety_checker=request.safety_checker,
            prompt_expansion=request.prompt_expansion,
            num_chains=request.num_chains,
            seed=request.seed,
            model_type=request.model_type,
            model_params=model_params,
            cancel_requested=cancel_requested,
            enable_character_consistency=request.enable_character_consistency,
            enable_face_enhancement=request.enable_face_enhancement,
            enable_face_swapping=request.enable_face_swapping,
            enable_quality_preservation=request.enable_quality_preservation,
            quality_vs_speed=request.quality_vs_speed
        ):
            # Update session with progress
            session = generation_sessions[session_id]
            session["last_update"] = datetime.now().isoformat()
            session["last_message"] = message
            
            if update_type == "progress":
                session["progress"] = data
                session["current_operation"] = message
            elif update_type == "analysis":
                session["image_analysis"] = data
            elif update_type == "vision":
                session["scene_vision"] = data
            elif update_type == "chains":
                session["total_chains"] = data
            elif update_type == "chain_complete":
                session["video_paths"] = session.get("video_paths", [])
                session["video_paths"].append(data)
                session["completed_chains"] = len(session["video_paths"])
            elif update_type == "final":
                session["final_video_path"] = data
                session["status"] = "completed"
                session["progress"] = 100.0
                session["end_time"] = datetime.now().isoformat()
            elif update_type == "error":
                session["status"] = "error"
                session["error_message"] = message
                session["end_time"] = datetime.now().isoformat()
                break
            elif update_type == "cancelled":
                session["status"] = "cancelled"
                session["end_time"] = datetime.now().isoformat()
                break
                
    except Exception as e:
        logger.exception(f"Error in generation session {session_id}")
        generation_sessions[session_id]["status"] = "error"
        generation_sessions[session_id]["error_message"] = str(e)
        generation_sessions[session_id]["end_time"] = datetime.now().isoformat()

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Scene Maker API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    return ConfigResponse(
        output_dir=config.OUTPUT_DIR,
        default_resolution=config.DEFAULT_RESOLUTION,
        default_inference_steps=config.DEFAULT_INFERENCE_STEPS,
        api_keys_configured=bool(config.FAL_API_KEY and config.OPENAI_API_KEY),
        available_models=["WAN (Default)", "Pixverse v3.5", "LUMA Ray2", "Kling 2.1 PRO"]
    )

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image and extract structured information"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file
        image_path = save_uploaded_file(file)
        
        # Analyze image using existing function
        analysis = openai_client.analyze_image_structured(image_path)
        
        # Clean up temporary file
        os.unlink(image_path)
        
        return ImageAnalysisResponse(**analysis)
        
    except Exception as e:
        logger.exception("Error analyzing image")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request_data: str = Form(..., description="JSON string of VideoGenerationRequest")
):
    """Start video generation process"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Parse request data
        request = VideoGenerationRequest.parse_raw(request_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")
    
    # Validate API keys
    if not config.FAL_API_KEY or not config.OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="API keys not configured")
    
    try:
        # Save uploaded file
        image_path = save_uploaded_file(file)
        
        # Create session
        session_id = str(uuid.uuid4())
        generation_sessions[session_id] = {
            "status": "initializing",
            "progress": 0.0,
            "current_operation": "Preparing...",
            "completed_chains": 0,
            "total_chains": request.num_chains,
            "video_paths": [],
            "final_video_path": None,
            "error_message": None,
            "image_path": image_path,
            "request": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        # Start generation in background
        background_tasks.add_task(run_generation_async, session_id, image_path, request)
        
        return VideoGenerationResponse(
            session_id=session_id,
            status="started",
            message="Video generation started successfully"
        )
        
    except Exception as e:
        logger.exception("Error starting video generation")
        raise HTTPException(status_code=500, detail=f"Error starting generation: {str(e)}")

@app.get("/generation-status/{session_id}", response_model=GenerationStatus)
async def get_generation_status(session_id: str):
    """Get status of video generation session"""
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = generation_sessions[session_id]
    
    return GenerationStatus(
        session_id=session_id,
        status=session["status"],
        progress=session.get("progress", 0.0),
        current_operation=session.get("current_operation", ""),
        completed_chains=session.get("completed_chains", 0),
        total_chains=session.get("total_chains", 0),
        video_paths=session.get("video_paths", []),
        final_video_path=session.get("final_video_path"),
        error_message=session.get("error_message")
    )

@app.post("/cancel-generation/{session_id}")
async def cancel_generation(session_id: str):
    """Cancel video generation session"""
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    generation_sessions[session_id]["cancelled"] = True
    return {"message": "Generation cancellation requested"}

@app.get("/download-video/{session_id}")
async def download_video(session_id: str):
    """Download final generated video"""
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = generation_sessions[session_id]
    
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Generation not completed")
    
    final_video_path = session.get("final_video_path")
    if not final_video_path or not os.path.exists(final_video_path):
        raise HTTPException(status_code=404, detail="Final video not found")
    
    return FileResponse(
        final_video_path,
        media_type="video/mp4",
        filename=f"generated_story_{session_id}.mp4"
    )

@app.get("/download-chain/{session_id}/{chain_index}")
async def download_chain_video(session_id: str, chain_index: int):
    """Download individual chain video"""
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = generation_sessions[session_id]
    video_paths = session.get("video_paths", [])
    
    if chain_index < 0 or chain_index >= len(video_paths):
        raise HTTPException(status_code=404, detail="Chain not found")
    
    video_path = video_paths[chain_index]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Chain video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"chain_{chain_index + 1}_{session_id}.mp4"
    )

@app.get("/sessions")
async def list_sessions():
    """List all generation sessions"""
    sessions = []
    for session_id, session in generation_sessions.items():
        sessions.append({
            "session_id": session_id,
            "status": session["status"],
            "created_at": session["created_at"],
            "progress": session.get("progress", 0.0),
            "completed_chains": session.get("completed_chains", 0),
            "total_chains": session.get("total_chains", 0)
        })
    return {"sessions": sessions}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete generation session and cleanup files"""
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = generation_sessions[session_id]
    
    # Cleanup files
    try:
        if "image_path" in session and os.path.exists(session["image_path"]):
            os.unlink(session["image_path"])
        
        # Note: Video files are in output directory, might want to keep them
        # Add cleanup logic here if needed
        
    except Exception as e:
        logger.warning(f"Error cleaning up session {session_id}: {str(e)}")
    
    del generation_sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.post("/generate-prompt")
async def generate_cinematic_prompt_endpoint(
    action_direction: str = Form(...),
    scene_vision: str = Form(...),
    frame_description: str = Form(...),
    image_description: str = Form(...),
    theme: str = Form(...),
    background: str = Form(...),
    main_subject: str = Form(...),
    tone_and_color: str = Form(...),
    current_chain: int = Form(...),
    total_chains: int = Form(...)
):
    """Generate cinematic prompt using existing LangChain functionality"""
    try:
        prompt = generate_cinematic_prompt(
            action_direction=action_direction,
            scene_vision=scene_vision,
            frame_description=frame_description,
            image_description=image_description,
            theme=theme,
            background=background,
            main_subject=main_subject,
            tone_and_color=tone_and_color,
            current_chain=current_chain,
            total_chains=total_chains
        )
        
        return {"prompt": prompt}
        
    except Exception as e:
        logger.exception("Error generating cinematic prompt")
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if API keys are configured
        api_keys_ok = bool(config.FAL_API_KEY and config.OPENAI_API_KEY)
        
        # Check if output directory exists
        output_dir_ok = os.path.exists(config.OUTPUT_DIR)
        
        return {
            "status": "healthy" if api_keys_ok and output_dir_ok else "degraded",
            "api_keys_configured": api_keys_ok,
            "output_directory_exists": output_dir_ok,
            "active_sessions": len(generation_sessions)
        }
        
    except Exception as e:
        logger.exception("Error in health check")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 