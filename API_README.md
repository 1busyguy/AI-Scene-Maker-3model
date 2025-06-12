# AI Scene Maker FastAPI

A professional REST API wrapper for the AI Scene Maker video generation system. This API provides programmatic access to all video generation functionality without using Gradio.

## 🚀 Quick Start

### 1. Install Additional Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and health |
| `GET` | `/config` | Current configuration |
| `GET` | `/health` | Health check with status |

### Image Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze-image` | Analyze uploaded image structure |

### Video Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate-video` | Start video generation process |
| `GET` | `/generation-status/{session_id}` | Get generation progress |
| `POST` | `/cancel-generation/{session_id}` | Cancel generation |

### File Downloads

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/download-video/{session_id}` | Download final video |
| `GET` | `/download-chain/{session_id}/{chain_index}` | Download individual chain |

### Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/sessions` | List all sessions |
| `DELETE` | `/session/{session_id}` | Delete session and cleanup |

### Utilities

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate-prompt` | Generate cinematic prompts |

## 🎬 Usage Example

### Basic Video Generation

```python
import requests
import json

# 1. Analyze image
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/analyze-image', files={'file': f})
analysis = response.json()

# 2. Start generation
generation_request = {
    "action_direction": "Character walks forward confidently",
    "theme": analysis["theme"],
    "main_subject": analysis["main_subject"],
    "resolution": "720p",
    "num_chains": 3,
    "model_type": "WAN (Default)"
}

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {'request_data': json.dumps(generation_request)}
    response = requests.post('http://localhost:8000/generate-video', files=files, data=data)

session_id = response.json()['session_id']

# 3. Monitor progress
while True:
    status = requests.get(f'http://localhost:8000/generation-status/{session_id}').json()
    print(f"Progress: {status['progress']:.1f}%")
    if status['status'] == 'completed':
        break
    time.sleep(2)

# 4. Download video
response = requests.get(f'http://localhost:8000/download-video/{session_id}')
with open('output.mp4', 'wb') as f:
    f.write(response.content)
```

## 📊 Request/Response Models

### VideoGenerationRequest

```json
{
  "action_direction": "Character walks forward",
  "theme": "Adventure theme",
  "background": "Mountain landscape",
  "main_subject": "Heroic character",
  "tone_and_color": "Warm, golden tones",
  "resolution": "720p",
  "num_chains": 3,
  "model_type": "WAN (Default)",
  "enable_character_consistency": true,
  "enable_face_enhancement": true,
  "quality_vs_speed": "Maximum Quality"
}
```

### GenerationStatus Response

```json
{
  "session_id": "uuid-string",
  "status": "running",
  "progress": 45.2,
  "current_operation": "Generating chain 2/3...",
  "completed_chains": 1,
  "total_chains": 3,
  "video_paths": ["/path/to/chain1.mp4"],
  "final_video_path": null,
  "error_message": null
}
```

## 🎛️ Model Configuration

### Available Models

- **WAN (Default)**: High-quality, 81 frames, customizable FPS
- **Pixverse v3.5**: Style-aware, 5-8 seconds, multiple aspect ratios
- **LUMA Ray2**: Premium quality, 5 seconds, cinematic
- **Kling 2.1 PRO**: Advanced AI, 5-10 seconds, creative control

### Model-Specific Parameters

#### Pixverse v3.5
```json
{
  "model_type": "Pixverse v3.5",
  "pixverse_duration": "5",
  "pixverse_style": "anime",
  "pixverse_negative_prompt": "blur, distortion"
}
```

#### LUMA Ray2
```json
{
  "model_type": "LUMA Ray2",
  "luma_duration": "5",
  "luma_aspect_ratio": "16:9"
}
```

#### Kling 2.1 PRO
```json
{
  "model_type": "Kling 2.1 PRO",
  "kling_duration": "5",
  "kling_aspect_ratio": "16:9",
  "kling_creativity": 0.7
}
```

## 🔧 Advanced Features

### Character Consistency
- `enable_character_consistency`: Maintains character appearance across chains
- `enable_face_enhancement`: Improves face quality and consistency
- `enable_face_swapping`: Replaces faces for perfect consistency

### Quality Control
- `quality_vs_speed`: "Maximum Quality", "Balanced", or "Speed"
- `enable_quality_preservation`: Prevents degradation across chains
- `inference_steps`: Control generation quality (1-40)

## 🛡️ Error Handling

The API provides comprehensive error handling:

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

Common error codes:
- `400`: Bad request (invalid parameters)
- `404`: Session/resource not found
- `500`: Internal server error

## 🔍 Monitoring and Debugging

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "api_keys_configured": true,
  "output_directory_exists": true,
  "active_sessions": 2
}
```

### Session Management
```bash
# List all sessions
curl http://localhost:8000/sessions

# Delete specific session
curl -X DELETE http://localhost:8000/session/{session_id}
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt -r requirements_api.txt

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
Ensure these are set:
```bash
export OPENAI_API_KEY="your-openai-key"
export FAL_API_KEY="your-fal-key"
export OUTPUT_DIR="./outputs"
```

## 📝 Complete Example

Run the included example:
```bash
python api_usage_example.py
```

This demonstrates the complete workflow from image analysis to video download.

## 🔗 Integration Notes

- **Async Support**: All endpoints are async-compatible
- **CORS Enabled**: Cross-origin requests supported
- **File Streaming**: Large video downloads use streaming
- **Session Persistence**: Sessions persist across API restarts
- **Background Processing**: Video generation runs in background
- **Real-time Updates**: Poll status endpoint for progress

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Monitoring**: http://localhost:8000/health 