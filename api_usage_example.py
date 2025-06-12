"""
AI Scene Maker API Usage Example

This script demonstrates how to use the FastAPI endpoints to generate videos.
Run the API server first: uvicorn api:app --reload
"""

import requests
import json
import time
from pathlib import Path

# API base URL
API_BASE = "http://localhost:8000"

def test_image_analysis(image_path: str):
    """Test image analysis endpoint"""
    print("🔍 Testing image analysis...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE}/analyze-image", files=files)
    
    if response.status_code == 200:
        analysis = response.json()
        print("✅ Image analysis successful:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        return analysis
    else:
        print(f"❌ Image analysis failed: {response.text}")
        return None

def start_video_generation(image_path: str, request_data: dict):
    """Start video generation"""
    print("🎬 Starting video generation...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'request_data': json.dumps(request_data)}
        response = requests.post(f"{API_BASE}/generate-video", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        session_id = result['session_id']
        print(f"✅ Generation started with session ID: {session_id}")
        return session_id
    else:
        print(f"❌ Failed to start generation: {response.text}")
        return None

def monitor_generation(session_id: str):
    """Monitor video generation progress"""
    print(f"📊 Monitoring generation progress for session: {session_id}")
    
    while True:
        response = requests.get(f"{API_BASE}/generation-status/{session_id}")
        
        if response.status_code == 200:
            status = response.json()
            print(f"Status: {status['status']} | Progress: {status['progress']:.1f}% | "
                  f"Operation: {status['current_operation']} | "
                  f"Chains: {status['completed_chains']}/{status['total_chains']}")
            
            if status['status'] in ['completed', 'error', 'cancelled']:
                return status
                
        time.sleep(2)  # Poll every 2 seconds

def download_video(session_id: str, output_path: str):
    """Download the final generated video"""
    print(f"⬇️ Downloading final video...")
    
    response = requests.get(f"{API_BASE}/download-video/{session_id}")
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Video saved to: {output_path}")
        return True
    else:
        print(f"❌ Failed to download video: {response.text}")
        return False

def main():
    """Main example workflow"""
    # Configuration
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    output_path = "generated_video.mp4"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable with a valid image file.")
        return
    
    # 1. Test API health
    print("🏥 Checking API health...")
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✅ API Status: {health['status']}")
        print(f"   API Keys Configured: {health['api_keys_configured']}")
    else:
        print("❌ API health check failed")
        return
    
    # 2. Analyze image
    analysis = test_image_analysis(image_path)
    if not analysis:
        return
    
    # 3. Prepare generation request
    generation_request = {
        "action_direction": "The character slowly walks forward with confidence",
        "theme": analysis.get("theme"),
        "background": analysis.get("background"), 
        "main_subject": analysis.get("main_subject"),
        "tone_and_color": analysis.get("tone_and_color"),
        "resolution": "720p",
        "num_chains": 3,
        "model_type": "WAN (Default)",
        "enable_character_consistency": True,
        "enable_face_enhancement": True,
        "quality_vs_speed": "Maximum Quality"
    }
    
    print(f"📝 Generation request: {json.dumps(generation_request, indent=2)}")
    
    # 4. Start generation
    session_id = start_video_generation(image_path, generation_request)
    if not session_id:
        return
    
    # 5. Monitor progress
    final_status = monitor_generation(session_id)
    
    # 6. Download result
    if final_status['status'] == 'completed':
        success = download_video(session_id, output_path)
        if success:
            print(f"🎉 Video generation completed successfully!")
            print(f"   Final video: {output_path}")
            print(f"   Session ID: {session_id}")
        
        # 7. List individual chain videos
        print(f"\n📹 Individual chain videos available:")
        for i in range(final_status['completed_chains']):
            print(f"   Chain {i+1}: {API_BASE}/download-chain/{session_id}/{i}")
    else:
        print(f"❌ Generation failed with status: {final_status['status']}")
        if final_status.get('error_message'):
            print(f"   Error: {final_status['error_message']}")

if __name__ == "__main__":
    main() 