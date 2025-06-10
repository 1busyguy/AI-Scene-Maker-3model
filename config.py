import os
import sys
import logging
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Required API keys
FAL_API_KEY = os.getenv("FAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional configuration with defaults
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
DEFAULT_RESOLUTION = os.getenv("DEFAULT_RESOLUTION", "720p")
# Ensure inference steps never exceeds 40 (API limit)
DEFAULT_INFERENCE_STEPS = min(int(os.getenv("DEFAULT_INFERENCE_STEPS", "40")), 40)
DEFAULT_SAFETY_CHECKER = os.getenv("DEFAULT_SAFETY_CHECKER", "False").lower() == "true"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate required environment variables
missing_vars = []
if not FAL_API_KEY:
    missing_vars.append("FAL_API_KEY")
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")

if missing_vars:
    error_message = f"""
ERROR: Missing required environment variables: {', '.join(missing_vars)}
Please set these variables in your .env file.

You can create a .env file by copying .env.example:
  cp .env.example .env

Then edit the .env file to add your API keys:
  - Get your OpenAI API key from: https://platform.openai.com/account/api-keys
  - Get your FAL.ai API key from: https://fal.ai/dashboard/keys
"""
    logger.error(error_message)
    
    # Only exit if running as main script, not during imports
    if __name__ == "__main__":
        print(error_message)
        sys.exit(1)

# Display configuration
if __name__ == "__main__":
    print("Configuration loaded successfully:")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Default Resolution: {DEFAULT_RESOLUTION}")
    print(f"  Default Inference Steps: {DEFAULT_INFERENCE_STEPS}")
    print(f"  API Keys Configured: {'Yes' if FAL_API_KEY and OPENAI_API_KEY else 'No'}")

# Face Swapping Configuration
FACE_SWAP_CONFIG = {
    "skip_frames": 1,  # Process every N frames (1 = all frames)
    "similarity_threshold": 0.5,  # Min similarity to swap (0.0-1.0)
    "batch_size": 4,  # Frames to process in parallel
    "use_gpu": True,  # Prefer GPU if available
    "max_video_size": 100,  # Max video size in MB to process
    "quality_preset": "balanced"  # "fast", "balanced", or "quality"
}