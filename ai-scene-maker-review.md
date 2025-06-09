# AI Scene Maker - Comprehensive Review & Installation Guide

## 📋 Table of Contents
1. [Installation Guide](#installation-guide)
2. [File Structure Breakdown](#file-structure-breakdown)
3. [Character Consistency Improvements](#character-consistency-improvements)
4. [Face Replacement Options](#face-replacement-options)
5. [Additional Enhancement Methods](#additional-enhancement-methods)

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 or higher
- Git
- FFmpeg (optional but recommended for video processing)
- API Keys:
  - OpenAI API Key
  - FAL.ai API Key

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/[username]/ai-scene-maker-3-models.git
cd ai-scene-maker-3-models
```

#### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install FFmpeg (Optional but Recommended)

**Windows:**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the archive
3. Add the `bin` folder to your system PATH
4. Verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### 5. Set Up API Keys

Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
FAL_API_KEY=your_fal_api_key_here

# Optional Configuration
OUTPUT_DIR=./outputs
MAX_RETRIES=3
DEFAULT_RESOLUTION=720p
DEFAULT_INFERENCE_STEPS=40
DEFAULT_SAFETY_CHECKER=False
```

#### 6. Run the Application
```bash
python app.py
```

The application will launch in your browser at `http://localhost:7860`

### Troubleshooting

**Issue: Missing API Keys Error**
- Ensure your `.env` file exists and contains valid API keys
- Restart the application after adding keys

**Issue: FFmpeg Not Found**
- The app will still work but video stitching may be limited
- Install FFmpeg following the steps above

**Issue: ImportError**
- Ensure you're in the virtual environment
- Run `pip install -r requirements.txt` again

---

## 📁 File Structure Breakdown

### Root Files

#### `app.py`
- **Purpose**: Main entry point for the application
- **Key Functions**:
  - Sets up logging configuration with custom filters
  - Suppresses verbose HTTP request logs
  - Creates and launches the Gradio UI

#### `config.py`
- **Purpose**: Central configuration management
- **Key Functions**:
  - Loads environment variables from `.env`
  - Validates required API keys
  - Sets default values for application settings
  - Creates output directories

#### `requirements.txt`
- **Purpose**: Lists all Python dependencies
- **Key Packages**:
  - `gradio`: Web UI framework
  - `fal-client`: FAL.ai API client
  - `openai`: OpenAI API client
  - `langchain`: LLM orchestration
  - `moviepy`: Video processing
  - `opencv-python`: Computer vision operations

### UI Module (`ui/`)

#### `gradio_ui.py`
- **Purpose**: Complete UI implementation
- **Key Components**:
  - `create_ui()`: Main UI builder with tabs for generation and API setup
  - `on_image_upload()`: Handles image analysis when uploaded
  - `ui_start_chain_generation()`: Manages the video generation process
  - `start_chain_generation_with_updates()`: Core chain generation logic
  - Progress tracking and real-time updates
  - Gallery display for individual video chains

### Utils Module (`utils/`)

#### `fal_client.py`
- **Purpose**: Interface with FAL.ai API
- **Key Functions**:
  - `generate_video_from_image()`: Creates videos from images using various models (WAN, Pixverse, LUMA, Kling)
  - `upload_file()`: Uploads images to FAL.ai storage
  - `download_video()`: Downloads generated videos
  - Model-specific parameter handling

#### `openai_client.py`
- **Purpose**: Interface with OpenAI API
- **Key Functions**:
  - `analyze_image_structured()`: Extracts theme, background, subject, tone, and action
  - `image_to_text()`: Basic image description
  - `generate_scene_vision()`: Creates cohesive scene descriptions
  - `determine_optimal_chain_count()`: Auto-calculates video chains needed

#### `langchain_prompts.py`
- **Purpose**: Advanced prompt generation using LangChain
- **Key Functions**:
  - `generate_cinematic_prompt()`: Creates cinematic prompts maintaining continuity
  - Considers story phases (Establishing, Setup, Development, Resolution)
  - Implements cinematography techniques

#### `video_processing.py`
- **Purpose**: Video and image processing operations
- **Key Functions**:
  - `extract_simple_last_frame()`: Gets the last frame for continuity
  - `auto_adjust_image()`: Fixes brightness/saturation issues
  - `stitch_videos()`: Combines multiple videos using FFmpeg or moviepy
  - `enhance_frame_quality()`: Improves frame quality between chains

#### `helpers.py`
- **Purpose**: General utility functions
- **Key Functions**:
  - File management utilities
  - Logging helpers
  - Filename sanitization

