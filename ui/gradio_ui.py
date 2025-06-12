import gradio as gr
import os
import logging
import time
import tempfile
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from utils import fal_client, openai_client, video_processing
from utils.langchain_prompts import generate_cinematic_prompt
from config import OUTPUT_DIR
import config
import json
from dotenv import set_key, find_dotenv

# Add chain state management imports
from utils.video_processing import (
    reset_chain_state, 
    validate_chain_input, 
    log_chain_debug_info,
    get_chain_state_manager
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Try to import enhanced quality modules
try:
    from utils.enhanced_fal_client import (
        upload_file_with_maximum_quality, 
        generate_video_with_quality_optimization,
        integrate_quality_preservation_into_chain
    )
    from utils.enhanced_video_processing import ChainQualityPreserver
    ENHANCED_QUALITY_AVAILABLE = True
    logger.info("Enhanced quality preservation modules loaded")
except ImportError as e:
    ENHANCED_QUALITY_AVAILABLE = False
    logger.warning(f"Enhanced quality modules not available: {e}")

# Suppress excessive logging from HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)

# Define CSS for customizing the UI
css = """
.output-video {
    max-height: 500px;
    margin: auto;
}
/* Ensure consistent sizing for gallery items */
.gallery-item {
    min-height: 200px;
    max-height: 250px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
}
/* Style videos and images in gallery */
.gallery-item img, .gallery-item video {
    object-fit: contain !important;
    max-height: 180px;
    width: 100%;
    margin: 0 auto;
}
/* Add some spacing and borders to make gallery items distinct */
#chain_gallery .gallery-item {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin: 8px;
    background-color: #f9f9f9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.2s ease-in-out;
}
#chain_gallery .gallery-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}
/* Make captions clearer */
#chain_gallery .caption {
    text-align: center;
    font-weight: bold;
    margin-top: 8px;
    background-color: rgba(0,0,0,0.05);
    padding: 4px;
    border-radius: 4px;
    position: absolute;
    bottom: 4px;
    left: 10px;
    right: 10px;
}
/* Make the gallery scrollable if needed */
#chain_gallery .gallery-container {
    overflow-y: auto;
    max-height: 700px;
    scrollbar-width: thin;
    scrollbar-color: #ccc transparent;
}
/* Style scrollbar for better visibility */
#chain_gallery .gallery-container::-webkit-scrollbar {
    width: 8px;
}
#chain_gallery .gallery-container::-webkit-scrollbar-track {
    background: transparent;
}
#chain_gallery .gallery-container::-webkit-scrollbar-thumb {
    background-color: #ccc;
    border-radius: 4px;
}
"""

# Suppress the specific Windows asyncio errors related to connection reset
# These are not critical and just clutter the terminal
if os.name == 'nt':  # Only on Windows
    # Set higher log level for asyncio to suppress connection reset errors
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    
    # Add a filter to remove the specific ConnectionResetError messages
    class ConnectionResetFilter(logging.Filter):
        def filter(self, record):
            return 'ConnectionResetError: [WinError 10054]' not in record.getMessage()
    
    asyncio_logger = logging.getLogger('asyncio')
    asyncio_logger.addFilter(ConnectionResetFilter())

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_api_key(key):
    """Mask API key for display"""
    if not key:
        return ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]

def load_current_keys():
    """Load current API keys (masked)"""
    openai_key = config.OPENAI_API_KEY or ""
    fal_key = config.FAL_API_KEY or ""
    
    # Mask the keys for display
    masked_openai = mask_api_key(openai_key)
    masked_fal = mask_api_key(fal_key)
    
    return {"openai_key": openai_key, "fal_key": fal_key}

def save_api_keys(openai_key, fal_key):
    """Save API keys to .env file"""
    try:
        # Find the .env file
        env_file = find_dotenv()
        
        # Create .env file if it doesn't exist
        if not env_file:
            with open(".env", "w") as f:
                f.write("# API Keys for AI Story Generator\n")
            env_file = find_dotenv()
            
        # Save keys to .env file
        if openai_key:
            set_key(env_file, "OPENAI_API_KEY", openai_key)
        if fal_key:
            set_key(env_file, "FAL_API_KEY", fal_key)
            
        # Update environment variables in the current session
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            config.OPENAI_API_KEY = openai_key
        if fal_key:
            os.environ["FAL_API_KEY"] = fal_key
            config.FAL_API_KEY = fal_key
            
        # If both keys provided, return success message
        if openai_key and fal_key:
            return "Both API keys saved successfully! Please restart the application for changes to take full effect."
        # If only one key provided, return partial success message
        elif openai_key or fal_key:
            keys_saved = []
            if openai_key:
                keys_saved.append("OpenAI")
            if fal_key:
                keys_saved.append("FAL.ai")
            return f"{' and '.join(keys_saved)} API key(s) saved successfully! Please restart the application for changes to take full effect."
        else:
            return "No API keys were provided."
        
    except Exception as e:
        logger.exception(f"Error saving API keys: {str(e)}")
        return f"Error saving API keys: {str(e)}"

def start_chain_generation_with_updates(action_direction, image, theme=None, background=None, 
                                       main_subject=None, tone_and_color=None, scene_vision=None, 
                                       resolution="720p", inference_steps=40, safety_checker=False, 
                                       prompt_expansion=True, num_chains=3, seed=-1, 
                                       model_type="WAN (Default)", model_params=None, 
                                       cancel_requested=lambda: False,
                                       enable_character_consistency=True,
                                       enable_face_enhancement=True,
                                       enable_face_swapping=False,
                                       enable_quality_preservation=True,
                                       quality_vs_speed="Maximum Quality"):
    """
    Generate a series of AI-generated videos with updates for the gradio UI
    Returns a generator that yields (update_type, data, message)
    """
    # CRITICAL: Reset chain state at the beginning of each session
    reset_chain_state()
    yield "progress", 0.01, "Initializing chain state management..."
    
    session_timestamp = int(time.time())
    session_dir = os.path.join(OUTPUT_DIR, f"story_generation_{session_timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Initialize chain state manager
    chain_state = get_chain_state_manager()
    
    # Handle image input and set as original
    current_image_path = None
    if isinstance(image, str):
        current_image_path = os.path.join(session_dir, "original_input.png")
        import shutil
        shutil.copy2(image, current_image_path)
    else:
        current_image_path = os.path.join(session_dir, "original_input.png")
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(current_image_path)
        elif isinstance(image, Image.Image):
            image.save(current_image_path)
    
    # CRITICAL: Set original image in chain state
    chain_state.set_original_image(current_image_path)
    
    # Auto-adjust ONLY the original image (prevent re-adjustment)
    yield "progress", 0.05, "Processing original image (one-time adjustment)..."
    try:
        # Force adjustment for original image, but mark to prevent re-processing
        adjusted_path, adjustments = video_processing.auto_adjust_image(
            current_image_path, 
            force_adjustment=True  # Only for the original image
        )
        if adjustments:
            logger.info(f"Original image adjustments: {adjustments}")
            current_image_path = adjusted_path
            # Update chain state with adjusted original
            chain_state.set_original_image(current_image_path)
    except Exception as e:
        logger.error(f"Error adjusting original image: {str(e)}")
    
    # Upload original image
    yield "progress", 0.1, "Uploading original image..."
    try:
        current_image_url = fal_client.upload_file(current_image_path)
        logger.info(f"Original image uploaded: {current_image_url}")
    except Exception as e:
        logger.error(f"Failed to upload original image: {str(e)}")
        yield "error", None, f"Failed to upload image: {str(e)}"
        return
    
    # Initialize tracking variables
    video_paths = []
    consistency_msg = ""
    
    # Main chain generation loop
    for chain in range(num_chains):
        if cancel_requested():
            yield "cancelled", None, "Generation cancelled by user"
            return
        
        chain_number = chain + 1
        base_progress = 0.2 + (chain * 0.7 / num_chains)
        step_size = 0.7 / num_chains
        
        yield "progress", base_progress, f"Starting chain {chain_number}/{num_chains}..."
        
        # CRITICAL: Start new chain in state manager
        selected_model = get_selected_model(model_type)
        chain_state.start_new_chain(chain_number, selected_model)
        
        # CRITICAL: Validate and correct input image for this chain
        expected_input_image = chain_state.get_current_input_image()
        
        # For KLING and other models, ensure we're using the RIGHT input image
        if chain_number == 1:
            # First chain should always use original
            if current_image_url != fal_client.upload_file(chain_state.original_image_path):
                logger.warning(f"🚨 FIRST CHAIN: Correcting input to use original image")
                current_image_url = fal_client.upload_file(chain_state.original_image_path)
        else:
            # Subsequent chains should use extracted frame from previous chain
            is_valid, corrected_path, message = validate_chain_input(chain_number, expected_input_image)
            
            if not is_valid:
                logger.warning(f"🚨 CHAIN {chain_number}: Input validation failed - {message}")
                # Re-upload the correct image
                yield "progress", base_progress + step_size * 0.1, f"Correcting input for chain {chain_number}..."
                try:
                    current_image_url = fal_client.upload_file(corrected_path)
                    logger.info(f"🔧 CHAIN {chain_number}: Corrected input uploaded: {current_image_url}")
                except Exception as e:
                    logger.error(f"Failed to upload corrected image: {str(e)}")
                    yield "error", None, f"Failed to correct chain input: {str(e)}"
                    return
        
        # Log current chain state for debugging
        log_chain_debug_info()
        
        # Generate enhanced prompt
        yield "progress", base_progress + step_size * 0.2, f"Generating enhanced prompt for chain {chain_number}..."
        try:
            # Use existing prompt generation logic
            cinematic_prompt = generate_cinematic_prompt(
                action_direction=action_direction,
                scene_vision=scene_vision,
                frame_description="",  # Will be updated in the loop
                image_description=f"A scene showing {main_subject} with {background} in the background.",
                theme=theme,
                background=background,
                main_subject=main_subject,
                tone_and_color=tone_and_color,
                current_chain=chain,
                total_chains=num_chains
            )
            logger.info(f"Chain {chain_number} prompt: {cinematic_prompt}")
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            cinematic_prompt = action_direction  # Fallback
        
        # Generate video
        yield "progress", base_progress + step_size * 0.3, f"Generating video for chain {chain_number}..."
        
        # Prepare video generation parameters
        video_gen_params = {
            'prompt': cinematic_prompt,
            'image_url': current_image_url,
            'resolution': resolution,
            'model': selected_model
        }
        # Add model-specific parameters if they exist
        if model_params:
            video_gen_params.update(model_params)
        
        if selected_model.lower() == "wan":
            video_gen_params.update({
                'inference_steps': inference_steps,
                'safety_checker': safety_checker,
                'prompt_expansion': prompt_expansion
            })
        
        try:
            video_url = fal_client.generate_video_from_image(**video_gen_params)
            logger.info(f"Chain {chain_number} video generated: {video_url}")
        except Exception as e:
            logger.error(f"Video generation failed for chain {chain_number}: {str(e)}")
            yield "error", None, f"Video generation failed: {str(e)}"
            return
        
        # Download video
        yield "progress", base_progress + step_size * 0.6, f"Downloading video for chain {chain_number}..."
        try:
            video_filename = f"chain_{chain_number:02d}_{selected_model.lower()}.mp4"
            video_path = os.path.join(session_dir, video_filename)
            
            fal_client.download_video(video_url, video_path)
            video_paths.append(video_path)
            
            logger.info(f"Chain {chain_number} video downloaded: {video_path}")
        except Exception as e:
            logger.error(f"Video download failed for chain {chain_number}: {str(e)}")
            yield "error", None, f"Video download failed: {str(e)}"
            return
        
        # Extract frame for next chain (if not the last chain)
        if chain < num_chains - 1:  # Not the last chain
            yield "progress", base_progress + step_size * 0.8, f"Extracting frame for next chain..."
            
            try:
                # Use CONTROLLED frame extraction to prevent over-enhancement
                best_frame_path, _ = video_processing.extract_high_quality_frame_for_chain(
                    video_path,
                    os.path.join(session_dir, f"chain_{chain_number:02d}_processed"),
                    original_image_path=chain_state.original_image_path,
                    chain_number=chain_number,
                    avoid_over_enhancement=True  # CRITICAL: Prevent over-enhancement
                )
                
                logger.info(f"🎯 CHAIN {chain_number}: Frame extracted for next chain: {best_frame_path}")
                
                # CRITICAL: Complete chain in state manager
                chain_state.complete_chain(chain_number, video_path, best_frame_path)
                
                # Upload frame for next chain (NO additional enhancement)
                yield "progress", base_progress + step_size * 0.95, f"Uploading frame for chain {chain_number + 1}..."
                try:
                    new_image_url = fal_client.upload_file(best_frame_path)
                    current_image_url = new_image_url
                    logger.info(f"🔗 CHAIN {chain_number}: Frame uploaded for next chain: {new_image_url}")
                except Exception as e:
                    logger.error(f"Frame upload failed: {str(e)}")
                    yield "error", None, f"Frame upload failed: {str(e)}"
                    return
                
            except Exception as e:
                logger.error(f"Frame extraction failed for chain {chain_number}: {str(e)}")
                yield "error", None, f"Frame extraction failed: {str(e)}"
                return
        else:
            # Last chain - just complete it
            chain_state.complete_chain(chain_number, video_path, None)
    
    # Final video stitching
    yield "progress", 0.95, "Stitching final video..."
    try:
        final_video_path = os.path.join(session_dir, f"final_story_{session_timestamp}.mp4")
        video_processing.stitch_videos(video_paths, final_video_path)
        
        logger.info(f"Final video created: {final_video_path}")
    except Exception as e:
        logger.error(f"Video stitching failed: {str(e)}")
        # Use first video as fallback
        final_video_path = video_paths[0] if video_paths else None
    
    # Log final chain state for debugging
    log_chain_debug_info()
    
    yield "final", final_video_path, f"Story generation completed successfully!{consistency_msg}"

def create_ui():
    # Check FFmpeg availability and create warning message
    ffmpeg_warning = ""
    if not getattr(video_processing, 'FFPROBE_AVAILABLE', True):
        ffmpeg_warning = """
        
        ⚠️ **Important Notice:** `ffprobe` is not detected on your system. While video generation will work normally, 
        videos may not display properly in this web interface. 
        
        **To fix this:** Install FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) 
        and ensure both `ffmpeg.exe` and `ffprobe.exe` are in your system PATH or project directory.
        
        Your generated videos will still be saved successfully in the `outputs` folder regardless.
        """
    
    with gr.Blocks(title="AI Scene Maker", css=css, theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            f"""
            # AI Scene Maker
            Upload an image and specify an action direction to generate a sequence of videos showing that action.
            {ffmpeg_warning}
            """
        )
        
        # Setup tabs - rearranged to put Generate Video first
        with gr.Tabs() as tabs:
            # Main UI in the first tab
            with gr.TabItem("Generate Video", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input column (left side)
                        image_upload = gr.Image(type="pil", label="Upload Image")
                        
                        # Add a message area to display image adjustment info
                        image_adjustment_info = gr.Markdown("", elem_id="image_adjustment_info")
                        
                        # Structured image analysis fields
                        theme = gr.Textbox(label="Theme of Image", interactive=True, lines=1)
                        background = gr.Textbox(label="Background Description", interactive=True, lines=3)
                        main_subject = gr.Textbox(label="Main Subject Description", interactive=True, lines=3)
                        tone_and_color = gr.Textbox(label="Tone and Color", interactive=True, lines=3)
                        action_direction = gr.Textbox(label="Action Direction", interactive=True, lines=3)
                        
                        # Add a button to explicitly update scene vision
                        with gr.Row():
                            update_vision_btn = gr.Button("Update Scene Vision", variant="secondary")
                        
                        # Scene vision based on analysis
                        scene_vision = gr.Textbox(label="Scene Vision", lines=6)

                        # Function to update scene vision when fields are edited by user
                        def update_scene_vision(theme_val, background_val, subject_val, tone_color_val, action_val):
                            if not all([theme_val, background_val, subject_val, tone_color_val, action_val]):
                                return gr.update(value=scene_vision.value, label="Scene Vision")  # Keep existing value if any field is empty
                            
                            # First stage is to indicate processing is happening
                            logger.info("Updating scene vision with user-edited fields...")
                            
                            try:
                                # Generate updated scene vision with user-edited values
                                scene_vision_prompt = f"""
                                Create a comprehensive scene vision based on the following analysis:
                                
                                Theme: {theme_val}
                                Background: {background_val}
                                Main Subject: {subject_val}
                                Tone and Color: {tone_color_val}
                                Action Direction: {action_val}
                                
                                Focus primarily on the main subject while maintaining the background elements, theme, and visual tone described.
                                """
                                
                                updated_vision = openai_client.generate_scene_vision(scene_vision_prompt, subject_val)
                                logger.info("Scene vision updated with user-edited fields")
                                
                                # Return updated vision with normal label
                                return gr.update(value=updated_vision, label="Scene Vision (Updated)")
                            except Exception as e:
                                logger.exception(f"Error updating scene vision: {str(e)}")
                                # Return existing value with error label
                                return gr.update(value=scene_vision.value, label="Scene Vision (Update failed - will use current values)")
                        
                        # Connect the update button to the update function
                        update_vision_btn.click(
                            fn=update_scene_vision,
                            inputs=[theme, background, main_subject, tone_and_color, action_direction],
                            outputs=[scene_vision],
                            show_progress=True
                        )
                        
                        # Remove the automatic update on field changes to give user more control
                        # and eliminate unnecessary API calls
                        
                        gr.Markdown("### Generation Settings")
                        
                        with gr.Row():
                            model_type = gr.Dropdown(
                                choices=["WAN (Default)", "Pixverse v3.5", "LUMA Ray2", "Kling 2.1 PRO"], 
                                value="WAN (Default)", 
                                label="Model"
                            )
                            
                            # WAN resolution options
                            wan_resolution = gr.Dropdown(
                                choices=["360p", "540p", "720p"], 
                                value="720p", 
                                label="Resolution",
                                visible=True
                            )
                            
                            # Pixverse resolution options (includes 1080p)
                            pixverse_resolution = gr.Dropdown(
                                choices=["360p", "540p", "720p", "1080p"], 
                                value="720p", 
                                label="Resolution",
                                visible=False
                            )
                            
                            # LUMA resolution options (540p, 720p, 1080p)
                            luma_resolution = gr.Dropdown(
                                choices=["540p", "720p", "1080p"], 
                                value="540p", 
                                label="Resolution",
                                visible=False
                            )
                            
                            # Kling resolution options (720p, 1080p)
                            kling_resolution = gr.Dropdown(
                                choices=["720p", "1080p"], 
                                value="720p", 
                                label="Resolution",
                                visible=False
                            )
                        
                        # Pixverse-specific options group
                        with gr.Group(visible=False) as pixverse_options:
                            with gr.Row():
                                pixverse_duration = gr.Dropdown(
                                    choices=["5", "8"], 
                                    value="5", 
                                    label="Duration (seconds)"
                                )
                                pixverse_style = gr.Dropdown(
                                    choices=["None", "anime", "3d_animation", "clay", "comic", "cyberpunk"], 
                                    value="None", 
                                    label="Style"
                                )
                            
                            pixverse_negative_prompt = gr.Textbox(
                                label="Negative Prompt", 
                                placeholder="Enter negative terms to exclude from generation",
                                value="cartoon, anime, illustration, drawing, painting, 3d, cgi, render, fake, doll, plastic, mannequin, unrealistic eyes, lowres, deformed, glitch, artifact, bad anatomy, bad proportions, blurry, grainy, oversaturated, oversharpened, watermark, signature, text, frame, border, overexposed, underexposed, unnatural lighting",
                                lines=2
                            )
                            
                        # LUMA-specific options group
                        with gr.Group(visible=False) as luma_options:
                            with gr.Row():
                                luma_duration = gr.Dropdown(
                                    choices=["5"], 
                                    value="5", 
                                    label="Duration (seconds)",
                                    info="LUMA Ray2 only supports 5 second duration"
                                )
                                luma_aspect_ratio = gr.Dropdown(
                                    choices=["16:9", "9:16", "4:3", "3:4", "21:9", "9:21"], 
                                    value="16:9", 
                                    label="Aspect Ratio"
                                )
                        
                        # Kling 2.1 PRO specific options group
                        with gr.Group(visible=False) as kling_options:
                            with gr.Row():
                                kling_duration = gr.Dropdown(
                                    choices=["5", "10"], 
                                    value="5", 
                                    label="Duration (seconds)",
                                    info="Kling 2.1 PRO supports 5 or 10 second duration"
                                )
                                kling_aspect_ratio = gr.Dropdown(
                                    choices=["16:9", "9:16", "1:1"], 
                                    value="16:9", 
                                    label="Aspect Ratio"
                                )
                            
                            kling_negative_prompt = gr.Textbox(
                                label="Negative Prompt", 
                                placeholder="Enter negative terms to exclude from generation",
                                value="cartoon, anime, illustration, drawing, painting, 3d, cgi, render, fake, doll, plastic, mannequin, extra fingers, extra limbs, mutated hands, fused fingers, distorted face, poorly drawn hands, unrealistic eyes, lowres, deformed, glitch, artifact, bad anatomy, bad proportions, blurry, grainy, oversaturated, oversharpened, watermark, signature, text, frame, border, overexposed, underexposed, unnatural lighting",
                                lines=2
                            )
                            
                            kling_creativity = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="Creativity Level",
                                info="Higher values = more creative/unexpected results"
                            )
                        
                        with gr.Row():
                            inference_steps = gr.Slider(
                                minimum=25, 
                                maximum=40,  # Changed from 50 to 40
                                value=40, 
                                step=1, 
                                label="Inference Steps"
                            )
                        
                        with gr.Row():
                            safety_checker = gr.Checkbox(label="Safety Filter", value=False)
                            prompt_expansion = gr.Checkbox(label="Prompt Expansion", value=True)

                        with gr.Row():
                            auto_determine = gr.Checkbox(label="Auto-determine Chain Count", value=False)
                            num_chains = gr.Slider(
                                minimum=1, 
                                maximum=10, 
                                value=3, 
                                step=1, 
                                label="Number of Chains"
                            )
                            
                        seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        
                        gr.Markdown("### Advanced Features")
                        with gr.Row():
                            enable_character_consistency = gr.Checkbox(
                                label="Enable Character Consistency", 
                                value=True,
                                info="Use AI to maintain character appearance across chains"
                            )
                            enable_face_enhancement = gr.Checkbox(
                                label="Enable Face Enhancement", 
                                value=True,
                                info="Enhance and restore faces for better quality"
                            )
                            enable_face_swapping = gr.Checkbox(
                                label="Enable Face Swapping", 
                                value=False,
                                info="Swap faces to ensure perfect character consistency (slower)"
                            )
                        
                        gr.Markdown("### Quality Settings")
                        with gr.Row():
                            enable_quality_preservation = gr.Checkbox(
                                label="🎯 Maximum Quality Preservation", 
                                value=True,
                                info="Use lossless frame extraction and advanced quality enhancement (slower but maintains detail)"
                            )
                            quality_vs_speed = gr.Radio(
                                choices=["Maximum Quality", "Balanced", "Maximum Speed"],
                                value="Maximum Quality",
                                label="Quality vs Speed Trade-off",
                                info="Quality: Best settings, slower generation | Speed: Faster generation, some quality loss"
                            )
                        
                        with gr.Row():
                            generate_btn = gr.Button("Generate", variant="primary")
                            cancel_btn = gr.Button("Cancel")
                        
                        generation_status = gr.Markdown("Upload an image and click Generate to start")
                        
                        # Add debug panel
                        debug_button, debug_output = create_debug_panel()
                    
                    with gr.Column(scale=1):
                        # Output column (right side)
                        
                        # Progress indicators
                        with gr.Group(visible=False) as progress_group:
                            gr.Markdown("### Generation Progress")
                            overall_progress = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                step=1,
                                label="Overall Progress",
                                interactive=False
                            )
                            chain_progress = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                step=1,
                                label="Current Chain Progress",
                                interactive=False
                            )
                            current_operation = gr.Markdown("Preparing...")
                            eta_display = gr.Markdown("Estimated time remaining: Calculating...")
                        
                        # Quality monitoring display
                        with gr.Group(visible=False) as quality_monitor:
                            gr.Markdown("### 📊 Quality Monitoring")
                            quality_score_display = gr.Number(
                                label="Current Quality Score",
                                value=0.0,
                                precision=2,
                                interactive=False
                            )
                            degradation_warning = gr.Markdown("", elem_id="degradation_warning")
                            quality_recommendations = gr.Textbox(
                                label="Quality Recommendations",
                                lines=3,
                                interactive=False
                            )
                            quality_report_download = gr.File(
                                label="Download Quality Report",
                                visible=False
                            )
                        
                        output_videos = gr.Video(label="Generated Story")
                        
                        chain_gallery = gr.Gallery(
                            label="Individual Chains", 
                            show_label=True, 
                            elem_id="chain_gallery",
                            columns=3, 
                            rows=4,  # Increased from 3 to 4 rows
                            height=700  # Increased from 400 to 700 for better visibility
                        )
                        
                        with gr.Row():
                            save_video_btn = gr.Button("Save Final Video")
                            enhance_btn = gr.Button("Enhance Final Video")
                        
                        # Add a visual separator
                        gr.Markdown("---")
                        
                        # Add Clear and Compose New Scene button in a separate row for emphasis
                        with gr.Row():
                            clear_btn = gr.Button("Clear and Compose New Scene", variant="primary", size="lg")
                        
                        output_status = gr.Markdown("")
                        
                        # Hidden state elements
                        generation_running = gr.State(False)
                        cancel_requested = gr.State(False)
                        video_paths = gr.State([])
                        final_video_path = gr.State(None)
                        download_link = gr.Markdown("")
            
            # API key inputs and save button as a second tab
            with gr.TabItem("Setup API Keys", id="setup"):
                current_keys = load_current_keys()
                
                openai_api_key = gr.Textbox(
                    value=mask_api_key(current_keys["openai_key"]), 
                    type="password", 
                    label="OpenAI API Key", 
                    placeholder="Enter your OpenAI API key"
                )
                
                fal_api_key = gr.Textbox(
                    value=mask_api_key(current_keys["fal_key"]), 
                    type="password", 
                    label="FAL API Key", 
                    placeholder="Enter your FAL API key"
                )
                
                save_keys_btn = gr.Button("Save API Keys")
                keys_status = gr.Markdown("Enter your API keys and click Save")
                
                save_keys_btn.click(
                    fn=save_api_keys,
                    inputs=[openai_api_key, fal_api_key],
                    outputs=[keys_status]
                )
            
            # Character Consistency Report tab
            with gr.TabItem("Consistency Report", id="report"):
                gr.Markdown("""
                ### Character Consistency Analysis
                View detailed reports on character consistency across video chains.
                """)
                
                report_display = gr.JSON(label="Character Consistency Report")
                
                with gr.Row():
                    refresh_report_btn = gr.Button("Refresh Report", variant="secondary")
                    clear_report_btn = gr.Button("Clear Report", variant="secondary")
                
                report_status = gr.Markdown("Click 'Refresh Report' to load the latest consistency analysis.")
                
                def load_report():
                    """Load the character consistency report from the output directory"""
                    try:
                        # Look for the most recent report in any session directory
                        import glob
                        
                        # Pattern to find all character validation reports
                        report_pattern = os.path.join(OUTPUT_DIR, "*/character_validation_report.json")
                        report_files = glob.glob(report_pattern)
                        
                        if not report_files:
                            # Also check for report in the main output directory
                            main_report_path = os.path.join(OUTPUT_DIR, "character_validation_report.json")
                            if os.path.exists(main_report_path):
                                report_files = [main_report_path]
                        
                        if report_files:
                            # Get the most recent report file
                            latest_report = max(report_files, key=os.path.getmtime)
                            
                            with open(latest_report, 'r') as f:
                                report_data = json.load(f)
                            
                            # Add metadata about the report
                            report_data['_metadata'] = {
                                'report_file': latest_report,
                                'last_modified': os.path.getmtime(latest_report),
                                'file_size': os.path.getsize(latest_report)
                            }
                            
                            return report_data, "Report loaded successfully!"
                        else:
                            return {
                                "message": "No character consistency report found",
                                "status": "No reports available",
                                "suggestion": "Generate a video with character consistency enabled to create a report"
                            }, "No reports found. Generate videos with character consistency enabled."
                    
                    except Exception as e:
                        return {
                            "error": str(e),
                            "message": "Failed to load consistency report"
                        }, f"Error loading report: {str(e)}"
                
                def clear_report():
                    """Clear the displayed report"""
                    return {}, "Report display cleared."
                
                refresh_report_btn.click(
                    fn=load_report,
                    outputs=[report_display, report_status]
                )
                
                clear_report_btn.click(
                    fn=clear_report,
                    outputs=[report_display, report_status]
                )

        # Connect the image upload event
        image_upload.change(
            fn=on_image_upload,
            inputs=[image_upload],
            outputs=[theme, background, main_subject, tone_and_color, action_direction, scene_vision, image_adjustment_info]
        )
        
        # Toggle model-specific options based on model selection
        def update_model_options(model_choice):
            if model_choice == "Pixverse v3.5":
                return (
                    gr.update(visible=False),  # wan_resolution
                    gr.update(visible=True),   # pixverse_resolution
                    gr.update(visible=False),  # luma_resolution
                    gr.update(visible=False),  # kling_resolution
                    gr.update(visible=True),   # pixverse_options
                    gr.update(visible=False),  # luma_options
                    gr.update(visible=False),  # kling_options
                    gr.update(visible=False),  # inference_steps (WAN only)
                    gr.update(visible=False),  # safety_checker (WAN only)
                    gr.update(visible=False)   # prompt_expansion (WAN only)
                )
            elif model_choice == "LUMA Ray2":
                return (
                    gr.update(visible=False),  # wan_resolution
                    gr.update(visible=False),  # pixverse_resolution
                    gr.update(visible=True),   # luma_resolution
                    gr.update(visible=False),  # kling_resolution
                    gr.update(visible=False),  # pixverse_options
                    gr.update(visible=True),   # luma_options
                    gr.update(visible=False),  # kling_options
                    gr.update(visible=False),  # inference_steps (WAN only)
                    gr.update(visible=False),  # safety_checker (WAN only)
                    gr.update(visible=False)   # prompt_expansion (WAN only)
                )
            elif model_choice == "Kling 2.1 PRO":
                return (
                    gr.update(visible=False),  # wan_resolution
                    gr.update(visible=False),  # pixverse_resolution
                    gr.update(visible=False),  # luma_resolution
                    gr.update(visible=True),   # kling_resolution
                    gr.update(visible=False),  # pixverse_options
                    gr.update(visible=False),  # luma_options
                    gr.update(visible=True),   # kling_options
                    gr.update(visible=False),  # inference_steps (WAN only)
                    gr.update(visible=False),  # safety_checker (WAN only)
                    gr.update(visible=False)   # prompt_expansion (WAN only)
                )
            else:  # WAN (Default)
                return (
                    gr.update(visible=True),   # wan_resolution
                    gr.update(visible=False),  # pixverse_resolution
                    gr.update(visible=False),  # luma_resolution
                    gr.update(visible=False),  # kling_resolution
                    gr.update(visible=False),  # pixverse_options
                    gr.update(visible=False),  # luma_options
                    gr.update(visible=False),  # kling_options
                    gr.update(visible=True),   # inference_steps (WAN only)
                    gr.update(visible=True),   # safety_checker (WAN only)
                    gr.update(visible=True)    # prompt_expansion (WAN only)
                )
                
        model_type.change(
            fn=update_model_options,
            inputs=[model_type],
            outputs=[
                wan_resolution, pixverse_resolution, luma_resolution, kling_resolution,
                pixverse_options, luma_options, kling_options,
                inference_steps, safety_checker, prompt_expansion
            ]
        )

        # Connect auto-chain toggle to disable/enable chain slider
        auto_determine.change(
            fn=toggle_chains,
            inputs=[auto_determine],
            outputs=[num_chains]
        )

        # Function to get the appropriate resolution based on model selection
        def get_resolution(model_choice, wan_res, pixverse_res, luma_res, kling_res):
            if model_choice == "Pixverse v3.5":
                return pixverse_res
            elif model_choice == "LUMA Ray2":
                return luma_res
            elif model_choice == "Kling 2.1 PRO":
                return kling_res
            else:
                return wan_res

        # Connect the generate button
        generate_btn.click(
            fn=ui_start_chain_generation,
            inputs=[
                action_direction, image_upload, 
                theme, background, main_subject, tone_and_color, scene_vision,
                model_type,
                wan_resolution, pixverse_resolution, luma_resolution, kling_resolution,
                inference_steps, safety_checker, 
                prompt_expansion, auto_determine, num_chains, 
                seed, pixverse_duration, pixverse_style, pixverse_negative_prompt,
                luma_duration, luma_aspect_ratio,
                kling_duration, kling_aspect_ratio, kling_negative_prompt, kling_creativity,
                enable_character_consistency, enable_face_enhancement, enable_face_swapping,
                enable_quality_preservation, quality_vs_speed,
                generation_running, cancel_requested
            ],
            outputs=[
                generation_running, chain_gallery, output_videos, 
                video_paths, final_video_path, generation_status, 
                output_status, download_link, progress_group,
                overall_progress, chain_progress, current_operation, eta_display,
                # Add quality monitoring outputs
                quality_monitor, quality_score_display, degradation_warning,
                quality_recommendations, quality_report_download
            ]
        )

        # Set up cancel button
        cancel_btn.click(
            fn=cancel_generation,
            inputs=[cancel_requested],
            outputs=[cancel_requested, generation_status]
        )

        # Set up save video button
        save_video_btn.click(
            fn=lambda path: gr.update(visible=True) if path else None,
            inputs=[final_video_path],
            outputs=[download_link]
        )

        # Set up enhance video button
        enhance_btn.click(
            fn=enhance_final_video,
            inputs=[final_video_path],
            outputs=[output_videos, output_status]
        )

        # Set up clear button
        clear_btn.click(
            fn=clear_session,
            inputs=[],
            outputs=[
                image_upload, 
                theme, 
                background, 
                main_subject, 
                tone_and_color, 
                action_direction, 
                scene_vision, 
                generation_status, 
                output_status, 
                image_adjustment_info, 
                download_link, 
                overall_progress, 
                chain_progress, 
                current_operation, 
                eta_display, 
                progress_group, 
                output_videos, 
                chain_gallery, 
                generation_running, 
                cancel_requested,
                video_paths, 
                final_video_path,
                enable_character_consistency,
                enable_face_enhancement,
                enable_face_swapping,
                enable_quality_preservation,
                quality_vs_speed
            ]
        )

    return iface

def get_selected_model(model_type: str) -> str:
    """Convert UI model type to internal model name"""
    model_mapping = {
        "WAN (Default)": "wan",
        "Pixverse v3.5": "pixverse", 
        "LUMA Ray2": "luma",
        "Kling 2.1 PRO": "kling"
    }
    return model_mapping.get(model_type, "wan")

def toggle_chains(auto):
    """Toggle chains slider interactivity based on auto-determine checkbox"""
    return gr.update(interactive=not auto)

def ui_start_chain_generation(action_dir, img, theme, background, main_subject, tone_color, vision, 
                           model_selection, wan_res, pixverse_res, luma_res, kling_res, steps, safety, expansion, 
                           auto_determine, chains, seed_val, pixverse_duration="5", 
                           pixverse_style="None", pixverse_negative_prompt="", 
                           luma_duration="5", luma_aspect_ratio="16:9",
                           kling_duration="5", kling_aspect_ratio="16:9", kling_negative_prompt="", kling_creativity=0.5,
                           enable_character_consistency=True, enable_face_enhancement=True, enable_face_swapping=False,
                           enable_quality_preservation=True, quality_vs_speed="Maximum Quality",
                           gen_running=False, cancel_req=False):
    """Handle the UI aspects of chain generation, including updating UI elements during generation"""
    # Update UI for generation start
    gen_running = True
    cancel_req = False
    
    # Clear previous results
    video_paths = []
    final_path = None
    
    # Initialize times for ETA calculation
    start_time = time.time()
    chain_start_time = start_time
    
    # Select the appropriate resolution based on model type
    if model_selection == "Pixverse v3.5":
        resolution = pixverse_res
    elif model_selection == "LUMA Ray2":
        resolution = luma_res
    elif model_selection == "Kling 2.1 PRO":
        resolution = kling_res
    else:
        resolution = wan_res
    
    # Show progress group and initial status
    yield (
        gen_running, None, None, video_paths, final_path, 
        "Starting chain generation...", "", gr.update(visible=False),
        gr.update(visible=True), 0, 0, 
        "Initializing...", "Estimated time remaining: Calculating...",
        # Add quality monitoring updates
        gr.update(visible=enable_quality_preservation), 0.0, "", "", gr.update(visible=False)
    )
    
    try:
        # If auto-determine is enabled, set chains to -1 to indicate auto mode
        if auto_determine:
            chains = -1
            
        # Prepare model-specific parameters
        if model_selection == "Pixverse v3.5":
            model_params = {
                "duration": int(pixverse_duration),
                "style": None if pixverse_style == "None" else pixverse_style,
                "negative_prompt": pixverse_negative_prompt,
                "aspect_ratio": "16:9"
            }
        elif model_selection == "LUMA Ray2":
            model_params = {
                "duration": 5,
                "aspect_ratio": luma_aspect_ratio
            }
        elif model_selection == "Kling 2.1 PRO":
            model_params = {
                "duration": int(kling_duration),
                "aspect_ratio": kling_aspect_ratio,
                "negative_prompt": kling_negative_prompt,
                "creativity": kling_creativity
            }
        else:
            model_params = {}
            
        # Log the user-edited inputs
        logger.info(f"Generation with user-edited fields: theme={theme}, background={background}, main_subject={main_subject}, tone_color={tone_color}, action_dir={action_dir}")
            
        # Run the chain generation with all structured components
        chain_generator = start_chain_generation_with_updates(
            action_direction=action_dir,
            image=img,
            theme=theme,
            background=background,
            main_subject=main_subject,
            tone_and_color=tone_color,
            scene_vision=vision,
            resolution=resolution,
            inference_steps=steps,
            safety_checker=safety,
            prompt_expansion=expansion,
            num_chains=chains,
            seed=seed_val,
            model_type=model_selection,
            model_params=model_params,
            cancel_requested=lambda: cancel_req,
            enable_character_consistency=enable_character_consistency,
            enable_face_enhancement=enable_face_enhancement,
            enable_face_swapping=enable_face_swapping,
            enable_quality_preservation=enable_quality_preservation,
            quality_vs_speed=quality_vs_speed
        )
        
        # Process chain generation results
        chain_videos = {}
        completed_chains = 0
        determined_chains = chains if chains > 0 else 3
        
        for update_type, result, message in chain_generator:
            if update_type == "progress":
                try:
                    chain_percent = float(result) * 100 if isinstance(result, (int, float)) else 0
                except (ValueError, TypeError):
                    chain_percent = 0
                
                overall_percent = (completed_chains / determined_chains) * 100
                if chain_percent > 0:
                    overall_percent += (chain_percent / 100) * (100 / determined_chains)
                    
                elapsed = time.time() - start_time
                if overall_percent > 0:
                    total_estimated = elapsed / (overall_percent / 100)
                    remaining = total_estimated - elapsed
                    eta_text = f"Estimated time remaining: {format_time(remaining)}"
                else:
                    eta_text = "Estimated time remaining: Calculating..."
                
                yield (
                    gen_running, None, None, video_paths, final_path, 
                    f"Progress: {message}", "", gr.update(visible=False),
                    gr.update(visible=True), min(99, overall_percent), chain_percent, 
                    f"Current operation: {message}", eta_text,
                    # Add quality monitoring updates
                    gr.update(visible=enable_quality_preservation), 0.0, "", "", gr.update(visible=False)
                )
            
            elif update_type == "quality_update":
                # Handle quality monitoring updates
                quality_score = result.get("quality_score", 0.0) if isinstance(result, dict) else 0.0
                warning = result.get("warning", "") if isinstance(result, dict) else ""
                recommendations = result.get("recommendations", "") if isinstance(result, dict) else ""
                has_report = result.get("report_path") is not None if isinstance(result, dict) else False
                
                yield (
                    gen_running, None, None, video_paths, final_path, 
                    message, "", gr.update(visible=False),
                    gr.update(visible=True), min(99, overall_percent), chain_percent, 
                    f"Current operation: {message}", eta_text,
                    # Quality monitoring updates
                    gr.update(visible=enable_quality_preservation), quality_score, warning, recommendations, 
                    gr.update(visible=has_report)
                )
            
            elif update_type == "chains":
                # Number of chains was determined
                determined_chains = result
                
                yield (
                    gen_running, None, None, video_paths, final_path, 
                    f"Number of chains determined: {result}", "", gr.update(visible=False),
                    gr.update(visible=True), min(10, (completed_chains / determined_chains) * 100), 0, 
                    "Preparing for video generation...", "Estimated time remaining: Calculating...",
                    # Add quality monitoring updates
                    gr.update(visible=enable_quality_preservation), 0.0, "", "", gr.update(visible=False)
                )
            
            elif update_type == "chain_complete":
                # New chain completed
                completed_chains += 1
                chain_number = int(message.split()[1])  # Extract the chain number from the message "Chain X completed"
                
                # Add a small pause to ensure all file operations are complete
                time.sleep(0.5)
                
                # Store video path by chain number in dictionary (use chain_number directly)
                chain_videos[chain_number] = result
                video_paths.append(result)
                
                # Create sorted gallery items using the chain numbers (without subtraction)
                sorted_chains = sorted(chain_videos.items())  # Sort by chain number
                gallery_items = [(path, f"Chain {number:02d}") for number, path in sorted_chains]
                
                # Log what's being displayed in the gallery for debugging
                logger.info(f"Gallery update - Items: {len(gallery_items)}, Chains: {[num for num, _ in sorted_chains]}")
                
                # Calculate elapsed time for this chain and ETA
                chain_elapsed = time.time() - chain_start_time
                chain_start_time = time.time()  # Reset for next chain
                
                # Calculate overall progress
                overall_percent = (completed_chains / determined_chains) * 100
                
                # Calculate ETA based on average time per chain
                elapsed = time.time() - start_time
                chains_remaining = determined_chains - completed_chains
                if completed_chains > 0:
                    avg_time_per_chain = elapsed / completed_chains
                    remaining = avg_time_per_chain * chains_remaining
                    eta_text = f"Estimated time remaining: {format_time(remaining)}"
                else:
                    eta_text = "Estimated time remaining: Calculating..."
                
                yield (
                    gen_running, gallery_items, None, video_paths, final_path, 
                    f"Chain {chain_number}/{determined_chains} completed ({overall_percent:.1f}% done)", "", gr.update(visible=False),
                    gr.update(visible=True), overall_percent, 100, 
                    f"Completed chain {chain_number}/{determined_chains} in {format_time(chain_elapsed)}", eta_text,
                    # Add quality monitoring updates
                    gr.update(visible=enable_quality_preservation), 0.0, "", "", gr.update(visible=False)
                )
            
            elif update_type == "final":
                # Final video completed
                final_path = result
                overall_percent = 100
                
                yield (
                    gen_running, gallery_items, final_path, video_paths, final_path, 
                    "Story generation completed successfully!", "", gr.update(visible=True),
                    gr.update(visible=True), 100, 100, 
                    "Generation complete!", f"Total time: {format_time(time.time() - start_time)}",
                    # Final quality monitoring update
                    gr.update(visible=False), 0.0, "", "", gr.update(visible=False)
                )
                
            elif update_type == "error":
                # Error occurred
                yield (
                    False, gallery_items if 'gallery_items' in locals() else None, None, 
                    video_paths, final_path, f"Error: {message}", "", gr.update(visible=False),
                    gr.update(visible=False), 0, 0, "", "",
                    # Error quality monitoring update
                    gr.update(visible=False), 0.0, "", "", gr.update(visible=False)
                )
                return
                
            elif update_type == "cancelled":
                # Generation was cancelled
                total_time = time.time() - start_time
                yield (
                    False, gallery_items if 'gallery_items' in locals() else None, None, 
                    video_paths, final_path, "Generation cancelled by user", "", gr.update(visible=False),
                    gr.update(visible=False), 0, 0, 
                    f"Generation cancelled after {format_time(total_time)}", "",
                    # Final quality monitoring update
                    gr.update(visible=False), 0.0, "", "", gr.update(visible=False)
                )
                return
        
        # Generation completed successfully
        gen_running = False
        if final_path:
            total_time = time.time() - start_time
            yield (
                gen_running, gallery_items, final_path, video_paths, final_path, 
                "Story generation completed successfully!", "", gr.update(visible=True),
                gr.update(visible=True), 100, 100, 
                "Generation complete!", f"Total time: {format_time(total_time)}",
                # Final quality monitoring update
                gr.update(visible=False), 0.0, "", "", gr.update(visible=False)
            )
        else:
            yield (
                gen_running, gallery_items if 'gallery_items' in locals() else None, None, 
                video_paths, final_path, "Generation completed but no final video was produced", "", gr.update(visible=False),
                gr.update(visible=False), 0, 0, "", "",
                # Final quality monitoring update
                gr.update(visible=False), 0.0, "", "", gr.update(visible=False)
            )
            
    except Exception as e:
        logger.exception(f"Error in chain generation: {str(e)}")
        yield (
            False, None, None, video_paths, final_path, 
            f"An unexpected error occurred: {str(e)}", "", gr.update(visible=False),
            gr.update(visible=False), 0, 0, "", "",
            # Error quality monitoring update
            gr.update(visible=False), 0.0, "", "", gr.update(visible=False)
        )

# Helper function for formatting time
def format_time(seconds):
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

# Cancel generation function
def cancel_generation(cancel_req):
    return True, "Cancellation requested. Waiting for current operations to complete..."

def on_image_upload(img):
    """
    Handle image upload and perform structured analysis to extract theme, background,
    main subject, tone and color, and action direction.
    """
    if img is None:
        return None, None, None, None, None, None, ""

    try:
        # Initialize adjustment_message
        adjustment_message = ""
        
        # Save image temporarily
        temp_path = os.path.join(tempfile.gettempdir(), "temp_upload.png")
        if isinstance(img, np.ndarray):
            Image.fromarray(img).save(temp_path)
        else:
            img.save(temp_path)

        # Auto-adjust the image if needed (too dark, too bright, or oversaturated)
        adjusted_path, adjustments = video_processing.auto_adjust_image(temp_path)
        
        # If adjustments were made, use the adjusted image
        if adjustments:
            logger.info(f"Auto-adjusted image: {', '.join(adjustments)}")
            temp_path = adjusted_path
            adjustment_message = f"📸 Image automatically adjusted: {', '.join(adjustments)}"

        # Get structured image analysis
        logger.info("Performing structured image analysis...")
        analysis = openai_client.analyze_image_structured(temp_path)
        
        # Generate scene vision based on all components
        scene_vision_prompt = f"""
        Create a comprehensive scene vision based on the following analysis:
        
        Theme: {analysis['theme']}
        Background: {analysis['background']}
        Main Subject: {analysis['main_subject']}
        Tone and Color: {analysis['tone_and_color']}
        Action Direction: {analysis['action_direction']}
        
        Focus primarily on the main subject while maintaining the background elements, theme, and visual tone described.
        """
        
        vision = openai_client.generate_scene_vision(scene_vision_prompt, analysis['main_subject'])

        logger.info(f"Theme: {analysis['theme']}")
        logger.info(f"Background: {analysis['background']}")
        logger.info(f"Main Subject: {analysis['main_subject']}")
        logger.info(f"Tone and Color: {analysis['tone_and_color']}")
        logger.info(f"Action Direction: {analysis['action_direction']}")
        logger.info(f"Generated Scene Vision: {vision}")

        # Add information about adjustments to the action direction if any were made
        action_direction = analysis['action_direction']
        if adjustments:
            action_direction = f"{action_direction} (Note: Image was automatically adjusted: {', '.join(adjustments)})"

        return (
            analysis['theme'],
            analysis['background'],
            analysis['main_subject'],
            analysis['tone_and_color'],
            action_direction,
            vision,
            adjustment_message
        )
    except Exception as e:
        logger.exception("Error processing uploaded image")
        # Return empty values with error message
        return (
            None,
            None,
            None,
            None,
            f"Error: {str(e)}",
            None,
            ""
        )

def enhance_final_video(video_path):
    """
    Apply enhancements to the final video to improve visual quality
    
    Args:
        video_path: Path to the video to enhance
        
    Returns:
        Tuple of (enhanced_video_path, status_message)
    """
    if not video_path or not os.path.exists(video_path):
        return None, "No video available to enhance"
        
    try:
        logger.info(f"Enhancing final video: {video_path}")
        
        # Create output path for enhanced video
        output_dir = os.path.dirname(video_path)
        enhanced_path = os.path.join(output_dir, "enhanced_" + os.path.basename(video_path))
        
        # For now we'll implement a simple enhancement using FFmpeg
        # This command stabilizes the video slightly and enhances colors
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "unsharp=5:5:1.0:5:5:0.0,eq=brightness=0.05:saturation=1.2",
            "-c:v", "libx264", "-crf", "18",
            enhanced_path
        ]
        
        # Convert command to string for logging
        cmd_str = " ".join(command)
        logger.info(f"Running enhancement command: {cmd_str}")
        
        try:
            import subprocess
            result = subprocess.run(command, check=True, capture_output=True)
            logger.info("Video enhancement completed successfully")
            
            if os.path.exists(enhanced_path):
                return enhanced_path, "Video enhancement completed successfully"
            else:
                return video_path, "Enhancement completed but file not found. Using original video."
                
        except subprocess.CalledProcessError as e:
            logger.exception(f"FFmpeg error during enhancement: {e.stderr.decode() if e.stderr else ''}")
            return video_path, f"Enhancement failed: FFmpeg error. Using original video."
            
        except FileNotFoundError:
            logger.warning("FFmpeg not found on system. Skipping enhancement.")
            return video_path, "Enhancement skipped: FFmpeg not installed. Using original video."
    
    except Exception as e:
        logger.exception(f"Error enhancing video: {str(e)}")
        return video_path, f"Enhancement error: {str(e)}. Using original video."

def clear_session():
    """
    Reset the UI to start a fresh session, clearing all previous generation data.
    Note: All generated files remain in the outputs directory for future reference.
    """
    import tempfile
    import os
    
    logger.info("Clearing session and resetting UI for new composition")
    logger.info("All previously generated files will remain in the outputs directory")
    
    try:
        # Clean up temporary files in the temp directory if they exist
        # Only remove files that match our patterns and ONLY from the temp directory
        # NEVER from the output directory
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith("temp_upload") or filename.startswith("adjusted_"):
                try:
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path) and os.path.abspath(file_path).startswith(os.path.abspath(temp_dir)):
                        # Extra safety check to ensure we're only deleting from temp dir
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {filename}: {str(e)}")
        
        # IMPORTANT: We do not delete any files in the output directory
        # This ensures all generated content is preserved
                        
    except Exception as e:
        logger.warning(f"Error cleaning up files: {str(e)}")
    
    # Return a list of reset values for all UI components in the order they appear in the outputs list
    return [
        # Reset the image upload component
        None,
        
        # Reset all text fields
        "",  # theme
        "",  # background
        "",  # main_subject
        "",  # tone_and_color
        "",  # action_direction
        "",  # scene_vision
        
        # Reset UI text elements
        "Session cleared. Upload a new image to start. All previously generated files remain in the outputs directory.",  # generation_status
        "Ready for a new composition. Previous outputs are preserved.",  # output_status
        "",  # image_adjustment_info
        "",  # download_link
        
        # Reset progress indicators
        0,  # overall_progress
        0,  # chain_progress
        "Waiting to start...",  # current_operation
        "Estimated time remaining: N/A",  # eta_display
        gr.update(visible=False),  # progress_group
        
        # Reset output containers
        None,  # output_videos
        None,  # chain_gallery
        
        # Reset state variables
        False,  # generation_running
        False,  # cancel_requested
        [],  # video_paths
        None,   # final_video_path
        True,   # enable_character_consistency (reset to default True)
        True,   # enable_face_enhancement (reset to default True)
        False,  # enable_face_swapping (reset to default False)
        True,   # enable_quality_preservation (reset to default True)
        "Maximum Quality"  # quality_vs_speed (reset to default Maximum Quality)
    ]

def create_debug_panel():
    """Create debug panel for chain state monitoring"""
    
    with gr.Accordion("🔍 Chain Debug Info", open=False):
        gr.Markdown("Monitor chain state and enhancement history")
        
        debug_button = gr.Button("Show Chain Debug Info")
        debug_output = gr.Textbox(
            label="Debug Information",
            lines=10,
            interactive=False
        )
        
        def show_debug_info():
            try:
                from utils.video_processing import get_chain_state_manager, get_enhancement_tracker
                
                chain_state = get_chain_state_manager()
                enhancement_tracker = get_enhancement_tracker()
                
                debug_info = chain_state.get_debug_info()
                
                debug_text = "=== CHAIN STATE ===\n"
                debug_text += f"Current Chain: {debug_info['chain_count']}\n"
                debug_text += f"Original Image: {debug_info['original_image']}\n"
                debug_text += f"Current Image: {debug_info['current_image']}\n"
                debug_text += f"Last Video: {debug_info['last_video']}\n"
                debug_text += f"Last Frame: {debug_info['last_extracted_frame']}\n\n"
                
                debug_text += "=== IMAGE PROGRESSION ===\n"
                for prog in debug_info.get('image_progression', []):
                    debug_text += f"Chain {prog['chain']}: {prog['type']} -> {os.path.basename(prog['path'])}\n"
                
                debug_text += "\n=== ENHANCEMENT HISTORY ===\n"
                for img_hash, history in enhancement_tracker.enhancement_history.items():
                    debug_text += f"Image: {os.path.basename(history['path'])}\n"
                    debug_text += f"  Enhancement: {history['enhancement_level']} ({history['enhancement_count']}x)\n"
                    debug_text += f"  Source: {history['source_type']}\n\n"
                
                return debug_text
                
            except Exception as e:
                return f"Debug info error: {str(e)}"
        
        debug_button.click(show_debug_info, outputs=[debug_output])
        
        return debug_button, debug_output