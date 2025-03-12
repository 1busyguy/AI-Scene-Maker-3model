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

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

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

def start_chain_generation_with_updates(action_direction, image, theme=None, background=None, main_subject=None, 
                                       tone_and_color=None, scene_vision=None, resolution="720p", 
                                       inference_steps=40, safety_checker=False, prompt_expansion=True, 
                                       num_chains=3, seed=-1, model_type="WAN (Default)", model_params=None, cancel_requested=lambda: False):
    """
    Generate a series of AI-generated videos with updates for the gradio UI
    Returns a generator that yields (update_type, data, message)
    """
    # All videos & images from the entire chain process
    video_paths = []
    frame_paths = []
    final_prompt_list = []
    
    # Initialize default model params if needed
    if model_params is None:
        model_params = {}
    
    # Store the structured image analysis components
    image_analysis = {
        "theme": theme,
        "background": background,
        "main_subject": main_subject,
        "tone_and_color": tone_and_color,
        "action_direction": action_direction
    }
    
    try:
        if cancel_requested():
            logger.info("Generation cancelled by user before starting")
            yield "cancelled", None, "Generation cancelled by user"
            return
            
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
        logger.info(f"Starting chain generation with: action_direction={action_direction}, resolution={resolution}")
        
        # Create session directory for this generation
        session_timestamp = int(time.time())
        session_dir = os.path.join(OUTPUT_DIR, f"story_generation_{session_timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Handle the image based on its type
        if image is None:
            logger.error("No image was provided")
            yield "error", None, "Error: No image was uploaded. Please upload an image before generating."
            return
        
        yield "progress", 0.05, "Processing input image..."
            
        # Save the uploaded image to the session directory
        if isinstance(image, str):
            # If image is already a file path, copy it to session directory
            image_path = os.path.join(session_dir, "input_image.png")
            import shutil
            shutil.copy2(image, image_path)
            logger.info(f"Copied image from {image} to {image_path}")
        else:
            # Handle PIL Image or numpy array
            image_path = os.path.join(session_dir, "input_image.png")
            logger.info(f"Saving image to {image_path}")
            
            # If image is a numpy array (from Gradio upload)
            if isinstance(image, np.ndarray):
                Image.fromarray(image).save(image_path)
            elif isinstance(image, Image.Image):
                image.save(image_path)
            else:
                # Try to save whatever it is as a file
                try:
                    with open(image_path, "wb") as f:
                        f.write(image)
                except Exception as e:
                    logger.exception(f"Could not save image, type: {type(image)}")
                    yield "error", None, f"Error: Could not process the uploaded image: {str(e)}. Please try a different image format."
                    return
        
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            yield "error", None, "Error: The image file could not be found. Please try uploading again."
            return
            
        # Auto-adjust the image if needed (too dark, too bright, or oversaturated)
        yield "progress", 0.07, "Analyzing image quality..."
        adjusted_path, adjustments = video_processing.auto_adjust_image(image_path)
        
        # If adjustments were made, use the adjusted image
        if adjustments:
            logger.info(f"Auto-adjusted image: {', '.join(adjustments)}")
            image_path = adjusted_path
            # Update action direction to indicate adjustment
            if 'action_direction' in image_analysis and image_analysis['action_direction']:
                image_analysis['action_direction'] = f"{image_analysis['action_direction']} (Note: Image was automatically adjusted: {', '.join(adjustments)})"
        
        # Get structured image analysis if any component is missing
        yield "progress", 0.1, "Analyzing image..."
        image_description = ""
        
        if not all([theme, background, main_subject, tone_and_color, action_direction]):
            try:
                logger.info("Some structured analysis components missing, getting complete analysis...")
                yield "progress", 0.12, "Performing detailed image analysis..."
                analysis = openai_client.analyze_image_structured(image_path)
                image_analysis.update({k: v for k, v in analysis.items() if not image_analysis.get(k)})
                
                # Set image description for backward compatibility
                image_description = f"A scene showing {image_analysis['main_subject']} with {image_analysis['background']} in the background."
                
                logger.info(f"Theme: {image_analysis['theme']}")
                logger.info(f"Background: {image_analysis['background']}")
                logger.info(f"Main Subject: {image_analysis['main_subject']}")
                logger.info(f"Tone and Color: {image_analysis['tone_and_color']}")
                logger.info(f"Action Direction: {image_analysis['action_direction']}")
                
                yield "analysis", image_analysis, "Generated structured image analysis"
            except Exception as e:
                error_msg = str(e)
                logger.exception("Error getting structured image analysis")
                yield "error", None, f"Error analyzing image: {error_msg}"
                return
        
        # Generate scene vision if not provided
        yield "progress", 0.15, "Generating scene vision..."
        if not scene_vision or len(scene_vision.strip()) == 0:
            try:
                # Create scene vision using all structured components
                scene_vision_prompt = f"""
                Create a comprehensive scene vision based on the following analysis:
                
                Theme: {image_analysis['theme']}
                Background: {image_analysis['background']}
                Main Subject: {image_analysis['main_subject']}
                Tone and Color: {image_analysis['tone_and_color']}
                Action Direction: {image_analysis['action_direction']}
                
                Focus primarily on the main subject while maintaining the background elements, theme, and visual tone described.
                """
                
                scene_vision = openai_client.generate_scene_vision(scene_vision_prompt, image_analysis['main_subject'])
                logger.info(f"Scene vision: {scene_vision}")
                yield "vision", scene_vision, "Generated scene vision"
            except Exception as e:
                logger.exception("Error generating scene vision")
                yield "error", None, f"Error generating scene vision with OpenAI API: {str(e)}"
                return

        # Auto-determine the number of chains if requested (num_chains = -1)
        if num_chains == -1:
            yield "progress", 0.18, "Auto-determining optimal chain count..."
            try:
                # Use OpenAI to determine the optimal number of chains based on complexity
                num_chains = openai_client.determine_optimal_chain_count(scene_vision, action_direction)
                logger.info(f"Auto-determined optimal chain count: {num_chains}")
                
                # Safety bounds check (1-10)
                num_chains = max(1, min(10, num_chains))
                yield "chains", num_chains, f"Auto-determined optimal chain count: {num_chains}"
            except Exception as e:
                logger.exception("Error determining optimal chain count")
                # Fallback to default
                num_chains = 3
                logger.info(f"Falling back to default chain count: {num_chains}")
                yield "chains", num_chains, f"Falling back to default chain count: {num_chains}"
        
        # Save original image for quality preservation
        original_image_path = image_path
        
        # Initialize lists to store video paths and key frames
        current_image_path = image_path
        
        # Upload initial image to FAL.ai
        yield "progress", 0.2, "Uploading initial image..."
        try:
            current_image_url = fal_client.upload_file(current_image_path)
            logger.info(f"Uploaded image URL: {current_image_url}")
        except Exception as e:
            error_msg = str(e)
            logger.exception("Error uploading image to FAL")
            
            if "API key" in error_msg or "authentication" in error_msg.lower():
                yield "error", None, "Error: Invalid FAL API key. Please check your FAL_API_KEY in the .env file."
                return
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                yield "error", None, "Error: You've exceeded your FAL API quota. Please check your usage and billing information."
                return
            else:
                yield "error", None, f"Error uploading image to FAL.ai: {error_msg}"
                return

        # Generate each video in the chain
        for chain in range(num_chains):
            if cancel_requested():
                logger.info(f"Generation cancelled by user at chain {chain+1}")
                yield "cancelled", None, "Generation cancelled by user"
                # Return what we have so far
                if len(video_paths) > 0:
                    try:
                        final_video_path = os.path.join(session_dir, "partial_story.mp4")
                        logger.info(f"Stitching videos to create partial video: {final_video_path}")
                        video_processing.stitch_videos(video_paths, final_video_path)
                        yield "final", final_video_path, f"Partial story with {len(video_paths)} chains completed before cancellation"
                    except Exception as e:
                        logger.exception("Error stitching partial videos after cancellation")
                        # Return individual videos if stitching failed
                        yield "videos", video_paths, f"Generated {len(video_paths)} chains before cancellation"
                return
            
            # Use 1-based chain numbers in messages for user display
            chain_number = chain + 1
            
            base_progress = 0.2 + (0.7 * chain / num_chains)
            step_size = 0.7 / num_chains
            yield "progress", base_progress, f"Starting chain {chain_number}/{num_chains}..."
            
            try:
                # Get description of the previous video's last frame, if there is one
                frame_desc = ""
                if len(frame_paths) > 0:
                    try:
                        yield "progress", base_progress + step_size * 0.1, "Analyzing previous frame..."
                        frame_desc = openai_client.image_to_text(frame_paths[-1])
                        logger.info(f"Frame description: {frame_desc}")
                    except Exception as e:
                        logger.exception("Error analyzing frame, continuing without frame description")
                        frame_desc = "The previous frame of the scene."
                else:
                    # For the first chain, use the compiled image description
                    frame_desc = image_description or f"A scene showing {image_analysis['main_subject']} with {image_analysis['background']} in the background."
                
                # Determine chain-specific parameters
                chain_specific_vision = scene_vision
                
                # Generate cinematic prompt using LangChain
                try:
                    yield "progress", base_progress + step_size * 0.2, "Generating cinematic prompt with LangChain..."
                    logger.info("=== Starting LangChain reasoning process for prompt generation ===")
                    logger.info(f"Story Phase: {chain_number}/{num_chains} ({int((chain / num_chains) * 100)}% complete)")
                    logger.info("Analyzing scene components for continuity...")
                    
                    cinematic_prompt = generate_cinematic_prompt(
                        action_direction=image_analysis['action_direction'],
                        scene_vision=chain_specific_vision,
                        frame_description=frame_desc,
                        image_description=image_description,
                        theme=image_analysis['theme'],
                        background=image_analysis['background'],
                        main_subject=image_analysis['main_subject'],
                        tone_and_color=image_analysis['tone_and_color'],
                        current_chain=chain,
                        total_chains=num_chains
                    )
                    logger.info(f"Generated cinematic prompt: {cinematic_prompt}")
                except Exception as e:
                    logger.exception("Error using LangChain for prompt generation, falling back")
                    cinematic_prompt = f"Continue the scene with {image_analysis['main_subject']} in the {image_analysis['background']}, showing {image_analysis['action_direction']}."
                
                # Save the prompt for later use
                final_prompt_list.append(cinematic_prompt)
                
                # Determine which model to use
                selected_model = "wan"  # Default
                if model_type == "Pixverse v3.5":
                    selected_model = "pixverse"
                    model_display_name = "Pixverse v3.5"
                    
                    # Check if using 1080p with 8s duration (which is not allowed)
                    if resolution == "1080p" and model_params.get("duration", 5) > 5:
                        logger.warning("1080p videos are limited to 5 seconds in Pixverse. Forcing duration to 5 seconds.")
                        model_params["duration"] = 5
                        yield "progress", base_progress + step_size * 0.28, "Note: 1080p limited to 5 seconds duration, adjusting..."
                elif model_type == "LUMA Ray2":
                    selected_model = "luma"
                    model_display_name = "LUMA Ray2"
                else:
                    model_display_name = "WAN"
                
                # Generate the video with FAL.ai
                yield "progress", base_progress + step_size * 0.3, f"Generating video with {model_display_name} via FAL.ai..."
                
                # Common parameters for video generation
                video_gen_params = {
                    "prompt": cinematic_prompt,
                    "image_url": current_image_url,
                    "resolution": resolution,
                    "seed": seed if seed != -1 else None,
                    "model": selected_model
                }
                
                # Add model-specific parameters
                if selected_model == "pixverse":
                    # Add Pixverse-specific parameters
                    video_gen_params.update({
                        "safety_checker": safety_checker,  # Used differently in Pixverse
                        "duration": model_params.get("duration", 5),
                        "negative_prompt": model_params.get("negative_prompt", "")
                    })
                    
                    # Add style parameter if specified
                    if model_params.get("style"):
                        video_gen_params["style"] = model_params["style"]
                elif selected_model == "luma":
                    # Add LUMA-specific parameters
                    video_gen_params.update({
                        "duration": 5,  # Always force 5 seconds for LUMA Ray2
                        "aspect_ratio": model_params.get("aspect_ratio", "16:9"),
                        "loop": False   # Always disabled - causes workflow issues
                    })
                    
                    # Use end image functionality is now disabled as it causes workflow issues
                    # This ensures each video is generated independently
                else:
                    # Add WAN-specific parameters
                    video_gen_params.update({
                        "num_frames": 81,  # Default num frames
                        "fps": 16,  # Default FPS
                        "inference_steps": inference_steps,
                        "safety_checker": safety_checker,
                        "prompt_expansion": prompt_expansion
                    })
                
                # Call the video generation function
                video_url = fal_client.generate_video_from_image(**video_gen_params)
                
                # Download the video
                yield "progress", base_progress + step_size * 0.7, f"Downloading video for chain {chain_number}..."
                # Using chain_number (1-indexed) in the filename instead of chain (0-indexed)
                video_path = os.path.join(session_dir, f"chain_{chain_number:02d}.mp4")
                fal_client.download_video(video_url, video_path)
                # Don't append to video_paths yet - we'll append the trimmed version if applicable
                
                # If not the last chain, extract best frame and trim video
                if chain < num_chains - 1:
                    yield "progress", base_progress + step_size * 0.8, "Finding highest quality frame for continuity..."
                    logger.info(f"Finding highest quality frame for chain {chain_number}")
                    
                    # Extract best frame from last 10 frames and trim video to end at this frame
                    best_frame_path, trimmed_video_path = video_processing.extract_and_trim_best_frame(
                        video_path,
                        os.path.join(session_dir, f"chain_{chain_number:02d}_processed")
                    )
                    
                    # Use the trimmed video if available
                    if trimmed_video_path != video_path and os.path.exists(trimmed_video_path):
                        logger.info(f"Video trimmed to end at highest quality frame: {os.path.basename(trimmed_video_path)}")
                        # Use trimmed video path instead of original
                        video_path = trimmed_video_path
                        yield "progress", base_progress + step_size * 0.82, "Video trimmed to end at optimal frame for smoother transitions..."
                        
                        # Add a small pause to ensure file system operations are complete
                        time.sleep(0.5)
                    
                    # Get structured analysis of the best frame
                    # This maintains the Theme and Main Subject while potentially updating Background and Tone/Color
                    try:
                        yield "progress", base_progress + step_size * 0.9, "Analyzing frame for continuity..."
                        logger.info("Getting structured analysis of the best frame...")
                        
                        # First, get a text description of the frame for LangChain
                        frame_desc = openai_client.image_to_text(best_frame_path)
                        frame_paths.append(best_frame_path)
                        logger.info(f"Got description for best frame: {frame_desc[:50]}...")
                        
                        # Now get structured analysis for continuity
                        frame_analysis = openai_client.analyze_image_structured(best_frame_path)
                        
                        # Update only certain components while maintaining others for continuity
                        image_analysis['background'] = frame_analysis['background']
                        image_analysis['tone_and_color'] = frame_analysis['tone_and_color']
                        
                        # Theme and Main Subject should remain consistent throughout
                        # Action Direction will be updated based on narrative progression
                        
                        logger.info(f"Updated Background: {image_analysis['background']}")
                        logger.info(f"Updated Tone and Color: {image_analysis['tone_and_color']}")
                        
                    except Exception as e:
                        logger.exception(f"Error getting frame analysis")
                        # Keep existing image analysis if this fails
                    
                    # Update action direction based on narrative progression
                    try:
                        yield "progress", base_progress + step_size * 0.95, "Updating action direction for next chain..."
                        logger.info("Updating action direction for narrative progression...")
                        progression_prompt = f"""
                        Based on the current story progression ({chain_number}/{num_chains}), 
                        suggest the next logical action direction that follows from: 
                        "{image_analysis['action_direction']}"
                        
                        Maintain the core theme: "{image_analysis['theme']}"
                        The main subject remains: "{image_analysis['main_subject']}"
                        
                        Provide ONLY a concise action direction (1-2 sentences) that progresses the narrative
                        """
                        
                        image_analysis['action_direction'] = openai_client.generate_scene_vision(
                            progression_prompt, 
                            image_analysis['main_subject']
                        )
                        
                        logger.info(f"Updated Action Direction: {image_analysis['action_direction']}")
                        
                    except Exception as e:
                        logger.exception(f"Error updating action direction")
                        # Keep existing action direction if this fails
                    
                    # Upload the best frame for the next video
                    yield "progress", base_progress + step_size * 0.98, "Uploading best frame for next chain..."
                    logger.info(f"Uploading best frame for next chain")
                    try:
                        current_image_url = fal_client.upload_file(best_frame_path)
                    except Exception as e:
                        logger.exception(f"Error uploading frame, trying alternative method")
                        try:
                            current_image_url = fal_client.upload_file_alternative(best_frame_path)
                        except Exception:
                            logger.exception(f"Both upload methods failed")
                            raise e
                    
                # Now append the final video path (original or trimmed) to video_paths
                video_paths.append(video_path)
                
                # Yield the complete chain with the formatted chain number
                yield "chain_complete", video_path, f"Chain {chain_number} completed"
                
            except Exception as e:
                logger.exception(f"Error in chain {chain_number}")
                if chain > 0:
                    # Return what we have so far if at least one chain was successful
                    yield "error", None, f"Completed {chain} chains. Error in chain {chain_number}: {str(e)}"
                    return
                else:
                    yield "error", None, f"Error in first chain: {str(e)}"
                    return
        
        # Stitch videos together
        yield "progress", 0.95, "Stitching videos together..."
        try:
            final_video_path = os.path.join(session_dir, "final_story.mp4")
            logger.info(f"Stitching videos to create final video: {final_video_path}")
            video_processing.stitch_videos(video_paths, final_video_path)
            logger.info(f"Completed story generation")
            yield "final", final_video_path, "Story generation completed successfully!"
        except Exception as e:
            logger.exception("Error stitching videos")
            # Return the individual videos even if stitching failed
            yield "videos", video_paths, f"Videos generated but could not be stitched together: {str(e)}"
    
    except Exception as e:
        logger.exception(f"Error in chain generation: {str(e)}")
        yield "error", None, f"An unexpected error occurred: {str(e)}"

def create_ui():
    # ... existing UI creation code ...
    
    with gr.Blocks(title="AI Scene Maker", css=css, theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            """
            # AI Scene Maker
            Upload an image and specify an action direction to generate a sequence of videos showing that action.
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
                                choices=["WAN (Default)", "Pixverse v3.5", "LUMA Ray2"], 
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
                                value="blurry, low quality, low resolution",
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
                        
                        with gr.Row():
                            generate_btn = gr.Button("Generate", variant="primary")
                            cancel_btn = gr.Button("Cancel")
                        
                        generation_status = gr.Markdown("Upload an image and click Generate to start")
                    
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

        # Connect the image upload event
        image_upload.change(
            fn=on_image_upload,
            inputs=[image_upload],
            outputs=[theme, background, main_subject, tone_and_color, action_direction, scene_vision, image_adjustment_info]
        )
        
        # Toggle model-specific options based on model selection
        def update_model_options(model_choice):
            if model_choice == "Pixverse v3.5":
                return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=True), gr.update(visible=False))
            elif model_choice == "LUMA Ray2":
                return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=False), gr.update(visible=False))
            else:  # WAN (Default)
                return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))
                
        model_type.change(
            fn=update_model_options,
            inputs=[model_type],
            outputs=[wan_resolution, pixverse_resolution, luma_resolution, pixverse_options, luma_options]
        )

        # Connect auto-chain toggle to disable/enable chain slider
        auto_determine.change(
            fn=toggle_chains,
            inputs=[auto_determine],
            outputs=[num_chains]
        )

        # Function to get the appropriate resolution based on model selection
        def get_resolution(model_choice, wan_res, pixverse_res, luma_res):
            if model_choice == "Pixverse v3.5":
                return pixverse_res
            elif model_choice == "LUMA Ray2":
                return luma_res
            else:
                return wan_res

        # Start generation when button clicked
        generate_btn.click(
            fn=ui_start_chain_generation,
            inputs=[
                action_direction, image_upload, 
                theme, background, main_subject, tone_and_color, scene_vision,
                # We need to select the appropriate resolution based on model
                # This will be handled inside the function
                model_type,  # We'll check this first to determine which resolution to use
                wan_resolution, pixverse_resolution, luma_resolution,
                inference_steps, safety_checker, 
                prompt_expansion, auto_determine, num_chains, 
                seed, pixverse_duration, pixverse_style, pixverse_negative_prompt,
                luma_duration, luma_aspect_ratio,
                generation_running, cancel_requested
            ],
            outputs=[
                generation_running, chain_gallery, output_videos, 
                video_paths, final_video_path, generation_status, 
                output_status, download_link, progress_group,
                overall_progress, chain_progress, current_operation, eta_display
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
                final_video_path
            ]
        )

    return iface

def toggle_chains(auto):
    """Toggle chains slider interactivity based on auto-determine checkbox"""
    return gr.update(interactive=not auto)

def ui_start_chain_generation(action_dir, img, theme, background, main_subject, tone_color, vision, 
                           model_selection, wan_res, pixverse_res, luma_res, steps, safety, expansion, 
                           auto_determine, chains, seed_val, pixverse_duration="5", 
                           pixverse_style="None", pixverse_negative_prompt="", 
                           luma_duration="5", luma_aspect_ratio="16:9",
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
    else:
        resolution = wan_res
    
    # Show progress group and initial status
    yield (
        gen_running, None, None, video_paths, final_path, 
        "Starting chain generation...", "", gr.update(visible=False),
        gr.update(visible=True), 0, 0, 
        "Initializing...", "Estimated time remaining: Calculating..."
    )
    
    try:
        # If auto-determine is enabled, set chains to -1 to indicate auto mode
        if auto_determine:
            chains = -1
            
        # Prepare model-specific parameters
        if model_selection == "Pixverse v3.5":
            # Pixverse-specific parameters
            model_params = {
                "duration": int(pixverse_duration),
                "style": None if pixverse_style == "None" else pixverse_style,
                "negative_prompt": pixverse_negative_prompt,
                "aspect_ratio": "16:9"  # Default for Pixverse
            }
        elif model_selection == "LUMA Ray2":
            # LUMA-specific parameters
            model_params = {
                "duration": 5,  # Always force 5 seconds for LUMA Ray2
                "aspect_ratio": luma_aspect_ratio,
                "loop": False   # Always disabled - causes workflow issues
            }
        else:
            # WAN parameters (empty dict as defaults are used)
            model_params = {}
            
        # Log the user-edited inputs to ensure we're using the most current values
        logger.info(f"Generation with user-edited fields: theme={theme}, background={background}, main_subject={main_subject}, tone_color={tone_color}, action_dir={action_dir}")
            
        # Run the chain generation with all structured components
        # Using the current field values from the UI inputs rather than any stored values
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
            cancel_requested=lambda: cancel_req
        )
        
        # Process chain generation results
        chain_videos = {}  # Dictionary to store {chain_number: video_path}
        completed_chains = 0
        determined_chains = chains if chains > 0 else 3  # Default assumption
        
        for update_type, result, message in chain_generator:
            if update_type == "progress":
                # Progress update - parse percentage from message if possible
                try:
                    chain_percent = float(result) * 100 if isinstance(result, (int, float)) else 0
                except (ValueError, TypeError):
                    chain_percent = 0
                
                # Calculate overall progress
                if chains > 0:
                    overall_percent = (completed_chains / chains) * 100
                    if chain_percent > 0:
                        overall_percent += (chain_percent / 100) * (100 / chains)
                else:
                    overall_percent = (completed_chains / determined_chains) * 100
                    
                # Calculate ETA
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
                    f"Current operation: {message}", eta_text
                )
            
            elif update_type == "chains":
                # Number of chains was determined
                determined_chains = result
                
                yield (
                    gen_running, None, None, video_paths, final_path, 
                    f"Number of chains determined: {result}", "", gr.update(visible=False),
                    gr.update(visible=True), min(10, (completed_chains / determined_chains) * 100), 0, 
                    "Preparing for video generation...", "Estimated time remaining: Calculating..."
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
                    f"Completed chain {chain_number}/{determined_chains} in {format_time(chain_elapsed)}", eta_text
                )
            
            elif update_type == "final":
                # Final video completed
                final_path = result
                overall_percent = 100
                
                yield (
                    gen_running, gallery_items, final_path, video_paths, final_path, 
                    "Story generation completed successfully!", "", gr.update(visible=True),
                    gr.update(visible=True), 100, 100, 
                    "Generation complete!", f"Total time: {format_time(time.time() - start_time)}"
                )
                
            elif update_type == "error":
                # Error occurred
                yield (
                    False, gallery_items if 'gallery_items' in locals() else None, None, 
                    video_paths, final_path, f"Error: {message}", "", gr.update(visible=False),
                    gr.update(visible=False), 0, 0, "", ""
                )
                return
                
            elif update_type == "cancelled":
                # Generation was cancelled
                total_time = time.time() - start_time
                yield (
                    False, gallery_items if 'gallery_items' in locals() else None, None, 
                    video_paths, final_path, "Generation cancelled by user", "", gr.update(visible=False),
                    gr.update(visible=False), 0, 0, 
                    f"Generation cancelled after {format_time(total_time)}", ""
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
                "Generation complete!", f"Total time: {format_time(total_time)}"
            )
        else:
            yield (
                gen_running, gallery_items if 'gallery_items' in locals() else None, None, 
                video_paths, final_path, "Generation completed but no final video was produced", "", gr.update(visible=False),
                gr.update(visible=False), 0, 0, "", ""
            )
            
    except Exception as e:
        logger.exception(f"Error in chain generation: {str(e)}")
        yield (
            False, None, None, video_paths, final_path, 
            f"An unexpected error occurred: {str(e)}", "", gr.update(visible=False),
            gr.update(visible=False), 0, 0, "", ""
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
        None   # final_video_path
    ]