import fal_client
import requests
import os
import logging
import time
import base64
import json
import config

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a custom filter to completely suppress connection logs
class HttpxConnectionFilter(logging.Filter):
    """Filter to completely suppress httpx connection logs"""
    def filter(self, record):
        # Return False for any logs about HTTP requests to queue.fal.run
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if 'HTTP Request:' in record.msg and 'queue.fal.run' in record.msg:
                return False
        return True

# Apply filter to root logger and httpx logger
logging.getLogger().addFilter(HttpxConnectionFilter())
httpx_logger = logging.getLogger("httpx")
httpx_logger.addFilter(HttpxConnectionFilter())

# Set the API key as an environment variable
os.environ["FAL_KEY"] = config.FAL_API_KEY

# Set log levels for HTTP libraries
logging.getLogger("httpx").setLevel(logging.ERROR)  # Only show errors, not warnings
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

def generate_video_from_image(prompt, image_url, resolution, num_frames=81, fps=16, seed=None, 
                         inference_steps=40, safety_checker=False, prompt_expansion=True, model="wan",
                         duration=5, style=None, negative_prompt="", aspect_ratio="16:9", loop=False, 
                         end_image_url=None):
    """
    Generate a video from an image using FAL.ai's text-to-video models
    
    Args:
        prompt: Text prompt for the video
        image_url: URL of the image to use as a starting point
        resolution: Resolution of the video, e.g. "576p" or "720p"
        num_frames: Number of frames in the video (for WAN model)
        fps: Frames per second (for WAN model)
        seed: Random seed for reproducibility (or None for random)
        inference_steps: Number of inference steps (higher = more quality, slower)
        safety_checker: Whether to enable safety checker
        prompt_expansion: Whether to enable prompt expansion
        model: Which model to use ("wan", "pixverse", or "luma")
        duration: Duration in seconds (5 or 8 for Pixverse, "5s" or "9s" for LUMA)
        style: Visual style for Pixverse (anime, 3d_animation, clay, comic, cyberpunk, or None)
        negative_prompt: Negative prompt for Pixverse to specify what to avoid
        aspect_ratio: Aspect ratio for the video (Pixverse and LUMA)
        loop: Whether the video should loop (LUMA only)
        end_image_url: URL of the image to end the video with (LUMA only)
        
    Returns:
        URL of the generated video
    """
    
    # Use the LUMA Ray2 model
    if model == "luma":
        logger.info(f"Using LUMA Ray2 model with prompt: {prompt}")
        
        # LUMA Ray2 only supports 5 seconds duration
        duration = 5
        
        # Convert duration from int to string format for LUMA
        luma_duration = f"{duration}s"
        
        # Prepare request parameters for LUMA
        request_params = {
            "prompt": prompt,
            "image_url": image_url,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": luma_duration,
            "loop": loop
        }
        
        # Add end image if provided
        if end_image_url:
            request_params["end_image_url"] = end_image_url
            
        # Log the parameters for debugging
        logger.info(f"LUMA parameters: {request_params}")
        
        # Temporarily disable httpx logging during video generation
        httpx_logger = logging.getLogger("httpx")
        original_level = httpx_logger.level
        httpx_logger.setLevel(logging.ERROR)
        
        try:
            logger.info(f"Sending LUMA Ray2 image-to-video request")
            
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        logger.info(f"LUMA progress: {log['message']}")

            # Use the LUMA model endpoint
            result = fal_client.subscribe(
                "fal-ai/luma-dream-machine/ray-2/image-to-video",
                arguments=request_params,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            
            video_url = result["video"]["url"]
            logger.info(f"LUMA video generated successfully: {video_url}")
            return video_url
            
        except Exception as e:
            logger.exception(f"Error generating video with LUMA: {str(e)}")
            raise
        finally:
            # Restore original logging level
            httpx_logger.setLevel(original_level)
            
    # Use the Pixverse v3.5 model
    elif model == "pixverse":
        logger.info(f"Using Pixverse v3.5 model with prompt: {prompt}")
        
        # Prepare request parameters for Pixverse
        request_params = {
            "prompt": prompt,
            "image_url": image_url,
            "resolution": resolution,
            "aspect_ratio": "16:9",  # Default aspect ratio
            "duration": duration,    # Duration in seconds (5 or 8)
        }
        
        # Add optional parameters
        if seed is not None:
            request_params["seed"] = seed
            
        # Add negative prompt if provided or if safety checker is enabled
        if negative_prompt:
            request_params["negative_prompt"] = negative_prompt
        elif safety_checker:
            request_params["negative_prompt"] = "blurry, low quality, low resolution, pixelated, noisy, grainy, out of focus, poorly lit, poorly exposed, poorly composed, poorly framed, poorly cropped, poorly color corrected, poorly color graded"
        else:
            request_params["negative_prompt"] = ""
            
        # Add style if provided
        if style:
            request_params["style"] = style
        
        # Ensure 1080p videos are limited to 5 seconds
        if resolution == "1080p" and duration > 5:
            logger.warning("1080p videos are limited to 5 seconds in Pixverse. Adjusting duration.")
            request_params["duration"] = 5
            
        # Log the parameters for debugging
        logger.info(f"Pixverse parameters: {request_params}")
        
        # Temporarily disable httpx logging during video generation
        httpx_logger = logging.getLogger("httpx")
        original_level = httpx_logger.level
        httpx_logger.setLevel(logging.ERROR)
        
        try:
            logger.info(f"Sending Pixverse image-to-video request")
            
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        logger.info(f"Pixverse progress: {log['message']}")

            # Use the Pixverse model endpoint
            result = fal_client.subscribe(
                "fal-ai/pixverse/v3.5/image-to-video",
                arguments=request_params,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            
            video_url = result["video"]["url"]
            logger.info(f"Pixverse video generated successfully: {video_url}")
            return video_url
            
        except Exception as e:
            logger.exception(f"Error generating video with Pixverse: {str(e)}")
            raise
        finally:
            # Restore original logging level
            httpx_logger.setLevel(original_level)
            
    else:  # Default to WAN model
        # Parse resolution to get width and height
        if resolution == "360p":
            width, height = 640, 360
        elif resolution == "540p":
            width, height = 960, 540
        elif resolution == "576p":
            width, height = 1024, 576
        elif resolution == "720p":
            width, height = 1280, 720
        elif resolution == "1080p":
            width, height = 1920, 1080
        else:
            logger.warning(f"Unknown resolution: {resolution}, defaulting to 720p")
            width, height = 1280, 720
        
        # Determine model based on parameters
        model = "fal-ai/wan-i2v"  # Default model with correct namespace
        
        # Prepare FAL.ai request parameters
        request_params = {
            "prompt": prompt,
            "image_url": image_url,
            "num_frames": num_frames,
            "frames_per_second": fps,  # Correct param name
            "width": width,
            "height": height,
            "sampler": "euler_a",
            "guidance_scale": 10.0,
        }
        
        # Add optional parameters
        if seed is not None:
            request_params["seed"] = seed
        
        if inference_steps:
            request_params["num_inference_steps"] = inference_steps
            
        if not safety_checker:
            request_params["enable_safety_checker"] = False  # Correct param name
            
        if not prompt_expansion:
            request_params["enable_prompt_expansion"] = False  # Correct param name
        
        # Temporarily disable httpx logging during video generation
        httpx_logger = logging.getLogger("httpx")
        original_level = httpx_logger.level
        httpx_logger.setLevel(logging.ERROR)
        
        try:
            # Use the fal_client for video generation
            logger.info("Starting video generation with FAL.ai")
            start_time = time.time()
            
            # We'll use two possible approaches depending on what's available in the API
            try:
                # First approach: Try using the run synchronous method 
                # which directly returns the result (newer versions)
                logger.info("Using synchronous FAL.ai call")
                result = fal_client.run(model, arguments=request_params)
                
                # If we got here, result is a dictionary
                logger.info(f"Video generation completed in {int(time.time() - start_time)} seconds")
                
                # Handle both possible response formats
                if isinstance(result, dict):
                    # Try to extract video URL from the result dict
                    if "video" in result and "url" in result["video"]:
                        return result["video"]["url"]
                    elif "video_url" in result:
                        return result["video_url"]
                    else:
                        logger.error(f"Unexpected result format: {result.keys()}")
                        raise Exception(f"Unexpected result format: {result.keys()}")
                else:
                    logger.error(f"Unexpected result type: {type(result)}")
                    raise Exception(f"Unexpected result type: {type(result)}")
                
            except (AttributeError, TypeError) as e:
                # Second approach: If run is not available or doesn't work as expected,
                # fall back to submit and manual polling
                logger.info(f"Synchronous call failed ({str(e)}), falling back to submit/polling method")
                
                # Use the submit method and poll for the result
                queue_handler = fal_client.submit(model, arguments=request_params)
                request_id = queue_handler.request_id
                
                # Poll for the result with adaptive polling intervals
                max_attempts = 120  # 2 minutes with 1s polling
                last_log_time = start_time
                
                for attempt in range(max_attempts):
                    # Only log progress occasionally to reduce noise
                    current_time = time.time()
                    if current_time - last_log_time > 5:  # Log every 5 seconds
                        elapsed = int(current_time - start_time)
                        logger.info(f"Video generation in progress... (elapsed: {elapsed}s)")
                        last_log_time = current_time
                    
                    # Check status
                    status = fal_client.status(model, request_id)
                    
                    if status.status == "COMPLETED":
                        logger.info(f"Video generation completed in {int(time.time() - start_time)} seconds")
                        result = fal_client.result(model, request_id)
                        
                        # Extract video URL from result
                        if "video" in result and "url" in result["video"]:
                            return result["video"]["url"]
                        elif "video_url" in result:
                            return result["video_url"]
                        else:
                            logger.error(f"Unexpected result format: {result.keys()}")
                            raise Exception(f"Unexpected result format: {result.keys()}")
                    
                    elif status.status == "FAILED":
                        error = "Unknown error"
                        if hasattr(status, "logs") and status.logs:
                            error = status.logs[-1]
                        logger.error(f"Video generation failed: {error}")
                        raise Exception(f"Video generation failed: {error}")
                    
                    # Sleep between polling
                    time.sleep(1.0)
                
                # If we get here, polling timed out
                raise Exception("Video generation timed out")
                
        except Exception as e:
            logger.exception(f"Error generating video: {str(e)}")
            raise
        finally:
            # Restore original logging level
            httpx_logger.setLevel(original_level)

def generate_video_from_text(prompt, resolution, aspect_ratio, inference_steps, safety_checker, prompt_expansion, seed=None):
    """
    Generate a video from text using fal.ai WAN-T2V model with robust error handling and retries.
    
    Args:
        prompt: Text prompt describing the video
        resolution: Video resolution (e.g., "480p", "720p")
        aspect_ratio: Aspect ratio of the video
        inference_steps: Number of inference steps
        safety_checker: Whether to enable safety checker
        prompt_expansion: Whether to enable prompt expansion
        seed: Random seed for reproducibility
        
    Returns:
        URL of the generated video
    """
    logger.info(f"Generating video from text with prompt: {prompt[:50]}...")
    try:
        # Create the arguments dictionary
        arguments = {
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": inference_steps,
            "enable_safety_checker": safety_checker,
            "enable_prompt_expansion": prompt_expansion
        }
        
        # Add seed if provided
        if seed is not None:
            arguments["seed"] = seed
        
        logger.info(f"Submitting request to fal.ai WAN-T2V model")
        
        # Submit the request to the queue
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Instead of submitting and then polling manually, use the subscribe method
                # which handles polling internally and returns once complete
                try:
                    # First approach: Use the subscribe method which handles polling more efficiently
                    logger.info("Using FAL.ai subscribe method for more efficient processing")
                    result = fal_client.subscribe(
                        "fal-ai/wan-t2v",
                        arguments=arguments,
                        with_logs=True,
                        on_queue_update=lambda status: logger.debug(f"Job status: {getattr(status, 'status', 'unknown')}")
                    )
                    logger.info(f"Request completed. Result received.")
                    return result["video"]["url"]
                except AttributeError:
                    # Fallback to manual polling with adaptive intervals if subscribe not available
                    handler = fal_client.submit(
                        "fal-ai/wan-t2v",
                        arguments=arguments
                    )
                    
                    request_id = handler.request_id
                    logger.info(f"Request submitted with ID: {request_id}")
                    
                    # Poll for the result with adaptive polling intervals
                    max_poll_attempts = 60
                    
                    # Start with short polling intervals, then increase gradually
                    # This reduces API calls while still being responsive at the start
                    base_interval = 5
                    
                    for poll_attempt in range(max_poll_attempts):
                        # Calculate dynamic polling interval - starts at 5s, gradually increases
                        # up to a maximum of 30 seconds for later attempts
                        if poll_attempt < 3:
                            poll_interval = base_interval  # 5s for first 3 attempts
                        elif poll_attempt < 6:
                            poll_interval = base_interval * 2  # 10s for next 3 attempts
                        elif poll_attempt < 10:
                            poll_interval = base_interval * 3  # 15s for next 4 attempts
                        else:
                            poll_interval = min(base_interval * 4, 30)  # 20s, max 30s for remaining attempts
                        
                        # Use debug level for polling status to reduce console output
                        logger.debug(f"Polling for result, attempt {poll_attempt+1}/{max_poll_attempts}")
                        
                        # Check status
                        status = fal_client.status("fal-ai/wan-t2v", request_id, with_logs=True)
                        
                        if hasattr(status, 'status'):
                            if status.status == "COMPLETED":
                                logger.info(f"Request completed. Fetching result.")
                                result = fal_client.result("fal-ai/wan-t2v", request_id)
                                return result["video"]["url"]
                            elif status.status == "FAILED":
                                error_message = "Unknown error"
                                if hasattr(status, 'error'):
                                    error_message = status.error
                                logger.error(f"Video generation failed: {error_message}")
                                break  # Break the polling loop and try again
                        
                        # Sleep before polling again
                        time.sleep(poll_interval)
                
                # If we reach here without returning, the polling timed out or failed
                logger.warning(f"Request {request_id} did not complete in time or failed. Attempt {attempt+1}/{max_retries}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying request... (attempt {attempt+2}/{max_retries})")
                    time.sleep(5)  # Wait before retrying
                else:
                    logger.error(f"All retry attempts failed")
                    raise Exception("Maximum retry attempts reached without success")
            
            except Exception as e:
                logger.error(f"Error during request attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying request... (attempt {attempt+2}/{max_retries})")
                    time.sleep(5)  # Wait before retrying
                else:
                    logger.error(f"All retry attempts failed")
                    raise
    
    except Exception as e:
        logger.exception(f"Error generating video from text: {str(e)}")
        raise Exception(f"Failed to generate video: {str(e)}")

def encode_image_to_base64(image_path):
    """Encode an image to base64 for sending to APIs."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def upload_file(file_path):
    """Upload a file to FAL.ai and return the URL"""
    logger.info(f"Uploading file: {file_path}")
    
    try:
        # Get the file extension and determine content type
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Map extensions to MIME types
        content_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        
        # Default to png if extension not found
        content_type = content_type_map.get(ext, 'image/png')
        
        # Read the file content first
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Upload using the content directly
        result = fal_client.upload(file_content, content_type=content_type)
        
        logger.info(f"File uploaded successfully using fal_client.upload")
        return result
    except Exception as e:
        logger.exception(f"Error uploading file: {str(e)}")
        # Try the alternative method if this fails
        return upload_file_alternative(file_path)

def upload_file_alternative(file_path):
    """
    Alternative method to upload a file to FAL.ai when the primary method fails.
    Reverts to the original REST API approach which might be more stable.
    """
    logger.info(f"Using alternative upload method for file: {file_path}")
    
    # Revert to original REST API method for reliability
    with open(file_path, "rb") as f:
        file_content = f.read()
        
    # Build a multipart/form-data request
    import uuid
    boundary = str(uuid.uuid4())
    content_type = f'multipart/form-data; boundary={boundary}'
    
    # Create the multipart body
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(file_path)}"\r\n'
        f'Content-Type: image/png\r\n\r\n'
    ).encode()
    body += file_content
    body += f'\r\n--{boundary}--\r\n'.encode()
    
    # Set headers with the API key
    headers = {
        "Authorization": f"Key {config.FAL_API_KEY}",
        "Content-Type": content_type
    }
    
    # Upload the file
    try:
        response = requests.post(
            "https://gateway.fal.ai/storage/upload",  # Updated URL
            headers=headers,
            data=body
        )
        response.raise_for_status()
        data = response.json()
        file_url = data.get("url")
        logger.info(f"File uploaded successfully via alternative method, URL: {file_url}")
        return file_url
    except Exception as e:
        logger.exception(f"Error in alternative upload method: {str(e)}")
        raise

def download_file(url, path):
    """Download a file from a URL to the specified path"""
    logger.info(f"Downloading file from {url} to {path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Download complete: {path}")
        return path
    except Exception as e:
        logger.exception(f"Error downloading file: {str(e)}")
        raise

def download_video(url, path):
    """Download a video from a URL to the specified path"""
    logger.info(f"Downloading video from {url}")
    return download_file(url, path)