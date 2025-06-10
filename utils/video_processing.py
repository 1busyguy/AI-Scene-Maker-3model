import cv2
import os
import logging
import subprocess
from moviepy.editor import concatenate_videoclips, VideoFileClip
import numpy as np
from PIL import Image, ImageEnhance
import hashlib

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_ffmpeg_available():
    """Check if ffmpeg and ffprobe are available in the system."""
    ffmpeg_available = False
    ffprobe_available = False
    
    try:
        # Check ffmpeg
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        ffmpeg_available = True
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError):
        pass
    
    try:
        # Check ffprobe (needed for Gradio video validation)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        ffprobe_available = True
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError):
        pass
    
    # If neither found, try checking Windows specific paths
    if not ffmpeg_available or not ffprobe_available:
        try:
            import platform
            if platform.system() == "Windows":
                # Check current directory
                if os.path.exists("ffmpeg.exe"):
                    ffmpeg_available = True
                if os.path.exists("ffprobe.exe"):
                    ffprobe_available = True
                
                # Check system PATH
                for path in os.environ["PATH"].split(os.pathsep):
                    if not ffmpeg_available and os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        ffmpeg_available = True
                    if not ffprobe_available and os.path.exists(os.path.join(path, "ffprobe.exe")):
                        ffprobe_available = True
                        
        except Exception as e:
            logger.debug(f"Error checking for ffmpeg/ffprobe: {str(e)}")
    
    # Log findings
    if ffmpeg_available and ffprobe_available:
        logger.info("FFmpeg and ffprobe found and available.")
        return True
    elif ffmpeg_available and not ffprobe_available:
        logger.warning("FFmpeg found but ffprobe missing. Videos may not display properly in UI.")
        return False
    elif not ffmpeg_available and ffprobe_available:
        logger.warning("ffprobe found but FFmpeg missing. Video processing will be limited.")
        return False
    else:
        logger.warning("Neither FFmpeg nor ffprobe found. Video features will be limited.")
        return False

def is_ffprobe_available():
    """Check specifically if ffprobe is available (needed for Gradio)."""
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        return True
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError):
        # On Windows, check current directory and PATH
        try:
            import platform
            if platform.system() == "Windows":
                if os.path.exists("ffprobe.exe"):
                    return True
                
                for path in os.environ["PATH"].split(os.pathsep):
                    if os.path.exists(os.path.join(path, "ffprobe.exe")):
                        return True
        except Exception:
            pass
        return False

FFMPEG_AVAILABLE = is_ffmpeg_available()
FFPROBE_AVAILABLE = is_ffprobe_available()

if FFMPEG_AVAILABLE:
    logger.info("FFmpeg found and available for video processing.")
else:
    logger.warning("ffmpeg not found. Video concatenation and trimming may not work properly.")

if not FFPROBE_AVAILABLE:
    logger.warning("ffprobe not found. Videos may not display properly in Gradio UI.")
    logger.info("To fix this, install FFmpeg with: https://ffmpeg.org/download.html")

def ensure_video_compatibility(video_path, force_convert=False):
    """
    Ensure video is compatible with Gradio display.
    
    Args:
        video_path: Path to video file
        force_convert: Force conversion even if ffprobe is available
        
    Returns:
        Path to compatible video (may be the same as input)
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return video_path
    
    # If ffprobe is available and we're not forcing conversion, assume video is OK
    if FFPROBE_AVAILABLE and not force_convert:
        return video_path
    
    # Try creating compatible version using OpenCV first
    try:
        output_path = video_path.replace('.mp4', '_compatible.mp4')
        
        # Read video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video with OpenCV: {video_path}")
            raise Exception("OpenCV failed to open video")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer with compatible settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error("Could not create compatible video writer with OpenCV")
            cap.release()
            raise Exception("OpenCV VideoWriter failed")
        
        # Copy frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if frame_count > 0 and os.path.exists(output_path):
            logger.info(f"Created compatible video using OpenCV: {output_path} ({frame_count} frames)")
            return output_path
        else:
            logger.warning("OpenCV conversion failed, trying moviepy...")
            raise Exception("OpenCV conversion produced no output")
            
    except Exception as opencv_error:
        logger.debug(f"OpenCV method failed: {opencv_error}")
        
        # Fallback to moviepy
        try:
            from moviepy.editor import VideoFileClip
            
            output_path = video_path.replace('.mp4', '_compatible.mp4')
            logger.info("Trying moviepy for video compatibility conversion...")
            
            clip = VideoFileClip(video_path)
            clip.write_videofile(
                output_path, 
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            clip.close()
            
            if os.path.exists(output_path):
                logger.info(f"Created compatible video using moviepy: {output_path}")
                return output_path
            else:
                logger.warning("Moviepy conversion failed")
                return video_path
                
        except Exception as moviepy_error:
            logger.debug(f"Moviepy method also failed: {moviepy_error}")
            logger.warning("Both OpenCV and moviepy failed to create compatible video")
            
            # Log helpful error message
            logger.error("=" * 60)
            logger.error("VIDEO DISPLAY ISSUE DETECTED!")
            logger.error("Videos may not display in the UI due to missing ffprobe.")
            logger.error("")
            logger.error("To fix this issue:")
            logger.error("1. Install FFmpeg from: https://ffmpeg.org/download.html")
            logger.error("2. Make sure ffmpeg.exe and ffprobe.exe are in your PATH")
            logger.error("3. Or place ffmpeg.exe and ffprobe.exe in your project directory")
            logger.error("")
            logger.error("Your videos are still being generated successfully,")
            logger.error("they just may not display properly in the web interface.")
            logger.error("=" * 60)
            
            return video_path

def extract_best_frame(video_path, output_path=None):
    """
    Extract the best frame from a video based on sharpness and contrast.
    
    Args:
        video_path: Path to the video file
        output_path: Where to save the extracted frame (optional)
        
    Returns:
        Path to the extracted frame
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    logger.info(f"Extracting best frame from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # We'll extract frames from the latter part of the video (not just the last frame)
    # This is because the last frame might be a fade-out
    frames = []
    scores = []
    
    # Calculate start frame (we'll check the last 30% of the video)
    start_frame = max(0, int(frame_count * 0.7))
    
    # Skip to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames and calculate scores
    frame_interval = max(1, int((frame_count - start_frame) / 10))  # Extract about 10 frames
    for i in range(start_frame, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Calculate score based on sharpness and contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
        _, std_dev = cv2.meanStdDev(gray)  # Contrast
        
        score = laplacian_var * std_dev[0][0]
        
        frames.append(frame)
        scores.append(score)
    
    # If no frames were extracted, try the last frame
    if not frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame from video: {video_path}")
        best_frame = frame
    else:
        # Get the frame with the highest score
        best_frame = frames[scores.index(max(scores))]
    
    cap.release()
    
    # Save the best frame
    if output_path is None:
        output_path = video_path.replace(".mp4", "_best_frame.jpg")
    
    cv2.imwrite(output_path, best_frame)
    return output_path

def extract_top_frames(video_path, num_frames=5, output_dir=None):
    """
    Extract the top N frames from a video based on quality metrics.
    Focus only on the very last frames for smoother transitions.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of top frames to extract
        output_dir: Directory to save extracted frames (optional)
        
    Returns:
        List of paths to the extracted frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(video_path)
    
    logger.info(f"Extracting top {num_frames} frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Focus ONLY on the last frames (max 10 frames from the end)
    # This ensures we're getting frames that will transition smoothly to the next video
    last_frames_count = min(10, frame_count)
    start_frame = max(0, frame_count - last_frames_count)
    
    # Extract frames and calculate quality scores
    all_frames = []
    frame_scores = []
    frame_positions = []
    
    # Capture all the last frames
    for i in range(start_frame, frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Calculate quality score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Quality metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
        _, std_dev = cv2.meanStdDev(gray)  # Contrast
        contrast = std_dev[0][0]
        
        # Brightness score (penalize too dark or too bright)
        mean_val = cv2.mean(gray)[0]
        brightness_score = 1.0 - abs((mean_val - 127.5) / 127.5)
        
        # Colorfulness
        b, g, r = cv2.split(frame)
        colorfulness = (np.std(r - g)**2 + np.std(r - b)**2 + np.std(g - b)**2) / 3
        
        # Combined weighted score
        score = (laplacian_var * 0.4 +
                contrast * 0.3 +
                brightness_score * 0.15 +
                colorfulness * 0.15)
                
        all_frames.append(frame)
        frame_scores.append(score)
        frame_positions.append(i)
    
    cap.release()
    
    # If no frames were extracted, use the last frame
    if not all_frames:
        logger.warning(f"No frames extracted, using last frame")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read frame from video: {video_path}")
        
        # Save the single frame
        output_path = os.path.join(output_dir, f"frame_last.jpg")
        cv2.imwrite(output_path, frame)
        return [output_path]
    
    # Sort frames by score (descending)
    sorted_indices = sorted(range(len(frame_scores)), key=lambda i: frame_scores[i], reverse=True)
    
    # Take the top N frames
    top_indices = sorted_indices[:min(num_frames, len(sorted_indices))]
    # Sort by frame position for sequential output
    top_indices = sorted(top_indices, key=lambda i: frame_positions[i])
    
    # Save the top frames
    output_paths = []
    for idx, frame_idx in enumerate(top_indices):
        frame = all_frames[frame_idx]
        position = frame_positions[frame_idx]
        score = frame_scores[frame_idx]
        
        output_path = os.path.join(output_dir, f"frame_{idx+1}.jpg")
        cv2.imwrite(output_path, frame)
        
        logger.info(f"Saved frame {idx+1}/{len(top_indices)} from position {position}/{frame_count} with score {score:.2f}")
        output_paths.append(output_path)
    
    return output_paths

def extract_last_frame(video_path, output_path=None):
    """
    Extract just the last frame from a video for continuity between chains.
    Uses lossless PNG format to prevent any quality degradation.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted frame
        
    Returns:
        Path to the extracted frame
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Determine output path - use PNG for lossless quality
    if output_path is None:
        output_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_last_frame.png")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Extracting last frame from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Go directly to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read last frame from video: {video_path}")
    
    # Save as PNG with maximum quality (lossless)
    success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    if not success:
        raise IOError(f"Failed to save frame to {output_path}")
    
    logger.info(f"Saved last frame to: {output_path} (lossless PNG format)")
    
    return output_path

def enhance_frame_quality(frame_path, output_path=None):
    """
    Apply quality enhancement to prevent degradation across chain generations.
    
    Args:
        frame_path: Path to the input frame
        output_path: Path to save the enhanced frame (optional)
        
    Returns:
        Path to the enhanced frame
    """
    try:
        # Open the image
        img = Image.open(frame_path).convert("RGB")
        
        # Apply a series of enhancements to improve quality
        # 1. Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)  # Subtle sharpening
        
        # 2. Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # Slight contrast boost
        
        # 3. Enhance color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.05)  # Subtle color enhancement
        
        # 4. Enhance brightness (if needed)
        # Get average brightness
        brightness = sum(img.convert("L").getdata()) / (img.width * img.height)
        if brightness < 100:  # Image is too dark
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)  # Brighten slightly
        elif brightness > 200:  # Image is too bright
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.95)  # Darken slightly
        
        # Save the enhanced image
        if output_path is None:
            output_path = frame_path.replace(".jpg", "_enhanced.jpg")
            if output_path == frame_path:
                output_path = frame_path.replace(".jpeg", "_enhanced.jpeg")
                if output_path == frame_path:
                    # Add suffix if extension wasn't changed
                    output_path = frame_path + ".enhanced.jpg"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with high quality
        img.save(output_path, quality=95)
        logger.info(f"Enhanced frame saved to: {output_path}")
        
        return output_path
    except Exception as e:
        logger.exception(f"Error enhancing frame quality: {str(e)}")
        # If enhancement fails, return original path
        return frame_path

def auto_adjust_image(image_path, output_path=None, brightness_threshold_low=90, brightness_threshold_high=210, 
                      saturation_threshold=0.7, contrast_enhance_factor=1.1):
    """
    Automatically analyze an image and adjust it if it's too dark, too bright, or oversaturated.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the adjusted image (optional)
        brightness_threshold_low: Threshold below which the image is considered too dark (0-255)
        brightness_threshold_high: Threshold above which the image is considered too bright (0-255)
        saturation_threshold: Threshold above which the image is considered oversaturated (0-1)
        contrast_enhance_factor: Factor to enhance contrast if needed
        
    Returns:
        Tuple of (path_to_adjusted_image, adjustments_made)
    """
    try:
        # Open the image
        img = Image.open(image_path).convert("RGB")
        adjustments = []
        
        # 1. Check and adjust brightness
        brightness = sum(img.convert("L").getdata()) / (img.width * img.height)
        logger.info(f"Image brightness: {brightness}")
        
        # Initial contrast analysis to determine the need for contrast preservation
        np_img = np.array(img.convert("L"))
        hist = cv2.calcHist([np_img], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()
        
        # Calculate histogram spread - standard deviation of histogram
        hist_indices = np.arange(256)
        hist_mean = np.sum(hist_indices * hist_normalized.flatten())
        hist_std = np.sqrt(np.sum(((hist_indices - hist_mean) ** 2) * hist_normalized.flatten()))
        logger.info(f"Histogram standard deviation: {hist_std}")
        
        # Check if image is too dark but has good contrast
        preserve_contrast = hist_std > 45  # Higher standard deviation means good contrast
        
        if brightness < brightness_threshold_low:
            # Image is too dark - brighten it more gently
            enhancer = ImageEnhance.Brightness(img)
            # More conservative brightness factor - max 30% increase instead of 50%
            brightness_factor = 1.0 + min(0.3, (brightness_threshold_low - brightness) / 120)
            img = enhancer.enhance(brightness_factor)
            adjustments.append(f"increased brightness by {(brightness_factor-1)*100:.0f}%")
            logger.info(f"Image was too dark, brightness increased by factor {brightness_factor}")
            
            # If the image had good contrast originally, boost contrast to preserve details
            if preserve_contrast:
                enhancer = ImageEnhance.Contrast(img)
                # Subtle contrast boost to balance the brightness increase
                contrast_boost = 1.0 + min(0.2, (brightness_factor - 1) * 0.6)
                img = enhancer.enhance(contrast_boost)
                adjustments.append(f"preserved details with {(contrast_boost-1)*100:.0f}% contrast boost")
                logger.info(f"Added contrast boost of {contrast_boost} to preserve details after brightening")
                
        elif brightness > brightness_threshold_high:
            # Image is too bright - darken it
            enhancer = ImageEnhance.Brightness(img)
            # Calculate adjustment factor based on how bright it is - slightly more conservative
            brightness_factor = max(0.75, 1.0 - (brightness - brightness_threshold_high) / 250)
            img = enhancer.enhance(brightness_factor)
            adjustments.append(f"decreased brightness by {(1-brightness_factor)*100:.0f}%")
            logger.info(f"Image was too bright, brightness decreased by factor {brightness_factor}")
            
            # Boost contrast slightly when darkening to avoid flat appearance
            enhancer = ImageEnhance.Contrast(img)
            contrast_boost = 1.0 + min(0.15, (1 - brightness_factor) * 0.5)
            img = enhancer.enhance(contrast_boost)
            adjustments.append(f"boosted contrast by {(contrast_boost-1)*100:.0f}%")
            logger.info(f"Added contrast boost of {contrast_boost} after darkening")
        
        # 2. Check and adjust saturation
        np_img = np.array(img)
        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv_img[:, :, 1]) / 255.0  # Normalize to 0-1
        logger.info(f"Image saturation: {saturation}")
        
        if saturation > saturation_threshold:
            # Image is oversaturated - reduce saturation
            enhancer = ImageEnhance.Color(img)
            # Calculate adjustment factor based on how saturated it is
            saturation_factor = max(0.5, 1.0 - (saturation - saturation_threshold))
            img = enhancer.enhance(saturation_factor)
            adjustments.append(f"decreased saturation by {(1-saturation_factor)*100:.0f}%")
            logger.info(f"Image was oversaturated, saturation decreased by factor {saturation_factor}")
        
        # 3. Re-analyze contrast after brightness/saturation adjustments
        np_img = np.array(img.convert("L"))
        hist = cv2.calcHist([np_img], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()
        
        # Check if histogram is concentrated in a narrow range
        middle_range_pct = hist_normalized[64:192].sum()
        logger.info(f"Percentage of pixels in middle range: {middle_range_pct*100:.1f}%")
        
        # Only increase contrast if the histogram is really concentrated (85% -> 90%)
        if middle_range_pct > 0.9:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_enhance_factor)
            adjustments.append(f"increased contrast by {(contrast_enhance_factor-1)*100:.0f}%")
            logger.info(f"Image lacked contrast, increased by factor {contrast_enhance_factor}")
        
        # If no adjustments were made, return the original image path
        if not adjustments:
            logger.info("No adjustments needed for the image")
            return image_path, []
        
        # Save the adjusted image
        if output_path is None:
            output_path = image_path.replace(".jpg", "_adjusted.jpg")
            if output_path == image_path:
                output_path = image_path.replace(".jpeg", "_adjusted.jpeg")
                if output_path == image_path:
                    output_path = image_path.replace(".png", "_adjusted.png")
                    if output_path == image_path:
                        # Add suffix if extension wasn't changed
                        output_path = image_path + ".adjusted.jpg"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with high quality
        img.save(output_path, quality=95)
        logger.info(f"Adjusted image saved to: {output_path}")
        
        return output_path, adjustments
    
    except Exception as e:
        logger.exception(f"Error automatically adjusting image: {str(e)}")
        return image_path, []

def stitch_videos(video_paths, output_path):
    """
    Stitch multiple videos together into a single video file.
    
    Args:
        video_paths: List of paths to video files
        output_path: Path to save the final video
        
    Returns:
        Path to the stitched video
    """
    if not video_paths:
        raise ValueError("No video paths provided")
        
    logger.info(f"Stitching {len(video_paths)} videos into: {output_path}")
    
    # Check if all videos exist
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Method 1: Use ffmpeg if available (more efficient)
        if FFMPEG_AVAILABLE:
            logger.info("Using ffmpeg for video concatenation")
            
            # Create a temporary list file for ffmpeg - use proper path formatting
            list_file_path = output_path.replace(".mp4", "_list.txt")
            output_dir = os.path.dirname(output_path)
            
            with open(list_file_path, 'w') as f:
                for path in video_paths:
                    # Convert to absolute path with forward slashes
                    abs_path = os.path.abspath(path).replace('\\', '/')
                    
                    # Write the path as-is without trying to make it relative
                    # This works better with FFmpeg's concat demuxer on Windows
                    f.write(f"file '{abs_path}'\n")
            
            # Run ffmpeg command to concatenate videos in the same directory as the list file
            # This helps FFmpeg properly resolve the paths
            working_dir = os.path.dirname(list_file_path)
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", os.path.basename(list_file_path), "-c", "copy", os.path.basename(output_path)
            ]
            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=working_dir  # Set the working directory for ffmpeg
                )
            except subprocess.CalledProcessError as e:
                # Log the detailed ffmpeg error output
                error_output = e.stderr if hasattr(e, 'stderr') and e.stderr else "No error details available"
                logger.error(f"FFmpeg error: {error_output}")
                
                # Attempt re-encoding method if direct copy fails
                logger.info("Direct concatenation failed, trying with re-encoding...")
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", os.path.basename(list_file_path), "-c:v", "libx264", "-preset", "medium", 
                    "-crf", "23", "-c:a", "aac", os.path.basename(output_path)
                ]
                logger.info(f"Running ffmpeg with re-encoding: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(
                        cmd, 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=working_dir  # Set the working directory for ffmpeg
                    )
                except subprocess.CalledProcessError as e2:
                    error_output = e2.stderr if hasattr(e2, 'stderr') and e2.stderr else "No error details available"
                    logger.error(f"FFmpeg re-encoding also failed: {error_output}")
                    raise RuntimeError(f"FFmpeg failed with both methods. Error: {error_output}")
            
            # Clean up the temporary list file
            try:
                os.remove(list_file_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {list_file_path}: {str(e)}")
            
            logger.info(f"Successfully stitched videos using ffmpeg: {output_path}")
            return output_path
            
        # Method 2: Use moviepy (backup method)
        logger.info("Using moviepy for video concatenation")
        clips = []
        
        try:
            for path in video_paths:
                logger.info(f"Loading video clip: {path}")
                clip = VideoFileClip(path)
                clips.append(clip)
                
            logger.info("Concatenating video clips")
            final_clip = concatenate_videoclips(clips)
            
            logger.info(f"Writing final video to: {output_path}")
            final_clip.write_videofile(output_path, codec="libx264", verbose=False, logger=None)
            
            # Close clips
            for clip in clips:
                clip.close()
                
            logger.info(f"Successfully stitched videos using moviepy: {output_path}")
            return output_path
                
        except Exception as e:
            logger.exception(f"Error with moviepy concatenation: {str(e)}")
            # Close any open clips
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            
            # Method 3: Fallback - just copy the first video as final output
            logger.warning("Both FFmpeg and moviepy failed. Using first video as output.")
            if video_paths:
                import shutil
                try:
                    shutil.copy2(video_paths[0], output_path)
                    logger.info(f"Copied first video to output: {output_path}")
                    return output_path
                except Exception as copy_error:
                    logger.error(f"Failed to copy first video: {copy_error}")
                    # Return the first video path as-is
                    return video_paths[0]
            else:
                raise ValueError("No videos to stitch")
            
    except Exception as e:
        logger.error(f"Error stitching videos: {str(e)}")
        raise

def extract_and_trim_best_frame(video_path, output_dir):
    """
    Extract a frame from the VERY END of the video for true progression continuity.
    This prioritizes progression over quality to ensure seamless chain transitions.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save outputs
        
    Returns:
        tuple: (best_frame_path, trimmed_video_path)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    best_frame_path = os.path.join(output_dir, f"{base_name}_best_frame.png")
    trimmed_video_path = os.path.join(output_dir, f"{base_name}_trimmed.mp4")
    
    logger.info(f"Extracting FINAL progression frame from video: {video_path}")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # MODIFIED APPROACH: Extract from frames 5-8 before the very end to avoid padding/duplicate frames
    # Many video encoders add padding frames at the absolute end
    offset_from_end = 5  # Skip the last 5 frames which might be padding
    num_frames_to_analyze = min(6, frame_count - offset_from_end)  # Analyze 6 frames before the offset
    start_frame = max(0, frame_count - offset_from_end - num_frames_to_analyze)
    end_frame = frame_count - offset_from_end
    
    logger.info(f"AVOIDING PADDING: Analyzing frames {start_frame}-{end_frame-1} (skipping last {offset_from_end} frames which might be padding)")
    
    # Extract the very last frames and pick the best progression frame
    frames = []
    scores = []
    positions = []
    
    for pos in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # For progression, we prioritize TEMPORAL POSITION over quality
        # The closer to the meaningful end, the better for continuity
        temporal_weight = (pos - start_frame + 1) / num_frames_to_analyze
        
        # Basic quality metrics (but weighted much lower than temporal position)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
        _, std_dev = cv2.meanStdDev(gray)  # Contrast
        contrast = std_dev[0][0]
        
        # Brightness balance (avoid completely black or white frames)
        mean_val = cv2.mean(gray)[0]
        brightness_score = 1.0 - abs((mean_val - 127.5) / 127.5)
        
        # Frame difference score (prefer frames that are different from previous ones)
        difference_score = 1.0  # Default score
        if len(frames) > 0:  # Compare with previous frame
            prev_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray, prev_gray)
            difference_score = np.mean(frame_diff) / 255.0  # Normalize to 0-1
        
        # HEAVILY weight temporal position for true progression
        score = (
            temporal_weight * 0.6 +          # 60% weight on being at the END
            (laplacian_var / 1000) * 0.15 +  # 15% weight on sharpness  
            (contrast / 100) * 0.1 +         # 10% weight on contrast
            brightness_score * 0.05 +        # 5% weight on brightness balance
            difference_score * 0.1           # 10% weight on being different from previous frame
        )
        
        frames.append(frame)
        scores.append(score)
        positions.append(pos)
        
        logger.info(f"Frame {pos}/{end_frame-1}: temporal_weight={temporal_weight:.2f}, diff_score={difference_score:.3f}, total_score={score:.3f}")
    
    cap.release()
    
    if not frames:
        raise ValueError("Could not extract any frames for progression analysis")
    
    # Find the frame with highest progression score (should heavily favor the last frame)
    best_idx = scores.index(max(scores))
    best_position = positions[best_idx]
    best_frame = frames[best_idx]
    
    # Save the progression frame as PNG (lossless)
    cv2.imwrite(best_frame_path, best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    logger.info(f"PROGRESSION frame saved: {best_frame_path} (score: {scores[best_idx]:.3f}, position: {best_position}/{frame_count-1}, avoiding last {offset_from_end} padding frames)")
    logger.info(f">>> This frame represents the TRUE END STATE for seamless chain continuity <<<")
    
    # Add debugging - compute hash of the extracted frame to verify uniqueness
    with open(best_frame_path, 'rb') as f:
        frame_hash = hashlib.md5(f.read()).hexdigest()
    logger.info(f"EXTRACTED FRAME HASH: {frame_hash}")
    logger.info(f"Frame file size: {os.path.getsize(best_frame_path)} bytes")
    
    # Always trim to the selected progression frame for consistency
    if best_position < frame_count - 1:
        logger.info(f"Trimming video to end at progression frame {best_position+1} (position {best_position+1}/{frame_count})")
        
        # Use FFmpeg to trim the video
        if FFMPEG_AVAILABLE:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vframes", str(best_position + 1),  # Include frames up to the progression frame
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-vsync", "vfr",  # Variable framerate to avoid frame duplication
                "-copyts",  # Preserve timestamps
                trimmed_video_path
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # Verify the trimmed video
                if os.path.exists(trimmed_video_path):
                    check_cap = cv2.VideoCapture(trimmed_video_path)
                    trimmed_frame_count = int(check_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    check_cap.release()
                    logger.info(f"Video trimmed for progression: {trimmed_video_path} (frames: {trimmed_frame_count}, original: {frame_count})")
                    
                    # File size info
                    orig_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                    trimmed_size = os.path.getsize(trimmed_video_path) / (1024 * 1024)  # MB
                    logger.info(f"Original: {orig_size:.2f}MB → Trimmed: {trimmed_size:.2f}MB")
                else:
                    logger.warning(f"Trimmed video not created, using original")
                    return best_frame_path, video_path
                
                return best_frame_path, trimmed_video_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"FFmpeg trimming failed: {e.stderr}, using original video")
                return best_frame_path, video_path
        else:
            logger.warning("FFmpeg not available, using original video")
            return best_frame_path, video_path
    else:
        logger.info("Progression frame is already the last frame")
        return best_frame_path, video_path

def extract_simple_last_frame(video_path, output_dir):
    """
    Simple frame extraction that gets the exact last frame without any filtering.
    This is a backup method in case the complex extraction is having issues.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save outputs
        
    Returns:
        tuple: (last_frame_path, original_video_path)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    last_frame_path = os.path.join(output_dir, f"{base_name}_last_frame.png")
    
    logger.info(f"SIMPLE EXTRACTION: Getting exact last frame from video: {video_path}")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Go to the absolute last frame
    last_frame_index = frame_count - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read last frame from video: {video_path}")
    
    # Save the last frame as PNG (lossless)
    cv2.imwrite(last_frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    logger.info(f"SIMPLE EXTRACTION: Last frame saved: {last_frame_path} (frame {last_frame_index}/{frame_count-1})")
    
    # Add debugging - compute hash of the extracted frame
    with open(last_frame_path, 'rb') as f:
        frame_hash = hashlib.md5(f.read()).hexdigest()
    logger.info(f"SIMPLE EXTRACTION HASH: {frame_hash}")
    logger.info(f"Frame file size: {os.path.getsize(last_frame_path)} bytes")
    
    # Return the frame path and the original video (no trimming for simple method)
    return last_frame_path, video_path

def extract_high_quality_frame_for_chain(video_path, output_dir):
    """
    Extract the highest quality frame for chain continuity with maximum quality preservation.
    
    This function prioritizes image quality to prevent degradation across video chains:
    - Uses lossless PNG format
    - Applies quality enhancement
    - Implements noise reduction
    - Preserves maximum detail
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save outputs
        
    Returns:
        tuple: (enhanced_frame_path, original_video_path)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_frame_path = os.path.join(output_dir, f"{base_name}_raw_frame.png")
    enhanced_frame_path = os.path.join(output_dir, f"{base_name}_enhanced_frame.png")
    
    logger.info(f"HIGH-QUALITY EXTRACTION: Extracting best quality frame from: {video_path}")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Extract frames from the last 20% of the video for quality analysis
    start_frame = max(0, int(frame_count * 0.8))
    frames_to_analyze = []
    quality_scores = []
    frame_positions = []
    
    # Analyze frames for quality (not just the last frame)
    sample_interval = max(1, (frame_count - start_frame) // 15)  # Sample up to 15 frames
    
    for pos in range(start_frame, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Calculate comprehensive quality score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast
        _, std_dev = cv2.meanStdDev(gray)
        contrast = std_dev[0][0]
        
        # Brightness balance (avoid overexposed/underexposed frames)
        mean_val = cv2.mean(gray)[0]
        brightness_score = 1.0 - abs((mean_val - 127.5) / 127.5)
        
        # Edge density (more edges = more detail)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Noise estimation (lower noise = better quality)
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise_level = np.mean(np.abs(gray.astype(float) - noise.astype(float)))
        noise_score = max(0, 1.0 - (noise_level / 50.0))  # Normalize noise
        
        # Color richness
        b, g, r = cv2.split(frame)
        color_variance = (np.var(r) + np.var(g) + np.var(b)) / 3
        color_score = min(1.0, color_variance / 1000.0)
        
        # Combined quality score (weighted)
        quality_score = (
            laplacian_var * 0.25 +      # Sharpness
            contrast * 0.2 +            # Contrast
            brightness_score * 100 * 0.15 +  # Brightness balance
            edge_density * 500 * 0.15 + # Edge detail
            noise_score * 100 * 0.1 +   # Low noise
            color_score * 100 * 0.15    # Color richness
        )
        
        frames_to_analyze.append(frame)
        quality_scores.append(quality_score)
        frame_positions.append(pos)
        
        logger.debug(f"Frame {pos}: sharpness={laplacian_var:.1f}, contrast={contrast:.1f}, "
                    f"brightness={brightness_score:.2f}, edges={edge_density:.3f}, "
                    f"noise={noise_score:.2f}, color={color_score:.2f}, total={quality_score:.1f}")
    
    cap.release()
    
    if not frames_to_analyze:
        # Fallback to last frame if analysis failed
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read any frame from video: {video_path}")
        
        frames_to_analyze = [frame]
        quality_scores = [0.0]
        frame_positions = [frame_count - 1]
    
    # Select the highest quality frame
    best_idx = quality_scores.index(max(quality_scores))
    best_frame = frames_to_analyze[best_idx]
    best_position = frame_positions[best_idx]
    best_score = quality_scores[best_idx]
    
    logger.info(f"SELECTED FRAME: position {best_position}/{frame_count-1}, quality score: {best_score:.1f}")
    
    # Save raw frame as lossless PNG
    success = cv2.imwrite(raw_frame_path, best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not success:
        raise IOError(f"Failed to save raw frame to {raw_frame_path}")
    
    # Apply quality enhancement
    enhanced_frame_path = enhance_frame_quality_advanced(raw_frame_path, enhanced_frame_path)
    
    # Log file sizes for debugging
    raw_size = os.path.getsize(raw_frame_path) / 1024  # KB
    enhanced_size = os.path.getsize(enhanced_frame_path) / 1024  # KB
    logger.info(f"QUALITY PRESERVATION: Raw frame: {raw_size:.1f}KB → Enhanced: {enhanced_size:.1f}KB")
    
    return enhanced_frame_path, video_path

def enhance_frame_quality_advanced(input_path, output_path):
    """
    Apply advanced quality enhancement to a frame to combat video chain degradation.
    
    Args:
        input_path: Path to input frame
        output_path: Path to save enhanced frame
        
    Returns:
        Path to enhanced frame
    """
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import numpy as np
        
        # Open image with PIL for high-quality processing
        img = Image.open(input_path).convert("RGB")
        original_size = img.size
        
        # 1. Upscale slightly to add detail before enhancement
        upscale_factor = 1.1
        new_size = (int(original_size[0] * upscale_factor), int(original_size[1] * upscale_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 2. Noise reduction (very light to preserve detail)
        img_array = np.array(img)
        
        # Convert to OpenCV format for noise reduction
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply light denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 3, 3, 7, 21)
        
        # Convert back to PIL
        img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        
        # 3. Enhance sharpness (moderate)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.15)
        
        # 4. Enhance contrast (light)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.08)
        
        # 5. Enhance color saturation (very light)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.03)
        
        # 6. Apply unsharp mask for detail enhancement
        img_array = np.array(img)
        
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(img_array, (3, 3), 1.0)
        unsharp_mask = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
        
        # Ensure values are in valid range
        unsharp_mask = np.clip(unsharp_mask, 0, 255)
        img = Image.fromarray(unsharp_mask.astype(np.uint8))
        
        # 7. Resize back to original dimensions with high-quality resampling
        img = img.resize(original_size, Image.Resampling.LANCZOS)
        
        # 8. Final brightness/contrast adjustment based on image analysis
        img_array = np.array(img.convert("L"))  # Convert to grayscale for analysis
        mean_brightness = np.mean(img_array)
        
        # Adjust if too dark or too bright
        if mean_brightness < 100:  # Too dark
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
        elif mean_brightness > 180:  # Too bright
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.98)
        
        # Save enhanced image with maximum quality
        img.save(output_path, format='PNG', optimize=False, compress_level=0)
        
        logger.info(f"ADVANCED ENHANCEMENT applied: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Advanced enhancement failed: {str(e)}, using basic enhancement")
        # Fallback to basic enhancement
        return enhance_frame_quality(input_path, output_path)