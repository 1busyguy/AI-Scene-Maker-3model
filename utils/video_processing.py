import cv2
import os
import logging
import subprocess
from moviepy.editor import concatenate_videoclips, VideoFileClip
import numpy as np
from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_ffmpeg_available():
    """Check if ffmpeg is available in the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

FFMPEG_AVAILABLE = is_ffmpeg_available()
if not FFMPEG_AVAILABLE:
    logger.warning("ffmpeg not found. Video concatenation may not work optimally.")

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
            raise
            
    except Exception as e:
        logger.error(f"Error stitching videos: {str(e)}")
        raise

def extract_and_trim_best_frame(video_path, output_dir):
    """
    Find the highest quality frame among the last 10 frames,
    trim the video to end with this frame, and return the frame.
    
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
    
    logger.info(f"Analyzing last 10 frames of video for highest quality: {video_path}")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Determine frames to analyze (last 10 or fewer)
    num_frames_to_analyze = min(10, frame_count)
    start_frame = max(0, frame_count - num_frames_to_analyze)
    
    # Extract and score the last frames
    frames = []
    scores = []
    positions = []
    
    for pos in range(start_frame, frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Calculate quality score based on multiple metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Contrast (standard deviation)
        _, std_dev = cv2.meanStdDev(gray)
        contrast = std_dev[0][0]
        
        # 3. Brightness score (penalize too dark or too bright)
        mean_val = cv2.mean(gray)[0]
        brightness_score = 1.0 - abs((mean_val - 127.5) / 127.5)
        
        # 4. Noise estimation (lower is better)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32)))
        noise_score = 1.0 / (1.0 + noise)
        
        # Combined weighted score
        score = (laplacian_var * 0.4 +  # Sharpness
                contrast * 0.3 +         # Contrast
                brightness_score * 0.2 +  # Balanced brightness
                noise_score * 0.1)        # Low noise
        
        frames.append(frame)
        scores.append(score)
        positions.append(pos)
    
    cap.release()
    
    if not frames:
        raise ValueError("Could not extract any frames for quality analysis")
    
    # Find best frame index and position
    best_idx = scores.index(max(scores))
    best_position = positions[best_idx]
    best_frame = frames[best_idx]
    
    # Save the best frame as PNG (lossless)
    cv2.imwrite(best_frame_path, best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    logger.info(f"Best frame saved to: {best_frame_path} (quality score: {scores[best_idx]:.2f}, position: {best_position}/{frame_count})")
    
    # If best frame isn't the last frame, trim the video
    if best_position < frame_count - 1:
        # Calculate duration to keep (in seconds)
        duration_to_keep = (best_position + 1) / fps
        
        logger.info(f"Trimming video to end at frame {best_position+1} (duration: {duration_to_keep:.2f}s)")
        
        # Use FFmpeg to trim the video
        if FFMPEG_AVAILABLE:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-t", str(duration_to_keep),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
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
                logger.info(f"Video trimmed successfully: {trimmed_video_path}")
                return best_frame_path, trimmed_video_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"FFmpeg trimming failed: {e.stderr}, falling back to original video")
                return best_frame_path, video_path
        else:
            logger.warning("FFmpeg not available for trimming, returning original video")
            return best_frame_path, video_path
    else:
        logger.info("Best frame is already the last frame, no trimming needed")
        return best_frame_path, video_path