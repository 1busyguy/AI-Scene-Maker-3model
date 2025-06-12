 # utils/video_processing.py - Fixed version addressing over-enhancement and chain issues
import cv2
import os
import logging
import subprocess
from moviepy.editor import concatenate_videoclips, VideoFileClip
import numpy as np
from PIL import Image, ImageEnhance
import hashlib
from typing import Tuple, Optional
import json
import time

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the enhanced processing module
try:
    from .enhanced_video_processing import (
        ChainQualityPreserver, 
        extract_optimal_frame_with_preservation,
        apply_quality_preservation_enhancement
    )
    ENHANCED_PROCESSING_AVAILABLE = True
    logger.info("Enhanced video processing available")
except ImportError as e:
    logger.warning(f"Enhanced processing not available: {e}")
    ENHANCED_PROCESSING_AVAILABLE = False

def is_ffmpeg_available():
    """Check if ffmpeg and ffprobe are available in the system."""
    ffmpeg_available = False
    ffprobe_available = False
    
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        ffmpeg_available = True
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError):
        pass
    
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        ffprobe_available = True
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError):
        pass
    
    if not ffmpeg_available or not ffprobe_available:
        try:
            import platform
            if platform.system() == "Windows":
                if os.path.exists("ffmpeg.exe"):
                    ffmpeg_available = True
                if os.path.exists("ffprobe.exe"):
                    ffprobe_available = True
                
                for path in os.environ["PATH"].split(os.pathsep):
                    if not ffmpeg_available and os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        ffmpeg_available = True
                    if not ffprobe_available and os.path.exists(os.path.join(path, "ffprobe.exe")):
                        ffprobe_available = True
        except Exception as e:
            logger.debug(f"Error checking for ffmpeg/ffprobe: {str(e)}")
    
    if ffmpeg_available and ffprobe_available:
        logger.info("FFmpeg and ffprobe found and available.")
        return True
    else:
        logger.warning("FFmpeg or ffprobe missing. Some video features will be limited.")
        return False

FFMPEG_AVAILABLE = is_ffmpeg_available()

# Maintain backwards compatibility - some parts of the UI expect FFPROBE_AVAILABLE
FFPROBE_AVAILABLE = FFMPEG_AVAILABLE

class EnhancementTracker:
    """Track enhancement history to prevent over-processing"""
    
    def __init__(self):
        self.enhancement_history = {}
        self.chain_metadata = {}
    
    def record_enhancement(self, image_path: str, enhancement_level: str, source_type: str = "unknown"):
        """
        Record that an image has been enhanced
        
        Args:
            image_path: Path to the enhanced image
            enhancement_level: Level of enhancement applied
            source_type: Type of source (original, extracted_frame, etc.)
        """
        # Create image hash for tracking
        try:
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            
            self.enhancement_history[image_hash] = {
                'path': image_path,
                'enhancement_level': enhancement_level,
                'source_type': source_type,
                'timestamp': time.time(),
                'enhancement_count': self.enhancement_history.get(image_hash, {}).get('enhancement_count', 0) + 1
            }
            
            logger.info(f"Enhancement recorded: {enhancement_level} on {source_type} (count: {self.enhancement_history[image_hash]['enhancement_count']})")
            
        except Exception as e:
            logger.error(f"Failed to record enhancement: {str(e)}")
    
    def get_enhancement_history(self, image_path: str) -> dict:
        """Get enhancement history for an image"""
        try:
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            return self.enhancement_history.get(image_hash, {})
        except:
            return {}
    
    def should_skip_enhancement(self, image_path: str, proposed_enhancement: str) -> Tuple[bool, str]:
        """
        Determine if enhancement should be skipped to prevent over-processing
        
        Returns:
            Tuple of (should_skip, reason)
        """
        history = self.get_enhancement_history(image_path)
        
        if not history:
            return False, "No previous enhancement"
        
        enhancement_count = history.get('enhancement_count', 0)
        last_enhancement = history.get('enhancement_level', '')
        source_type = history.get('source_type', '')
        
        # Skip if already heavily enhanced
        if enhancement_count >= 3:
            return True, f"Already enhanced {enhancement_count} times"
        
        # Skip if already enhanced at same or higher level
        enhancement_levels = {
            'none': 0,
            'light': 1,
            'moderate': 2,
            'heavy': 3,
            'maximum': 4
        }
        
        current_level = enhancement_levels.get(last_enhancement, 0)
        proposed_level = enhancement_levels.get(proposed_enhancement, 1)
        
        if current_level >= proposed_level:
            return True, f"Already enhanced at {last_enhancement} level"
        
        # Skip if source was already an extracted frame and we're about to enhance again
        if source_type == "extracted_frame" and enhancement_count >= 1:
            return True, "Extracted frames should not be over-enhanced"
        
        return False, "Enhancement approved"

# Global enhancement tracker
_enhancement_tracker = EnhancementTracker()

def get_enhancement_tracker():
    """Get global enhancement tracker"""
    return _enhancement_tracker

class ChainStateManager:
    """Manage state across video chain generation to prevent KLING re-use issue"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset chain state for new generation session"""
        self.chain_count = 0
        self.original_image_path = None
        self.current_image_path = None
        self.last_video_path = None
        self.last_extracted_frame = None
        self.model_history = []
        self.image_progression = []
        
        logger.info("Chain state reset for new session")
    
    def set_original_image(self, image_path: str):
        """Set the original input image"""
        self.original_image_path = image_path
        self.current_image_path = image_path
        self.image_progression.append({
            'chain': 0,
            'type': 'original',
            'path': image_path,
            'timestamp': time.time()
        })
        logger.info(f"Original image set: {image_path}")
    
    def start_new_chain(self, chain_number: int, model: str):
        """Start a new chain in the sequence"""
        self.chain_count = chain_number
        self.model_history.append({
            'chain': chain_number,
            'model': model,
            'input_image': self.current_image_path,
            'timestamp': time.time()
        })
        
        logger.info(f"🔗 CHAIN {chain_number}: Starting with model {model}")
        logger.info(f"🔗 CHAIN {chain_number}: Input image: {self.current_image_path}")
    
    def complete_chain(self, chain_number: int, video_path: str, extracted_frame_path: str):
        """Complete a chain and update state"""
        self.last_video_path = video_path
        self.last_extracted_frame = extracted_frame_path
        
        # Update current image for next chain
        if extracted_frame_path and os.path.exists(extracted_frame_path):
            self.current_image_path = extracted_frame_path
            
            self.image_progression.append({
                'chain': chain_number,
                'type': 'extracted_frame',
                'path': extracted_frame_path,
                'source_video': video_path,
                'timestamp': time.time()
            })
            
            logger.info(f"🔗 CHAIN {chain_number}: Completed, next input: {extracted_frame_path}")
        else:
            logger.warning(f"🔗 CHAIN {chain_number}: No extracted frame, keeping current image")
    
    def get_current_input_image(self) -> str:
        """Get the current image to use for the next chain"""
        if self.chain_count == 0:
            # First chain uses original
            return self.original_image_path
        else:
            # Subsequent chains use last extracted frame
            if self.last_extracted_frame and os.path.exists(self.last_extracted_frame):
                logger.info(f"🔗 Using extracted frame for chain {self.chain_count + 1}: {self.last_extracted_frame}")
                return self.last_extracted_frame
            else:
                logger.warning(f"🔗 Extracted frame not available, falling back to original: {self.original_image_path}")
                return self.original_image_path
    
    def validate_chain_progression(self, chain_number: int, proposed_image: str) -> Tuple[bool, str]:
        """
        Validate that chain progression is correct (fixes KLING re-use issue)
        
        Returns:
            Tuple of (is_valid, reason/correction)
        """
        expected_image = self.get_current_input_image()
        
        # For first chain, should use original
        if chain_number == 1:
            if proposed_image == self.original_image_path:
                return True, "Correct: Using original image for first chain"
            else:
                return False, f"ERROR: First chain should use original image {self.original_image_path}, not {proposed_image}"
        
        # For subsequent chains, should use extracted frame from previous chain
        if chain_number > 1:
            if self.last_extracted_frame and proposed_image == self.last_extracted_frame:
                return True, f"Correct: Using extracted frame from previous chain"
            elif proposed_image == self.original_image_path:
                return False, f"ERROR: Chain {chain_number} should use extracted frame {self.last_extracted_frame}, not original image"
            else:
                return False, f"ERROR: Chain {chain_number} should use extracted frame {self.last_extracted_frame}, not {proposed_image}"
        
        return True, "Chain progression validated"
    
    def get_debug_info(self) -> dict:
        """Get debug information about chain state"""
        return {
            'chain_count': self.chain_count,
            'original_image': self.original_image_path,
            'current_image': self.current_image_path,
            'last_video': self.last_video_path,
            'last_extracted_frame': self.last_extracted_frame,
            'model_history': self.model_history,
            'image_progression': self.image_progression
        }

# Global chain state manager
_chain_state = ChainStateManager()

def get_chain_state_manager():
    """Get global chain state manager"""
    return _chain_state

def extract_high_quality_frame_for_chain(video_path: str, output_dir: str, 
                                        original_image_path: Optional[str] = None,
                                        chain_number: int = 1,
                                        avoid_over_enhancement: bool = True) -> Tuple[str, str]:
    """
    Extract the highest quality frame for chain continuity with over-enhancement prevention.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save outputs
        original_image_path: Path to original image for quality reference
        chain_number: Current chain number (for logging)
        avoid_over_enhancement: Whether to prevent over-enhancement
        
    Returns:
        tuple: (enhanced_frame_path, original_video_path)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"🎯 CHAIN {chain_number}: Quality frame extraction from: {video_path}")
    
    # Update chain state
    chain_state = get_chain_state_manager()
    
    # Use enhanced processing if available and this is not over-enhancement
    enhancement_level = "none"
    if ENHANCED_PROCESSING_AVAILABLE and original_image_path and os.path.exists(original_image_path):
        if avoid_over_enhancement:
            # Check if we should skip enhancement
            enhancement_tracker = get_enhancement_tracker()
            should_skip, reason = enhancement_tracker.should_skip_enhancement(
                video_path, "moderate"
            )
            
            if should_skip:
                logger.info(f"🎯 CHAIN {chain_number}: Skipping enhancement - {reason}")
                return _extract_frame_without_enhancement(video_path, output_dir, chain_number)
            else:
                logger.info(f"🎯 CHAIN {chain_number}: Applying controlled enhancement - {reason}")
                enhancement_level = "light"  # Use lighter enhancement for chain frames
        
        try:
            logger.info(f"🎯 CHAIN {chain_number}: Using enhanced quality preservation")
            quality_preserver = ChainQualityPreserver(original_image_path)
            enhanced_frame_path, video_path_returned = extract_optimal_frame_with_preservation(
                video_path, output_dir, quality_preserver
            )
            
            # Record enhancement
            enhancement_tracker = get_enhancement_tracker()
            enhancement_tracker.record_enhancement(enhanced_frame_path, enhancement_level, "extracted_frame")
            
            return enhanced_frame_path, video_path_returned
            
        except Exception as e:
            logger.warning(f"🎯 CHAIN {chain_number}: Enhanced processing failed: {str(e)}, using standard")
    
    # Fallback to standard processing with light enhancement
    return _extract_high_quality_frame_standard(video_path, output_dir, chain_number, enhancement_level="light")

def _extract_frame_without_enhancement(video_path: str, output_dir: str, chain_number: int) -> Tuple[str, str]:
    """Extract frame without any enhancement to prevent over-processing"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_path = os.path.join(output_dir, f"{base_name}_chain{chain_number}_raw.png")
    
    logger.info(f"🎯 CHAIN {chain_number}: Raw extraction (no enhancement)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Extract from 90% of the video (avoiding potential end artifacts)
    target_frame = max(0, min(frame_count - 5, int(frame_count * 0.9)))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame from video: {video_path}")
    
    # Save as raw PNG (no enhancement)
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    logger.info(f"🎯 CHAIN {chain_number}: Raw frame extracted: {frame_path}")
    
    # Record as no enhancement
    enhancement_tracker = get_enhancement_tracker()
    enhancement_tracker.record_enhancement(frame_path, "none", "extracted_frame")
    
    return frame_path, video_path

def _extract_high_quality_frame_standard(video_path: str, output_dir: str, 
                                        chain_number: int, enhancement_level: str = "light") -> Tuple[str, str]:
    """Standard high-quality frame extraction with controlled enhancement"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_frame_path = os.path.join(output_dir, f"{base_name}_chain{chain_number}_raw.png")
    enhanced_frame_path = os.path.join(output_dir, f"{base_name}_chain{chain_number}_enhanced.png")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Extract frames from the last 15% of the video, avoiding the very last frames
    start_frame = max(0, int(frame_count * 0.85))
    end_frame = max(start_frame + 1, frame_count - 3)
    
    best_frame = None
    best_score = -1
    best_position = -1
    
    logger.info(f"🎯 CHAIN {chain_number}: Analyzing frames {start_frame} to {end_frame-1}")
    
    for pos in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Calculate quality score
        score = _calculate_frame_quality_score(frame)
        
        if score > best_score:
            best_score = score
            best_frame = frame
            best_position = pos
    
    cap.release()
    
    if best_frame is None:
        raise ValueError("Could not find suitable frame for extraction")
    
    logger.info(f"🎯 CHAIN {chain_number}: Selected frame {best_position}/{frame_count-1} (score: {best_score:.2f})")
    
    # Save raw frame
    cv2.imwrite(raw_frame_path, best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # Apply controlled enhancement based on level
    if enhancement_level == "none":
        final_path = raw_frame_path
    else:
        final_path = _apply_controlled_enhancement(raw_frame_path, enhanced_frame_path, enhancement_level)
    
    # Record enhancement
    enhancement_tracker = get_enhancement_tracker()
    enhancement_tracker.record_enhancement(final_path, enhancement_level, "extracted_frame")
    
    return final_path, video_path

def _apply_controlled_enhancement(input_path: str, output_path: str, level: str = "light") -> str:
    """Apply controlled enhancement to prevent over-processing"""
    try:
        from PIL import Image, ImageEnhance
        
        # Load image
        img = Image.open(input_path).convert("RGB")
        img_array = np.array(img)
        
        # Convert to OpenCV for processing
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply enhancement based on level
        if level == "light":
            # Very light enhancement for chain frames
            img_cv = _apply_light_enhancement(img_cv)
        elif level == "moderate":
            # Moderate enhancement (only for first enhancement)
            img_cv = _apply_moderate_enhancement(img_cv)
        elif level == "heavy":
            # Heavy enhancement (rarely used)
            img_cv = _apply_heavy_enhancement(img_cv)
        
        # Convert back to PIL and save
        final_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        final_img.save(output_path, format='PNG', optimize=False, compress_level=0)
        
        logger.info(f"Controlled enhancement applied ({level}): {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Controlled enhancement failed: {str(e)}")
        return input_path

def _apply_light_enhancement(img: np.ndarray) -> np.ndarray:
    """Apply very light enhancement for chain continuity"""
    # Very subtle sharpening
    kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    result = cv2.addWeighted(img, 0.8, sharpened, 0.2, 0)
    
    # Very light saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)  # 5% saturation boost
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result

def _apply_moderate_enhancement(img: np.ndarray) -> np.ndarray:
    """Apply moderate enhancement for first-time processing"""
    # Moderate sharpening
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    result = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
    
    # Moderate saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # 10% saturation boost
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result

def _apply_heavy_enhancement(img: np.ndarray) -> np.ndarray:
    """Apply heavy enhancement (use sparingly)"""
    # Strong sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    result = cv2.addWeighted(img, 0.6, sharpened, 0.4, 0)
    
    # Strong saturation boost
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)  # 15% saturation boost
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result

def _calculate_frame_quality_score(frame: np.ndarray) -> float:
    """Calculate comprehensive quality score for frame selection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Contrast
    contrast = np.std(gray)
    
    # Brightness balance
    brightness = np.mean(gray)
    brightness_balance = 1.0 - abs((brightness - 127.5) / 127.5)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Color richness
    b, g, r = cv2.split(frame)
    color_variance = (np.var(r) + np.var(g) + np.var(b)) / 3
    
    # Noise estimation (lower is better)
    noise_std = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
    noise_score = max(0, 1.0 - (noise_std / 20.0))
    
    # Combined quality score
    quality_score = (
        (sharpness / 1000) * 0.25 +      # Sharpness
        (contrast / 100) * 0.2 +         # Contrast  
        brightness_balance * 0.2 +       # Brightness balance
        (edge_density * 100) * 0.15 +    # Edge detail
        (color_variance / 1000) * 0.1 +  # Color richness
        noise_score * 0.1                # Low noise
    )
    
    return quality_score

def extract_simple_last_frame(video_path: str, output_dir: str, 
                             chain_number: int = 1,
                             avoid_over_enhancement: bool = True) -> Tuple[str, str]:
    """
    Simple frame extraction with over-enhancement prevention.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    last_frame_path = os.path.join(output_dir, f"{base_name}_chain{chain_number}_simple.png")
    
    logger.info(f"⚡ CHAIN {chain_number}: Simple frame extraction from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Get frame from 95% through the video (avoids potential end padding)
    target_frame = max(0, min(frame_count - 5, int(frame_count * 0.95)))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame from video: {video_path}")
    
    # Check if we should avoid enhancement
    enhancement_level = "none"
    if not avoid_over_enhancement:
        enhancement_level = "light"
        frame = _apply_light_enhancement(frame)
    
    # Save as lossless PNG
    cv2.imwrite(last_frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    logger.info(f"⚡ CHAIN {chain_number}: Frame extracted: {last_frame_path} (frame {target_frame}/{frame_count-1})")
    
    # Record enhancement
    enhancement_tracker = get_enhancement_tracker()
    enhancement_tracker.record_enhancement(last_frame_path, enhancement_level, "extracted_frame")
    
    return last_frame_path, video_path

# Rest of the functions remain the same but with controlled enhancement...
# [Previous functions like auto_adjust_image, stitch_videos, etc. continue here]

def auto_adjust_image(image_path: str, output_path: Optional[str] = None, 
                      brightness_threshold_low: int = 90, brightness_threshold_high: int = 210, 
                      saturation_threshold: float = 0.7, contrast_enhance_factor: float = 1.1,
                      force_adjustment: bool = False) -> Tuple[str, list]:
    """
    Enhanced auto-adjustment with over-enhancement prevention.
    """
    # Check if we should skip enhancement
    if not force_adjustment:
        enhancement_tracker = get_enhancement_tracker()
        should_skip, reason = enhancement_tracker.should_skip_enhancement(image_path, "moderate")
        
        if should_skip:
            logger.info(f"Skipping auto-adjustment: {reason}")
            return image_path, []
    
    try:
        img = Image.open(image_path).convert("RGB")
        adjustments = []
        
        # Convert to numpy for analysis
        img_array = np.array(img)
        
        # Analyze in HSV color space for better color handling
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # 1. Brightness analysis and adjustment
        brightness = np.mean(hsv[:, :, 2])  # V channel in HSV
        logger.info(f"Image brightness (HSV V): {brightness}")
        
        if brightness < brightness_threshold_low:
            # Increase brightness while preserving color relationships
            brightness_factor = min(1.3, 1.0 + (brightness_threshold_low - brightness) / 150)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
            adjustments.append(f"increased brightness by {(brightness_factor-1)*100:.0f}%")
            
            # Compensate saturation to prevent washing out
            saturation_compensation = min(1.1, 1.0 + (brightness_factor - 1) * 0.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_compensation, 0, 255)
            adjustments.append(f"preserved color saturation (+{(saturation_compensation-1)*100:.0f}%)")
            
        elif brightness > brightness_threshold_high:
            brightness_factor = max(0.8, 1.0 - (brightness - brightness_threshold_high) / 200)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
            adjustments.append(f"decreased brightness by {(1-brightness_factor)*100:.0f}%")
        
        # 2. Saturation analysis and adjustment
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        logger.info(f"Image saturation: {saturation}")
        
        if saturation > saturation_threshold:
            saturation_factor = max(0.7, 1.0 - (saturation - saturation_threshold) * 0.8)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            adjustments.append(f"reduced oversaturation by {(1-saturation_factor)*100:.0f}%")
        
        # Convert back to RGB
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        img = Image.fromarray(img_array)
        
        # 3. Contrast enhancement if needed
        gray_array = np.array(img.convert("L"))
        hist_std = np.std(gray_array)
        
        if hist_std < 45:  # Low contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_enhance_factor)
            adjustments.append(f"increased contrast by {(contrast_enhance_factor-1)*100:.0f}%")
        
        # Save adjusted image if adjustments were made
        if not adjustments:
            logger.info("No adjustments needed for the image")
            return image_path, []
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_adjusted{ext}"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, quality=95)
        
        # Record enhancement
        enhancement_tracker = get_enhancement_tracker()
        enhancement_tracker.record_enhancement(output_path, "moderate", "auto_adjusted")
        
        logger.info(f"Enhanced auto-adjusted image saved: {output_path}")
        return output_path, adjustments
        
    except Exception as e:
        logger.exception(f"Error in enhanced auto-adjustment: {str(e)}")
        return image_path, []

def ensure_video_compatibility(video_path: str, force_convert: bool = False) -> str:
    """Enhanced video compatibility with better codec settings"""
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return video_path
    
    if FFPROBE_AVAILABLE and not force_convert:
        return video_path
    
    try:
        output_path = video_path.replace('.mp4', '_compatible.mp4')
        
        # Use high-quality settings for compatibility conversion
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("OpenCV failed to open video")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use higher quality codec settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise Exception("OpenCV VideoWriter failed")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply light enhancement during conversion
            frame = _apply_light_enhancement(frame)
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if frame_count > 0 and os.path.exists(output_path):
            logger.info(f"Created enhanced compatible video: {output_path}")
            return output_path
        else:
            return video_path
            
    except Exception as e:
        logger.debug(f"Compatibility conversion failed: {e}")
        return video_path

def stitch_videos(video_paths: list, output_path: str) -> str:
    """Enhanced video stitching with quality preservation"""
    if not video_paths:
        raise ValueError("No video paths provided")
        
    logger.info(f"🎬 QUALITY STITCHING: Combining {len(video_paths)} videos into: {output_path}")
    
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if FFMPEG_AVAILABLE:
            logger.info("Using FFmpeg with high-quality settings for concatenation")
            
            list_file_path = output_path.replace(".mp4", "_list.txt")
            
            with open(list_file_path, 'w') as f:
                for path in video_paths:
                    abs_path = os.path.abspath(path).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")
            
            working_dir = os.path.dirname(list_file_path)
            
            # Use high-quality encoding settings
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", os.path.basename(list_file_path),
                "-c:v", "libx264",
                "-preset", "slow",  # Better quality
                "-crf", "18",       # High quality
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                os.path.basename(output_path)
            ]
            
            try:
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=working_dir
                )
                
                # Clean up
                try:
                    os.remove(list_file_path)
                except:
                    pass
                
                logger.info(f"High-quality video stitching completed: {output_path}")
                return output_path
                
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg high-quality stitching failed: {e.stderr}")
                
                # Fallback to copy method if encoding fails
                logger.info("Trying copy method...")
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", os.path.basename(list_file_path), "-c", "copy", 
                    os.path.basename(output_path)
                ]
                
                subprocess.run(cmd, check=True, cwd=working_dir)
                try:
                    os.remove(list_file_path)
                except:
                    pass
                return output_path
        
        # Fallback to moviepy with quality settings
        logger.info("Using moviepy with quality preservation")
        clips = []
        
        try:
            for path in video_paths:
                clip = VideoFileClip(path)
                clips.append(clip)
                
            final_clip = concatenate_videoclips(clips)
            
            # Write with high quality settings
            final_clip.write_videofile(
                output_path, 
                codec="libx264",
                audio_codec="aac",
                bitrate="8000k",  # High bitrate for quality
                verbose=False, 
                logger=None
            )
            
            for clip in clips:
                clip.close()
                
            logger.info(f"High-quality moviepy stitching completed: {output_path}")
            return output_path
                
        except Exception as e:
            logger.exception(f"Moviepy stitching failed: {str(e)}")
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            
            # Final fallback
            if video_paths:
                import shutil
                shutil.copy2(video_paths[0], output_path)
                logger.warning(f"All stitching methods failed, using first video: {output_path}")
                return output_path
            else:
                raise ValueError("No videos to stitch")
            
    except Exception as e:
        logger.error(f"Error in enhanced video stitching: {str(e)}")
        raise

# Backwards compatibility functions
def extract_last_frame(video_path: str, output_path: Optional[str] = None, 
                      chain_number: int = 1) -> str:
    """Backwards compatibility wrapper with chain awareness"""
    if output_path is None:
        output_dir = os.path.dirname(video_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_last_frame.png")
    
    frame_path, _ = extract_simple_last_frame(
        video_path, 
        os.path.dirname(output_path), 
        chain_number=chain_number,
        avoid_over_enhancement=True
    )
    return frame_path

def extract_best_frame(video_path: str, output_path: Optional[str] = None, 
                      chain_number: int = 1) -> str:
    """Backwards compatibility wrapper with chain awareness"""
    if output_path is None:
        output_dir = os.path.dirname(video_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_best_frame.png")
    
    frame_path, _ = extract_high_quality_frame_for_chain(
        video_path, 
        os.path.dirname(output_path),
        chain_number=chain_number,
        avoid_over_enhancement=True
    )
    return frame_path

# Chain management helper functions
def reset_chain_state():
    """Reset chain state for new generation session"""
    get_chain_state_manager().reset()
    get_enhancement_tracker().enhancement_history.clear()
    logger.info("🔄 Chain state and enhancement history reset")

def validate_chain_input(chain_number: int, proposed_image_path: str) -> Tuple[bool, str, str]:
    """
    Validate and potentially correct chain input to prevent KLING re-use issue
    
    Returns:
        Tuple of (is_valid, corrected_path, message)
    """
    chain_state = get_chain_state_manager()
    is_valid, message = chain_state.validate_chain_progression(chain_number, proposed_image_path)
    
    if not is_valid:
        # Return the correct image path
        correct_path = chain_state.get_current_input_image()
        logger.warning(f"🚨 CHAIN VALIDATION FAILED: {message}")
        logger.info(f"🔧 CORRECTING: Using {correct_path} instead of {proposed_image_path}")
        return False, correct_path, f"Corrected chain input: {message}"
    else:
        return True, proposed_image_path, message

def log_chain_debug_info():
    """Log debug information about current chain state"""
    chain_state = get_chain_state_manager()
    debug_info = chain_state.get_debug_info()
    
    logger.info("🔍 CHAIN DEBUG INFO:")
    logger.info(f"  Current Chain: {debug_info['chain_count']}")
    logger.info(f"  Original Image: {debug_info['original_image']}")
    logger.info(f"  Current Image: {debug_info['current_image']}")
    logger.info(f"  Last Video: {debug_info['last_video']}")
    logger.info(f"  Last Frame: {debug_info['last_extracted_frame']}")
    
    if debug_info['image_progression']:
        logger.info("  Image Progression:")
        for prog in debug_info['image_progression']:
            logger.info(f"    Chain {prog['chain']}: {prog['type']} -> {os.path.basename(prog['path'])}")

# Export the state managers for use in gradio_ui.py
__all__ = [
    'extract_high_quality_frame_for_chain',
    'extract_simple_last_frame', 
    'auto_adjust_image',
    'stitch_videos',
    'ensure_video_compatibility',
    'extract_last_frame',
    'extract_best_frame',
    'reset_chain_state',
    'validate_chain_input',
    'log_chain_debug_info',
    'get_chain_state_manager',
    'get_enhancement_tracker',
    'FFMPEG_AVAILABLE',
    'FFPROBE_AVAILABLE'
]