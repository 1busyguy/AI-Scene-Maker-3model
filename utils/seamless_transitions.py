# utils/seamless_transitions.py - Complete seamless transition system

import cv2
import numpy as np
import os
import logging
from typing import Tuple, Optional, List
from PIL import Image, ImageFilter, ImageEnhance
import subprocess

logger = logging.getLogger(__name__)

class SeamlessTransitionManager:
    """
    Complete solution for seamless video transitions
    """
    
    def __init__(self):
        self.frame_buffer = []  # Store last few frames for analysis
        self.motion_vectors = []  # Track motion between frames
        
    def extract_perfect_transition_frame(self, video_path: str, output_dir: str, 
                                       chain_number: int = 1) -> Tuple[str, str, dict]:
        """
        Extract the absolute last frame for perfect transitions
        
        Returns:
            Tuple of (frame_path, trimmed_video_path, motion_data)
        """
        logger.info(f"🎯 PERFECT EXTRACTION: Chain {chain_number}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get the ABSOLUTE LAST FRAME (not 95% or 90%)
        last_frame_idx = frame_count - 1
        
        logger.info(f"🎯 Extracting frame {last_frame_idx}/{frame_count-1} (ABSOLUTE LAST)")
        
        # Extract last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
        ret, last_frame = cap.read()
        
        if not ret:
            # Fallback: try frame -2, -3, etc.
            for offset in range(2, 6):
                try_frame = max(0, last_frame_idx - offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, try_frame)
                ret, last_frame = cap.read()
                if ret:
                    last_frame_idx = try_frame
                    logger.warning(f"🎯 Used frame {try_frame} instead of absolute last")
                    break
        
        if not ret:
            cap.release()
            raise ValueError("Cannot extract any usable frame")
        
        # Also extract previous frame for motion analysis
        prev_frame = None
        if last_frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx - 1)
            ret_prev, prev_frame = cap.read()
            if not ret_prev:
                prev_frame = None
        
        cap.release()
        
        # Analyze motion between frames
        motion_data = self._analyze_frame_motion(prev_frame, last_frame)
        
        # Save the last frame
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join(output_dir, f"{base_name}_absolute_last_frame.png")
        
        # Apply transition-optimized processing
        optimized_frame = self._optimize_frame_for_transition(last_frame, motion_data)
        cv2.imwrite(frame_path, optimized_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Create trimmed video that ends EXACTLY at the extracted frame
        trimmed_video_path = self._create_exact_trimmed_video(
            video_path, output_dir, last_frame_idx, base_name
        )
        
        logger.info(f"🎯 PERFECT FRAME: {frame_path}")
        logger.info(f"🎯 TRIMMED VIDEO: {trimmed_video_path}")
        
        return frame_path, trimmed_video_path, motion_data
    
    def _analyze_frame_motion(self, prev_frame: Optional[np.ndarray], 
                             current_frame: np.ndarray) -> dict:
        """
        Analyze motion between consecutive frames
        """
        motion_data = {
            'has_motion': False,
            'motion_vectors': [],
            'dominant_direction': 'static',
            'motion_magnitude': 0.0,
            'optical_flow': None
        }
        
        if prev_frame is None:
            return motion_data
        
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, 
                np.array([[]], dtype=np.float32), None
            )
            
            # Dense optical flow for better analysis
            flow_dense = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion statistics
            magnitude, angle = cv2.cartToPolar(flow_dense[..., 0], flow_dense[..., 1])
            
            motion_data['has_motion'] = np.mean(magnitude) > 0.5
            motion_data['motion_magnitude'] = float(np.mean(magnitude))
            motion_data['optical_flow'] = flow_dense
            
            # Determine dominant motion direction
            if motion_data['has_motion']:
                mean_angle = np.mean(angle)
                if mean_angle < np.pi/4 or mean_angle > 7*np.pi/4:
                    motion_data['dominant_direction'] = 'right'
                elif np.pi/4 <= mean_angle < 3*np.pi/4:
                    motion_data['dominant_direction'] = 'down'
                elif 3*np.pi/4 <= mean_angle < 5*np.pi/4:
                    motion_data['dominant_direction'] = 'left'
                else:
                    motion_data['dominant_direction'] = 'up'
            
            logger.info(f"🎯 MOTION ANALYSIS: {motion_data['dominant_direction']}, magnitude: {motion_data['motion_magnitude']:.2f}")
            
        except Exception as e:
            logger.error(f"Motion analysis failed: {e}")
        
        return motion_data
    
    def _optimize_frame_for_transition(self, frame: np.ndarray, motion_data: dict) -> np.ndarray:
        """
        Optimize frame for seamless transitions
        """
        optimized = frame.copy()
        
        try:
            # 1. Apply motion-based stabilization
            if motion_data.get('has_motion', False):
                # Slight motion blur in the direction of movement to help transitions
                direction = motion_data.get('dominant_direction', 'static')
                
                if direction == 'right':
                    kernel = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
                elif direction == 'left':
                    kernel = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
                elif direction == 'down':
                    kernel = np.array([[0.1], [0.2], [0.4], [0.2], [0.1]])
                elif direction == 'up':
                    kernel = np.array([[0.1], [0.2], [0.4], [0.2], [0.1]])
                else:
                    kernel = None
                
                if kernel is not None:
                    optimized = cv2.filter2D(optimized, -1, kernel)
            
            # 2. Temporal smoothing to reduce frame-to-frame variations
            optimized = cv2.bilateralFilter(optimized, 5, 50, 50)
            
            # 3. Slight edge enhancement for consistency
            gray = cv2.cvtColor(optimized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((2,2), np.uint8))
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Blend edges back into image (very subtle)
            optimized = cv2.addWeighted(optimized, 0.95, edges_colored, 0.05, 0)
            
        except Exception as e:
            logger.error(f"Frame optimization failed: {e}")
            return frame
        
        return optimized
    
    def _create_exact_trimmed_video(self, input_video: str, output_dir: str, 
                                   end_frame: int, base_name: str) -> str:
        """
        Create video that ends EXACTLY at the specified frame
        """
        trimmed_path = os.path.join(output_dir, f"{base_name}_exact_trim.mp4")
        
        try:
            # Method 1: FFmpeg frame-accurate trimming
            if self._is_ffmpeg_available():
                logger.info("🎯 Using FFmpeg for exact trimming")
                
                # Calculate exact time
                cap = cv2.VideoCapture(input_video)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # End time = (end_frame + 1) / fps
                end_time = (end_frame + 1) / fps
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", input_video,
                    "-t", f"{end_time:.6f}",
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "18",
                    "-avoid_negative_ts", "make_zero",
                    trimmed_path
                ]
                
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                    text=True, timeout=60
                )
                
                if result.returncode == 0 and os.path.exists(trimmed_path):
                    logger.info(f"🎯 FFmpeg exact trim successful: {end_frame} frames")
                    return trimmed_path
                else:
                    logger.error(f"FFmpeg failed: {result.stderr}")
            
            # Method 2: OpenCV frame-by-frame (fallback)
            logger.info("🎯 Using OpenCV for exact trimming")
            
            cap = cv2.VideoCapture(input_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(trimmed_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
            if os.path.exists(trimmed_path):
                logger.info(f"🎯 OpenCV exact trim successful: {frame_idx} frames")
                return trimmed_path
            
        except Exception as e:
            logger.error(f"Exact trimming failed: {e}")
        
        # Fallback: return original
        logger.warning("🎯 Using original video (trimming failed)")
        return input_video
    
    def create_transition_blend_frame(self, end_frame_path: str, start_frame_path: str, 
                                    output_path: str, blend_strength: float = 0.15) -> str:
        """
        Create a blended frame to smooth transitions between videos
        """
        try:
            end_img = cv2.imread(end_frame_path)
            start_img = cv2.imread(start_frame_path)
            
            if end_img is None or start_img is None:
                logger.error("Cannot load transition frames")
                return start_frame_path
            
            # Resize to match if needed
            if end_img.shape != start_img.shape:
                start_img = cv2.resize(start_img, (end_img.shape[1], end_img.shape[0]))
            
            # Create blend
            blended = cv2.addWeighted(
                end_img, 1 - blend_strength,
                start_img, blend_strength,
                0
            )
            
            cv2.imwrite(output_path, blended, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            logger.info(f"🎯 Transition blend created: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Transition blend failed: {e}")
            return start_frame_path
    
    def _is_ffmpeg_available(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(["ffmpeg", "-version"], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            return True
        except:
            return False
    
    def create_crossfade_transition(self, video1_path: str, video2_path: str, 
                                  output_path: str, crossfade_duration: float = 0.5) -> str:
        """
        Create a crossfade transition between two videos
        """
        try:
            if not self._is_ffmpeg_available():
                logger.warning("FFmpeg not available for crossfade")
                return video2_path
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video1_path,
                "-i", video2_path,
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:offset=0[v]",
                "-map", "[v]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                output_path
            ]
            
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True, timeout=120
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"🎯 Crossfade transition created: {output_path}")
                return output_path
            else:
                logger.error(f"Crossfade failed: {result.stderr}")
                return video2_path
                
        except Exception as e:
            logger.error(f"Crossfade transition failed: {e}")
            return video2_path


# Global instance
_transition_manager = None

def get_transition_manager() -> SeamlessTransitionManager:
    """Get global transition manager"""
    global _transition_manager
    if _transition_manager is None:
        _transition_manager = SeamlessTransitionManager()
    return _transition_manager


def extract_absolute_last_frame(video_path: str, output_dir: str, chain_number: int = 1) -> Tuple[str, str, dict]:
    """
    Extract the absolute last frame for perfect transitions
    """
    manager = get_transition_manager()
    return manager.extract_perfect_transition_frame(video_path, output_dir, chain_number)


def create_seamless_video_chain(video_paths: List[str], output_path: str, 
                               use_crossfade: bool = True) -> str:
    """
    Create seamless video chain with transition optimization
    """
    if len(video_paths) < 2:
        return video_paths[0] if video_paths else ""
    
    manager = get_transition_manager()
    
    if use_crossfade and manager._is_ffmpeg_available():
        # Use crossfade transitions
        logger.info("🎯 Creating seamless chain with crossfades")
        
        current_video = video_paths[0]
        
        for i, next_video in enumerate(video_paths[1:], 1):
            temp_output = output_path.replace('.mp4', f'_temp_{i}.mp4')
            current_video = manager.create_crossfade_transition(
                current_video, next_video, temp_output, crossfade_duration=0.3
            )
        
        # Move final result
        if current_video != output_path:
            os.rename(current_video, output_path)
        
        return output_path
    
    else:
        # Fallback to concatenation
        logger.info("🎯 Creating seamless chain with concatenation")
        from .video_processing_v2 import stitch_videos
        return stitch_videos(video_paths, output_path)