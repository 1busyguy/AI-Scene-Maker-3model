# utils/face_enhancement.py
"""
Face Enhancement Module
Post-processing face enhancement and restoration for video frames
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import tempfile
import shutil
from utils.face_swapping import FaceSwapper, EnhancedVideoFaceConsistencyEnhancer

logger = logging.getLogger(__name__)

class FaceEnhancer:
    """Handles face enhancement and restoration for video frames"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize face enhancement models
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        self.face_enhancer = None
        self.bg_upsampler = None
        self.model_loaded = False
        
        # Model paths (will be downloaded automatically)
        self.model_path = 'gfpgan/weights/GFPGANv1.4.pth'
        self.bg_model_path = 'realesrgan/weights/RealESRGAN_x4plus.pth'
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize GFPGAN and RealESRGAN models"""
        try:
            # Initialize RealESRGAN for background
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=4)
            
            self.bg_upsampler = RealESRGANer(
                scale=4,
                model_path=self.bg_model_path,
                dni_weight=None,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == 'cuda' else False,
                device=self.device
            )
            
            # Initialize GFPGAN
            self.face_enhancer = GFPGANer(
                model_path=self.model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.bg_upsampler,
                device=self.device
            )
            
            self.model_loaded = True
            logger.info(f"Face enhancement models initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing face enhancement models: {str(e)}")
            logger.info("Models will be downloaded on first use")
            self.model_loaded = False
    
    def _apply_light_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply light enhancement that's more API-compatible
        Uses OpenCV-based enhancement instead of heavy AI models
        """
        try:
            # Convert to float for processing
            img_float = img.astype(np.float32) / 255.0
            
            # Subtle contrast enhancement
            contrast_factor = 1.1
            img_float = np.clip((img_float - 0.5) * contrast_factor + 0.5, 0, 1)
            
            # Gentle brightness adjustment
            brightness_factor = 1.05
            img_float = np.clip(img_float * brightness_factor, 0, 1)
            
            # Light sharpening using unsharp mask
            gaussian_blur = cv2.GaussianBlur(img_float, (5, 5), 1.0)
            unsharp_mask = cv2.addWeighted(img_float, 1.5, gaussian_blur, -0.5, 0)
            img_float = np.clip(unsharp_mask, 0, 1)
            
            # Noise reduction
            img_float = cv2.bilateralFilter(img_float, 5, 0.1, 0.1)
            
            # Convert back to uint8
            enhanced_img = (img_float * 255).astype(np.uint8)
            
            logger.info("Applied light enhancement")
            return enhanced_img
            
        except Exception as e:
            logger.error(f"Error in light enhancement: {str(e)}")
            return img
    
    def enhance_face(self, image_path: str, output_path: Optional[str] = None, 
                    enhance_background: bool = True, light_mode: bool = False) -> str:
        """
        Enhance faces in a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image (optional)
            enhance_background: Whether to enhance the background as well
            light_mode: Use lighter enhancement that's more API-compatible
            
        Returns:
            Path to enhanced image
        """
        if not self.model_loaded:
            logger.warning("Face enhancement models not loaded. Attempting to initialize...")
            self._initialize_models()
            if not self.model_loaded:
                logger.error("Failed to load models. Returning original image.")
                return image_path
        
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            if light_mode:
                # Light enhancement mode - API-friendly
                restored_img = self._apply_light_enhancement(img)
            else:
                # Full GFPGAN enhancement
                cropped_faces, restored_faces, restored_img = self.face_enhancer.enhance(
                    img, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True,
                    weight=0.5  # Blend weight for more natural results
                )
                
                # If no faces detected, just enhance the whole image
                if restored_img is None:
                    if enhance_background and self.bg_upsampler:
                        restored_img, _ = self.bg_upsampler.enhance(img, outscale=2)
                    else:
                        restored_img = img
            
            # Convert back to BGR for OpenCV
            if restored_img.shape[2] == 4:  # RGBA
                restored_img = restored_img[:, :, :3]
            
            # Save enhanced image
            if output_path is None:
                # Properly construct output path
                base_path, ext = os.path.splitext(image_path)
                output_path = f"{base_path}_enhanced{ext}"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, restored_img)
            logger.info(f"Enhanced face saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error enhancing face: {str(e)}")
            return image_path
    
    def enhance_video(self, video_path: str, output_path: Optional[str] = None,
                     skip_frames: int = 1, batch_size: int = 4) -> str:
        """
        Enhance faces in an entire video
        
        Args:
            video_path: Path to input video
            output_path: Path to save enhanced video
            skip_frames: Process every Nth frame (1 = process all frames)
            batch_size: Number of frames to process in parallel
            
        Returns:
            Path to enhanced video
        """
        if not self.model_loaded:
            logger.warning("Face enhancement models not loaded. Returning original video.")
            return video_path
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prepare output
            if output_path is None:
                base_path, ext = os.path.splitext(video_path)
                output_path = f"{base_path}_enhanced{ext}"
            
            # Create local temporary directory for frames to avoid Windows permission issues
            video_dir = os.path.dirname(os.path.abspath(video_path))
            local_temp_dir = os.path.join(video_dir, ".temp_face_enhancement")
            os.makedirs(local_temp_dir, exist_ok=True)
            
            # Create subdirectory for enhanced frames
            enhanced_frames_dir = os.path.join(local_temp_dir, 'enhanced')
            os.makedirs(enhanced_frames_dir, exist_ok=True)
            
            logger.info(f"Processing {total_frames} frames from video...")
            
            # Process frames
            frame_count = 0
            frames_to_process = []
            frame_indices = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    frames_to_process.append(frame)
                    frame_indices.append(frame_count)
                    
                    # Process batch
                    if len(frames_to_process) >= batch_size:
                        self._process_frame_batch(
                            frames_to_process, frame_indices, enhanced_frames_dir
                        )
                        frames_to_process = []
                        frame_indices = []
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames...")
            
            # Process remaining frames
            if frames_to_process:
                self._process_frame_batch(
                    frames_to_process, frame_indices, enhanced_frames_dir
                )
            
            cap.release()
            
            # Reassemble video
            logger.info("Reassembling enhanced video...")
            self._create_video_from_frames(
                enhanced_frames_dir, output_path, fps, (width, height)
            )
            
            # Cleanup local temp directory
            try:
                shutil.rmtree(local_temp_dir)
                logger.debug(f"Cleaned up temp directory: {local_temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp directory {local_temp_dir}: {cleanup_error}")
            
            logger.info(f"Enhanced video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error enhancing video: {str(e)}")
            return video_path
    
    def _process_frame_batch(self, frames: List[np.ndarray], indices: List[int], 
                           output_dir: str):
        """Process a batch of frames"""
        for frame, idx in zip(frames, indices):
            try:
                # Enhance frame
                _, _, restored_img = self.face_enhancer.enhance(
                    frame, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True,
                    weight=0.5
                )
                
                if restored_img is None:
                    restored_img = frame
                
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(frame_path, restored_img)
                
            except Exception as e:
                logger.warning(f"Error processing frame {idx}: {str(e)}")
                # Save original frame on error
                frame_path = os.path.join(output_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(frame_path, frame)
    
    def _create_video_from_frames(self, frames_dir: str, output_path: str, 
                                fps: int, size: Tuple[int, int]):
        """Create video from enhanced frames"""
        import subprocess
        
        # Use ffmpeg to create video
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def enhance_character_region(self, image_path: str, face_region: dict,
                               output_path: Optional[str] = None) -> str:
        """
        Enhance only the character/face region of an image
        
        Args:
            image_path: Path to input image
            face_region: Dictionary with 'x', 'y', 'w', 'h' keys
            output_path: Path to save enhanced image
            
        Returns:
            Path to enhanced image
        """
        try:
            img = cv2.imread(image_path)
            
            # Extract face region with padding
            padding = 50
            x = max(0, face_region['x'] - padding)
            y = max(0, face_region['y'] - padding)
            w = face_region['w'] + 2 * padding
            h = face_region['h'] + 2 * padding
            
            # Ensure we don't exceed image boundaries
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            
            # Extract region
            face_img = img[y:y2, x:x2]
            
            # Enhance face region
            _, _, restored_face = self.face_enhancer.enhance(
                face_img, 
                has_aligned=False, 
                only_center_face=True, 
                paste_back=True
            )
            
            if restored_face is not None:
                # Paste back into original image
                img[y:y2, x:x2] = restored_face
            
            # Save
            if output_path is None:
                base_path, ext = os.path.splitext(image_path)
                output_path = f"{base_path}_face_enhanced{ext}"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, img)
            return output_path
            
        except Exception as e:
            logger.error(f"Error enhancing character region: {str(e)}")
            return image_path
    
    def create_consistent_face_reference(self, video_paths: List[str], 
                                       output_path: Optional[str] = None) -> str:
        """
        Create a reference image with the best/most consistent face from all videos
        
        Args:
            video_paths: List of video paths to analyze
            output_path: Path to save reference image
            
        Returns:
            Path to reference image
        """
        best_faces = []
        
        for video_path in video_paths:
            # Extract a few frames from each video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames
            sample_indices = np.linspace(0, frame_count-1, min(5, frame_count), dtype=int)
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Detect faces
                    cropped_faces, _, _ = self.face_enhancer.enhance(
                        frame, 
                        has_aligned=False, 
                        only_center_face=False, 
                        paste_back=False
                    )
                    
                    if cropped_faces is not None:
                        best_faces.extend(cropped_faces)
            
            cap.release()
        
        if best_faces:
            # Select the highest quality face
            # For simplicity, we'll use the largest face
            largest_face = max(best_faces, key=lambda x: x.shape[0] * x.shape[1])
            
            # Enhance the selected face
            _, restored_faces, _ = self.face_enhancer.enhance(
                largest_face, 
                has_aligned=True, 
                only_center_face=True, 
                paste_back=False
            )
            
            if restored_faces is not None and len(restored_faces) > 0:
                reference_face = restored_faces[0]
            else:
                reference_face = largest_face
            
            # Save reference
            if output_path is None:
                output_path = os.path.join(os.path.dirname(video_paths[0]), 
                                         "character_reference_face.png")
            
            cv2.imwrite(output_path, reference_face)
            logger.info(f"Created face reference at: {output_path}")
            
            return output_path
        
        logger.warning("No faces found in videos for reference creation")
        return None


class VideoFaceConsistencyEnhancer(EnhancedVideoFaceConsistencyEnhancer):
    """Enhanced face consistency enhancer with face swapping and original interface"""
    
    def __init__(self, reference_image_path: str):
        """
        Initialize with a reference image - maintains original interface
        
        Args:
            reference_image_path: Path to reference image with desired face
        """
        # Initialize the enhanced parent class
        super().__init__(reference_image_path)
        
        # Maintain original interface properties
        self.reference_image_path = reference_image_path
        self.face_enhancer = FaceEnhancer()
        self.reference_features = None
        
        # Extract reference features (original method)
        self._extract_reference_features()
    
    def _extract_reference_features(self):
        """Extract facial features from reference image - original method"""
        try:
            import face_recognition
            
            # Load reference image
            ref_img = face_recognition.load_image_file(self.reference_image_path)
            
            # Get face encoding
            face_locations = face_recognition.face_locations(ref_img)
            if face_locations:
                face_encodings = face_recognition.face_encodings(ref_img, face_locations)
                if face_encodings:
                    self.reference_features = face_encodings[0]
                    logger.info("Reference face features extracted successfully")
            
        except Exception as e:
            logger.error(f"Error extracting reference features: {str(e)}")
    
    def enhance_video_chain_consistency(self, video_paths: List[str], 
                                      output_dir: Optional[str] = None) -> List[str]:
        """
        Enhance face consistency across a chain of videos - original interface
        Now with enhanced functionality from parent class
        
        Args:
            video_paths: List of video paths in the chain
            output_dir: Directory to save enhanced videos
            
        Returns:
            List of enhanced video paths
        """
        if output_dir is None:
            output_dir = os.path.dirname(video_paths[0])
        
        # Try to use enhanced functionality from parent class if available
        try:
            # Check if parent class has enhanced methods
            if hasattr(super(), 'enhance_video_chain_with_swapping'):
                logger.info("Using enhanced face swapping functionality")
                return super().enhance_video_chain_with_swapping(video_paths, output_dir)
        except Exception as e:
            logger.warning(f"Enhanced functionality failed, falling back to original: {str(e)}")
        
        # Fallback to original implementation
        enhanced_paths = []
        
        # First, create a consistent face reference from all videos
        reference_face = self.face_enhancer.create_consistent_face_reference(video_paths)
        
        # Process each video
        for i, video_path in enumerate(video_paths):
            logger.info(f"Enhancing video {i+1}/{len(video_paths)} for consistency...")
            
            output_path = os.path.join(
                output_dir, 
                f"enhanced_consistent_{os.path.basename(video_path)}"
            )
            
            # Enhance with face consistency
            enhanced_path = self._enhance_video_with_reference(
                video_path, reference_face, output_path
            )
            
            enhanced_paths.append(enhanced_path)
        
        return enhanced_paths
    
    def _enhance_video_with_reference(self, video_path: str, reference_face_path: str,
                                    output_path: str) -> str:
        """Enhance video using reference face for consistency - original method"""
        # For now, use standard enhancement
        # In production, this would apply face swapping or style transfer
        return self.face_enhancer.enhance_video(video_path, output_path)