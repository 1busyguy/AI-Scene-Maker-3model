# utils/enhanced_video_processing.py
"""
Enhanced video processing module to prevent quality degradation across video chains
Focus on color preservation and detail retention
"""

import cv2
import os
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import hashlib
from typing import Tuple, Optional
import subprocess

logger = logging.getLogger(__name__)

class ChainQualityPreserver:
    """Manages quality preservation across video chain generations"""
    
    def __init__(self, original_image_path: str):
        """
        Initialize with the original reference image
        
        Args:
            original_image_path: Path to the original input image
        """
        self.original_image_path = original_image_path
        self.original_color_profile = None
        self.original_detail_profile = None
        self._analyze_original_image()
    
    def _analyze_original_image(self):
        """Analyze the original image to create reference profiles"""
        try:
            # Load original image
            original_img = cv2.imread(self.original_image_path)
            if original_img is None:
                logger.error(f"Could not load original image: {self.original_image_path}")
                return
            
            # Create color profile
            self.original_color_profile = self._extract_color_profile(original_img)
            
            # Create detail profile
            self.original_detail_profile = self._extract_detail_profile(original_img)
            
            logger.info("Original image profiles created for quality preservation")
            
        except Exception as e:
            logger.error(f"Error analyzing original image: {str(e)}")
    
    def _extract_color_profile(self, img: np.ndarray) -> dict:
        """Extract comprehensive color profile from image"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        profile = {
            # HSV statistics
            'hue_mean': np.mean(hsv[:, :, 0]),
            'hue_std': np.std(hsv[:, :, 0]),
            'saturation_mean': np.mean(hsv[:, :, 1]),
            'saturation_std': np.std(hsv[:, :, 1]),
            'value_mean': np.mean(hsv[:, :, 2]),
            'value_std': np.std(hsv[:, :, 2]),
            
            # LAB statistics
            'lightness_mean': np.mean(lab[:, :, 0]),
            'a_mean': np.mean(lab[:, :, 1]),
            'b_mean': np.mean(lab[:, :, 2]),
            
            # RGB channel correlations
            'rg_correlation': np.corrcoef(img[:, :, 2].flatten(), img[:, :, 1].flatten())[0, 1],
            'rb_correlation': np.corrcoef(img[:, :, 2].flatten(), img[:, :, 0].flatten())[0, 1],
            'gb_correlation': np.corrcoef(img[:, :, 1].flatten(), img[:, :, 0].flatten())[0, 1],
            
            # Color distribution
            'dominant_colors': self._extract_dominant_colors(img)
        }
        
        return profile
    
    def _extract_detail_profile(self, img: np.ndarray) -> dict:
        """Extract detail characteristics from image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate various detail metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture analysis using Local Binary Pattern
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            texture_variance = np.var(lbp)
        except ImportError:
            # Fallback if scikit-image not available
            texture_variance = np.var(gray)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        profile = {
            'sharpness': laplacian_var,
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'contrast': np.std(gray),
            'brightness': np.mean(gray)
        }
        
        return profile
    
    def _extract_dominant_colors(self, img: np.ndarray, k: int = 5) -> list:
        """Extract dominant colors using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = img.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)
        
        # Get the dominant colors
        colors = kmeans.cluster_centers_
        
        # Calculate the percentage of each color
        labels = kmeans.labels_
        percentages = np.bincount(labels) / len(labels)
        
        # Sort by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        
        dominant_colors = []
        for i in sorted_indices:
            dominant_colors.append({
                'color': colors[i].astype(int).tolist(),
                'percentage': float(percentages[i])
            })
        
        return dominant_colors

def extract_optimal_frame_with_preservation(video_path: str, output_dir: str, 
                                          quality_preserver: Optional[ChainQualityPreserver] = None) -> Tuple[str, str]:
    """
    Extract the optimal frame with maximum quality preservation
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save outputs
        quality_preserver: Optional quality preservation manager
        
    Returns:
        tuple: (enhanced_frame_path, original_video_path)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_frame_path = os.path.join(output_dir, f"{base_name}_raw_optimal.png")
    enhanced_frame_path = os.path.join(output_dir, f"{base_name}_preserved_frame.png")
    
    logger.info(f"QUALITY PRESERVATION: Extracting optimal frame from: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # Strategy: Analyze frames from the last 15% of the video
    # This ensures progression while avoiding potential fade/padding frames
    start_percentage = 0.85
    start_frame = max(0, int(frame_count * start_percentage))
    
    # Also avoid the very last few frames which might be padding
    end_frame = max(start_frame + 1, frame_count - 3)
    
    logger.info(f"Analyzing frames {start_frame} to {end_frame-1} for optimal extraction")
    
    best_frame = None
    best_score = -1
    best_position = -1
    
    # Analyze candidate frames
    for pos in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Calculate comprehensive quality score
        score = calculate_frame_quality_score(frame, quality_preserver)
        
        if score > best_score:
            best_score = score
            best_frame = frame
            best_position = pos
    
    cap.release()
    
    if best_frame is None:
        raise ValueError("Could not find suitable frame for extraction")
    
    logger.info(f"Selected frame at position {best_position}/{frame_count-1} with quality score: {best_score:.2f}")
    
    # Save raw frame
    cv2.imwrite(raw_frame_path, best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # Apply quality preservation enhancement
    enhanced_frame_path = apply_quality_preservation_enhancement(
        raw_frame_path, enhanced_frame_path, quality_preserver
    )
    
    return enhanced_frame_path, video_path

def calculate_frame_quality_score(frame: np.ndarray, quality_preserver: Optional[ChainQualityPreserver] = None) -> float:
    """Calculate comprehensive quality score for frame selection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Basic quality metrics
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = np.std(gray)
    brightness = np.mean(gray)
    
    # Brightness balance score (prefer well-balanced exposure)
    brightness_balance = 1.0 - abs((brightness - 127.5) / 127.5)
    
    # Color richness
    b, g, r = cv2.split(frame)
    color_variance = (np.var(r) + np.var(g) + np.var(b)) / 3
    
    # Edge detail
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Base quality score
    quality_score = (
        (sharpness / 1000) * 0.3 +       # Sharpness
        (contrast / 100) * 0.25 +        # Contrast
        brightness_balance * 0.2 +       # Brightness balance
        (color_variance / 1000) * 0.15 + # Color richness
        (edge_density * 100) * 0.1       # Edge detail
    )
    
    # If we have quality preserver, add preservation bonus
    if quality_preserver and quality_preserver.original_color_profile:
        preservation_bonus = calculate_preservation_bonus(frame, quality_preserver)
        quality_score += preservation_bonus * 0.3  # 30% weight for preservation
    
    return quality_score

def calculate_preservation_bonus(frame: np.ndarray, quality_preserver: ChainQualityPreserver) -> float:
    """Calculate bonus score for frames that preserve original qualities"""
    current_color_profile = quality_preserver._extract_color_profile(frame)
    original_profile = quality_preserver.original_color_profile
    
    # Calculate color similarity scores
    saturation_similarity = 1.0 - abs(
        current_color_profile['saturation_mean'] - original_profile['saturation_mean']
    ) / 255.0
    
    hue_similarity = 1.0 - abs(
        current_color_profile['hue_mean'] - original_profile['hue_mean']
    ) / 180.0
    
    value_similarity = 1.0 - abs(
        current_color_profile['value_mean'] - original_profile['value_mean']
    ) / 255.0
    
    # Overall preservation score
    preservation_score = (saturation_similarity + hue_similarity + value_similarity) / 3
    
    return preservation_score

def apply_quality_preservation_enhancement(input_path: str, output_path: str, 
                                         quality_preserver: Optional[ChainQualityPreserver] = None) -> str:
    """
    Apply enhancement that preserves original image qualities
    """
    try:
        # Load image
        img = Image.open(input_path).convert("RGB")
        img_array = np.array(img)
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply targeted enhancements based on original image analysis
        if quality_preserver and quality_preserver.original_color_profile:
            img_cv = apply_color_preservation(img_cv, quality_preserver)
        
        # Apply detail preservation
        img_cv = apply_detail_preservation(img_cv)
        
        # Apply noise reduction while preserving detail
        img_cv = apply_smart_denoising(img_cv)
        
        # Convert back to PIL format
        img_final = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Save with maximum quality
        img_final.save(output_path, format='PNG', optimize=False, compress_level=0)
        
        logger.info(f"Quality preservation enhancement applied: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Enhancement failed: {str(e)}, using original")
        return input_path

def apply_color_preservation(img: np.ndarray, quality_preserver: ChainQualityPreserver) -> np.ndarray:
    """Apply color corrections to match original color profile"""
    original_profile = quality_preserver.original_color_profile
    current_profile = quality_preserver._extract_color_profile(img)
    
    # Convert to HSV for color adjustments
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    
    # Adjust saturation to match original
    saturation_ratio = original_profile['saturation_mean'] / max(current_profile['saturation_mean'], 1)
    saturation_ratio = np.clip(saturation_ratio, 0.8, 1.2)  # Limit adjustment range
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_ratio, 0, 255)
    
    # Adjust value (brightness) to match original
    value_ratio = original_profile['value_mean'] / max(current_profile['value_mean'], 1)
    value_ratio = np.clip(value_ratio, 0.9, 1.1)  # Limit adjustment range
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_ratio, 0, 255)
    
    # Convert back to BGR
    corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Blend with original to avoid over-correction
    alpha = 0.7  # 70% correction, 30% original
    result = cv2.addWeighted(corrected, alpha, img, 1 - alpha, 0)
    
    return result

def apply_detail_preservation(img: np.ndarray) -> np.ndarray:
    """Apply detail enhancement to counteract video compression loss"""
    # Create unsharp mask
    gaussian = cv2.GaussianBlur(img, (0, 0), 1.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    
    # Apply edge enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Blend edge enhancement
    edge_enhanced = cv2.addWeighted(unsharp, 0.95, edges_colored, 0.05, 0)
    
    return edge_enhanced

def apply_smart_denoising(img: np.ndarray) -> np.ndarray:
    """Apply noise reduction while preserving important details"""
    # Use bilateral filter for edge-preserving denoising
    denoised = cv2.bilateralFilter(img, 5, 75, 75)
    
    # Create edge mask to protect details
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0) / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=-1)
    
    # Blend: use original in edge areas, denoised in smooth areas
    result = img.astype(np.float64) * edge_mask + denoised.astype(np.float64) * (1 - edge_mask)
    
    return result.astype(np.uint8)

def create_lossless_video_segment(video_path: str, start_frame: int, end_frame: int, output_path: str) -> str:
    """
    Create a lossless video segment to preserve maximum quality
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        # Use FFmpeg with lossless settings
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"select='between(n\\,{start_frame}\\,{end_frame})'",
            "-c:v", "libx264",
            "-preset", "veryslow",  # Best compression efficiency
            "-crf", "0",  # Lossless
            "-pix_fmt", "yuv444p",  # Full chroma resolution
            "-vsync", "vfr",
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Created lossless video segment: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg lossless encoding failed: {e.stderr}")
        # Fallback to high quality but not lossless
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"select='between(n\\,{start_frame}\\,{end_frame})'",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "12",  # Very high quality
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Created high-quality video segment: {output_path}")
        return output_path