# utils/enhanced_fal_client.py
"""
Enhanced FAL client with quality optimization for preventing degradation
"""

import fal_client
import requests
import os
import logging
import time
import base64
import json
import config
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import hashlib

logger = logging.getLogger(__name__)

# Set the API key
os.environ["FAL_KEY"] = config.FAL_API_KEY

def upload_file_with_maximum_quality(file_path: str, preserve_original_quality: bool = True) -> str:
    """
    Upload a file to FAL.ai with maximum quality preservation settings.
    
    Args:
        file_path: Path to the file to upload
        preserve_original_quality: Whether to apply quality preservation preprocessing
        
    Returns:
        URL of the uploaded file
    """
    logger.info(f"🎯 MAXIMUM QUALITY UPLOAD: {file_path}")
    
    try:
        # Create quality-optimized version
        optimized_path = None
        
        if preserve_original_quality:
            optimized_path = _create_quality_optimized_image(file_path)
        else:
            optimized_path = file_path
        
        # Upload with PNG format for lossless quality
        with open(optimized_path, 'rb') as f:
            file_content = f.read()
        
        # Force PNG content type for maximum quality
        result = fal_client.upload(file_content, content_type='image/png')
        
        # Log quality metrics
        original_size = os.path.getsize(file_path) / 1024  # KB
        optimized_size = os.path.getsize(optimized_path) / 1024  # KB
        logger.info(f"QUALITY UPLOAD: {original_size:.1f}KB → {optimized_size:.1f}KB (lossless PNG)")
        
        # Clean up temporary file if created
        if optimized_path != file_path and os.path.exists(optimized_path):
            try:
                os.remove(optimized_path)
            except:
                pass
        
        logger.info(f"Maximum quality upload successful: {result}")
        return result
        
    except Exception as e:
        logger.warning(f"Maximum quality upload failed: {str(e)}, falling back to standard")
        return _upload_file_standard_fallback(file_path)

def _create_quality_optimized_image(file_path: str) -> str:
    """
    Create a quality-optimized version of the image for upload.
    This prevents quality loss during FAL.ai processing.
    """
    try:
        # Load image
        img = Image.open(file_path).convert("RGB")
        img_array = np.array(img)
        
        # Apply quality preservation preprocessing
        enhanced_array = _apply_quality_preservation_preprocessing(img_array)
        
        # Convert back to PIL
        enhanced_img = Image.fromarray(enhanced_array)
        
        # Create temporary file with quality optimization
        temp_dir = os.path.dirname(file_path)
        temp_filename = f"quality_optimized_{int(time.time())}_{os.path.basename(file_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Ensure we use PNG extension for lossless
        if not temp_path.lower().endswith('.png'):
            temp_path = os.path.splitext(temp_path)[0] + '.png'
        
        # Save with maximum quality PNG settings
        enhanced_img.save(temp_path, format='PNG', optimize=False, compress_level=0)
        
        logger.info(f"Quality optimization applied: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Quality optimization failed: {str(e)}")
        return file_path

def _apply_quality_preservation_preprocessing(img_array: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing to preserve quality through FAL.ai processing.
    """
    import cv2
    
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. Pre-sharpen to compensate for compression
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * 0.5
    pre_sharpened = cv2.filter2D(img_cv, -1, kernel)
    img_cv = cv2.addWeighted(img_cv, 0.7, pre_sharpened, 0.3, 0)
    
    # 2. Enhance color saturation slightly to prevent washing out
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).astype(np.float64)
    
    # Boost saturation by 5-10% to compensate for compression
    current_saturation = np.mean(hsv[:, :, 1])
    if current_saturation < 200:  # Avoid over-saturating already vibrant images
        saturation_boost = min(1.1, 1.0 + (200 - current_saturation) / 1000)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    
    # Convert back to BGR
    img_cv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 3. Light contrast enhancement to preserve details
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 4. Ensure no clipping
    img_cv = np.clip(img_cv, 0, 255)
    
    # Convert back to RGB
    result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    return result

def _upload_file_standard_fallback(file_path: str) -> str:
    """Standard file upload fallback"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Try PNG first, then fall back to original format
        _, ext = os.path.splitext(file_path)
        content_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        
        content_type = content_type_map.get(ext.lower(), 'image/png')
        result = fal_client.upload(file_content, content_type=content_type)
        
        logger.info(f"Standard upload successful: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"Standard upload failed: {str(e)}")
        raise

def generate_video_with_quality_optimization(prompt: str, image_url: str, resolution: str, 
                                            model: str = "wan", **kwargs) -> str:
    """
    Generate video with quality-optimized prompts and parameters.
    
    Args:
        prompt: Text prompt for the video
        image_url: URL of the image to use as starting point
        resolution: Video resolution
        model: Model to use
        **kwargs: Additional model-specific parameters
        
    Returns:
        URL of the generated video
    """
    # Optimize prompt for quality preservation
    optimized_prompt = _optimize_prompt_for_quality(prompt, model)
    
    # Optimize parameters for quality
    optimized_kwargs = _optimize_parameters_for_quality(kwargs, model, resolution)
    
    logger.info(f"🎬 QUALITY-OPTIMIZED GENERATION with {model.upper()}")
    logger.info(f"Original prompt: {prompt}")
    logger.info(f"Optimized prompt: {optimized_prompt}")
    
    # Call the original generation function with optimized parameters
    from .fal_client import generate_video_from_image
    
    return generate_video_from_image(
        prompt=optimized_prompt,
        image_url=image_url,
        resolution=resolution,
        model=model,
        **optimized_kwargs
    )

def _optimize_prompt_for_quality(prompt: str, model: str) -> str:
    """
    Optimize prompt to encourage quality preservation and detail retention.
    """
    # Quality enhancement keywords based on model
    quality_enhancers = {
        "wan": [
            "high quality", "detailed", "sharp focus", "crisp", 
            "vivid colors", "rich details", "professional"
        ],
        "pixverse": [
            "high resolution", "detailed textures", "sharp", "vibrant",
            "professional quality", "cinematic"
        ],
        "luma": [
            "high quality", "detailed", "sharp", "vivid", 
            "professional cinematography"
        ],
        "kling": [
            "high quality", "detailed", "sharp focus", "vivid colors",
            "professional"
        ]
    }
    
    # Quality preservation instructions
    preservation_terms = [
        "maintaining consistent visual quality",
        "preserving all details from the input image",
        "keeping rich colors and sharp textures"
    ]
    
    # Check if prompt already has quality terms
    prompt_lower = prompt.lower()
    has_quality_terms = any(term in prompt_lower for term in 
                           ["quality", "detailed", "sharp", "vivid", "crisp", "professional"])
    
    # Add quality enhancers if not present
    if not has_quality_terms:
        enhancers = quality_enhancers.get(model, quality_enhancers["wan"])
        selected_enhancers = enhancers[:2]  # Use first 2 to avoid over-stuffing
        
        # Add quality terms at the end to not interfere with creative content
        enhanced_prompt = f"{prompt}, {', '.join(selected_enhancers)}"
    else:
        enhanced_prompt = prompt
    
    # Add preservation instruction for chain continuity
    if "maintaining" not in prompt_lower and "preserving" not in prompt_lower:
        preservation_term = preservation_terms[0]  # Use first one
        enhanced_prompt = f"{enhanced_prompt}, {preservation_term}"
    
    return enhanced_prompt

def _optimize_parameters_for_quality(kwargs: dict, model: str, resolution: str) -> dict:
    """
    Optimize generation parameters for maximum quality output.
    """
    optimized = kwargs.copy()
    
    if model == "wan":
        # WAN model optimizations
        optimized["inference_steps"] = max(kwargs.get("inference_steps", 40), 35)  # Minimum 35 steps
        optimized["safety_checker"] = kwargs.get("safety_checker", False)  # Disable for better quality
        optimized["prompt_expansion"] = kwargs.get("prompt_expansion", True)  # Enable for better prompts
        
    elif model == "pixverse":
        # Pixverse optimizations
        if "negative_prompt" not in optimized or not optimized["negative_prompt"]:
            optimized["negative_prompt"] = "low quality, blurry, pixelated, compression artifacts, low resolution, washed out colors, flat lighting, poor details"
        else:
            # Enhance existing negative prompt
            quality_negatives = "low quality, blurry, washed out colors"
            if quality_negatives not in optimized["negative_prompt"]:
                optimized["negative_prompt"] = f"{optimized['negative_prompt']}, {quality_negatives}"
        
        # Optimize duration for quality vs length trade-off
        if resolution == "1080p":
            optimized["duration"] = 5  # Force 5s for 1080p
        else:
            optimized["duration"] = kwargs.get("duration", 8)  # Prefer 8s for lower resolutions
            
    elif model == "luma":
        # LUMA optimizations
        optimized["duration"] = "5s"  # Always 5 seconds for LUMA
        # LUMA doesn't have negative prompts, so focus on aspect ratio optimization
        if resolution in ["720p", "1080p"]:
            optimized["aspect_ratio"] = kwargs.get("aspect_ratio", "16:9")
        
    elif model == "kling":
        # Kling optimizations
        if "negative_prompt" not in optimized or not optimized["negative_prompt"]:
            # Use shorter, more effective negative prompt for Kling
            optimized["negative_prompt"] = "blur, low quality, artifacts"
        
        # Optimize creativity for quality
        current_creativity = kwargs.get("creativity", 0.5)
        # Lower creativity often produces higher quality, more consistent results
        optimized["creativity"] = min(current_creativity, 0.6)
        
        # Ensure proper duration
        optimized["duration"] = str(kwargs.get("duration", 5))
    
    logger.info(f"Parameter optimization for {model}: {optimized}")
    return optimized

def create_quality_comparison_report(original_image_path: str, extracted_frame_path: str) -> dict:
    """
    Create a quality comparison report between original and extracted frame.
    """
    try:
        # Load images
        original = Image.open(original_image_path).convert("RGB")
        extracted = Image.open(extracted_frame_path).convert("RGB")
        
        # Convert to numpy arrays
        orig_array = np.array(original)
        extract_array = np.array(extracted)
        
        # Resize extracted to match original if needed
        if orig_array.shape != extract_array.shape:
            extracted_resized = extracted.resize(original.size, Image.Resampling.LANCZOS)
            extract_array = np.array(extracted_resized)
        
        # Calculate quality metrics
        report = {
            "timestamp": time.time(),
            "original_path": original_image_path,
            "extracted_path": extracted_frame_path,
            "metrics": {}
        }
        
        # 1. Color similarity
        orig_hsv = cv2.cvtColor(orig_array, cv2.COLOR_RGB2HSV)
        extract_hsv = cv2.cvtColor(extract_array, cv2.COLOR_RGB2HSV)
        
        color_similarity = {
            "hue_diff": float(np.mean(np.abs(orig_hsv[:,:,0] - extract_hsv[:,:,0]))),
            "saturation_diff": float(np.mean(np.abs(orig_hsv[:,:,1] - extract_hsv[:,:,1]))),
            "value_diff": float(np.mean(np.abs(orig_hsv[:,:,2] - extract_hsv[:,:,2])))
        }
        
        # 2. Detail preservation (edge comparison)
        orig_gray = cv2.cvtColor(orig_array, cv2.COLOR_RGB2GRAY)
        extract_gray = cv2.cvtColor(extract_array, cv2.COLOR_RGB2GRAY)
        
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        extract_edges = cv2.Canny(extract_gray, 50, 150)
        
        edge_preservation = float(np.sum(orig_edges & extract_edges) / max(np.sum(orig_edges), 1))
        
        # 3. Overall similarity (SSIM would be ideal but keep it simple)
        mse = float(np.mean((orig_array - extract_array) ** 2))
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse)) if mse > 0 else float('inf')
        
        report["metrics"] = {
            "color_similarity": color_similarity,
            "edge_preservation": edge_preservation,
            "mse": mse,
            "psnr": psnr,
            "quality_score": _calculate_overall_quality_score(color_similarity, edge_preservation, psnr)
        }
        
        # 4. File size comparison
        orig_size = os.path.getsize(original_image_path)
        extract_size = os.path.getsize(extracted_frame_path)
        
        report["file_info"] = {
            "original_size_kb": orig_size / 1024,
            "extracted_size_kb": extract_size / 1024,
            "size_ratio": extract_size / orig_size if orig_size > 0 else 0
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error creating quality report: {str(e)}")
        return {"error": str(e)}

def _calculate_overall_quality_score(color_sim: dict, edge_preservation: float, psnr: float) -> float:
    """Calculate overall quality score from 0-1"""
    # Color score (lower differences = higher score)
    color_score = 1.0 - (
        (color_sim["hue_diff"] / 180.0) * 0.3 +
        (color_sim["saturation_diff"] / 255.0) * 0.4 +
        (color_sim["value_diff"] / 255.0) * 0.3
    )
    color_score = max(0, color_score)
    
    # Edge preservation score (already 0-1)
    edge_score = edge_preservation
    
    # PSNR score (convert to 0-1, typical range 20-50)
    psnr_score = min(1.0, max(0, (psnr - 20) / 30)) if psnr != float('inf') else 1.0
    
    # Weighted overall score
    overall_score = (
        color_score * 0.4 +
        edge_score * 0.3 +
        psnr_score * 0.3
    )
    
    return float(overall_score)

def save_quality_report(report: dict, output_dir: str) -> str:
    """Save quality report to JSON file"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(report.get("timestamp", time.time()))
        report_path = os.path.join(output_dir, f"quality_report_{timestamp}.json")
        
        # Convert numpy types to JSON-serializable types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean the report
        clean_report = json.loads(json.dumps(report, default=convert_types))
        
        with open(report_path, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        logger.info(f"Quality report saved: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error saving quality report: {str(e)}")
        return ""

# Enhanced upload function that wraps the quality optimization
def upload_file(file_path: str, high_quality: bool = True) -> str:
    """
    Enhanced upload function with quality optimization.
    
    Args:
        file_path: Path to file to upload
        high_quality: Whether to apply quality optimization
        
    Returns:
        URL of uploaded file
    """
    if high_quality:
        return upload_file_with_maximum_quality(file_path, preserve_original_quality=True)
    else:
        return _upload_file_standard_fallback(file_path)

# Integration function for the main pipeline
def integrate_quality_preservation_into_chain(video_paths: list, original_image_path: str, 
                                            output_dir: str) -> tuple:
    """
    Integrate quality preservation into the video chain generation process.
    
    Args:
        video_paths: List of generated video paths
        original_image_path: Path to the original input image
        output_dir: Output directory for reports and enhanced videos
        
    Returns:
        tuple: (quality_reports, enhancement_summary)
    """
    quality_reports = []
    enhancement_summary = {
        "total_videos": len(video_paths),
        "average_quality_score": 0.0,
        "quality_degradation": 0.0,
        "recommendations": []
    }
    
    try:
        # Create quality reports for each video in the chain
        for i, video_path in enumerate(video_paths):
            logger.info(f"Creating quality report for video {i+1}/{len(video_paths)}")
            
            # Extract frame from video for comparison
            from .video_processing import extract_simple_last_frame
            temp_frame_dir = os.path.join(output_dir, f"temp_analysis_{i}")
            os.makedirs(temp_frame_dir, exist_ok=True)
            
            try:
                frame_path, _ = extract_simple_last_frame(video_path, temp_frame_dir)
                
                # Create quality report
                report = create_quality_comparison_report(original_image_path, frame_path)
                
                if "error" not in report:
                    # Save report
                    report_path = save_quality_report(report, output_dir)
                    quality_reports.append(report)
                    
                    # Update summary
                    quality_score = report["metrics"]["quality_score"]
                    enhancement_summary["average_quality_score"] += quality_score
                    
                    # Check for degradation
                    if i == 0:
                        initial_quality = quality_score
                    else:
                        current_degradation = initial_quality - quality_score
                        enhancement_summary["quality_degradation"] = max(
                            enhancement_summary["quality_degradation"], 
                            current_degradation
                        )
                
                # Clean up temp files
                try:
                    import shutil
                    shutil.rmtree(temp_frame_dir)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Error processing video {i+1}: {str(e)}")
                continue
        
        # Calculate averages and recommendations
        if quality_reports:
            enhancement_summary["average_quality_score"] /= len(quality_reports)
            
            # Generate recommendations
            avg_quality = enhancement_summary["average_quality_score"]
            degradation = enhancement_summary["quality_degradation"]
            
            if avg_quality < 0.7:
                enhancement_summary["recommendations"].append(
                    "Consider using higher inference steps or different model for better quality"
                )
            
            if degradation > 0.2:
                enhancement_summary["recommendations"].append(
                    "Significant quality degradation detected. Consider shorter chains or face swapping"
                )
            
            if avg_quality < 0.8 and degradation > 0.15:
                enhancement_summary["recommendations"].append(
                    "Enable enhanced frame extraction and quality preservation features"
                )
        
        logger.info(f"Quality analysis complete. Average score: {enhancement_summary['average_quality_score']:.2f}")
        
        return quality_reports, enhancement_summary
        
    except Exception as e:
        logger.error(f"Error in quality preservation integration: {str(e)}")
        return [], {"error": str(e)}