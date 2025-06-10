# utils/face_swapping.py
"""
Face Swapping Module for Character Consistency
Uses InsightFace for high-quality face swapping across video chains
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
import tempfile
import shutil
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

def safe_cv2_cleanup():
    """
    Safely cleanup OpenCV resources without GUI dependencies
    Avoids Windows OpenCV GUI errors in headless environments
    """
    try:
        # Only attempt GUI cleanup if we're in an environment that supports it
        import sys
        if hasattr(cv2, 'destroyAllWindows') and 'pytest' not in sys.modules:
            # Try to detect if GUI is available before calling
            try:
                cv2.destroyAllWindows()
            except cv2.error as e:
                # Silently ignore GUI-related errors
                if "is not implemented" in str(e) or "GTK" in str(e):
                    pass
                else:
                    raise
    except Exception as e:
        # Log warning but don't crash
        logger.debug(f"OpenCV GUI cleanup skipped: {str(e)}")
        pass

class FaceSwapper:
    """Handles face swapping for character consistency in videos"""
    
    def __init__(self, device: str = 'cuda' if onnxruntime.get_device() == 'GPU' else 'cpu'):
        """
        Initialize face swapping models
        
        Args:
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        self.face_app = None
        self.face_swapper = None
        self.reference_faces = {}
        self.model_loaded = False
        
        # Model paths
        self.model_dir = 'insightface_models'
        self.swapper_model_path = os.path.join(self.model_dir, 'inswapper_128.onnx')
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize InsightFace models"""
        try:
            # Create model directory
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Initialize face analysis app
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                root=self.model_dir,
                providers=providers
            )
            self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
            
            # Check if swapper model exists, if not download it
            if not os.path.exists(self.swapper_model_path):
                logger.info("Downloading face swapper model...")
                self._download_swapper_model()
            
            # Load face swapper model
            self.face_swapper = insightface.model_zoo.get_model(
                self.swapper_model_path,
                providers=providers
            )
            
            self.model_loaded = True
            logger.info(f"Face swapping models initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing face swapping models: {str(e)}")
            self.model_loaded = False
    
    def _download_swapper_model(self):
        """Download the face swapper model"""
        try:
            import urllib.request
            # URL for the inswapper model
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
            
            logger.info(f"Downloading from {url}")
            urllib.request.urlretrieve(url, self.swapper_model_path)
            logger.info("Face swapper model downloaded successfully")
            
        except Exception as e:
            logger.error(f"Error downloading swapper model: {str(e)}")
            raise
    
    def extract_face_embedding(self, image_path: str) -> Optional[Dict]:
        """
        Extract face embedding from reference image
        
        Args:
            image_path: Path to reference image
            
        Returns:
            Face data dictionary or None
        """
        if not self.model_loaded:
            logger.error("Models not loaded")
            return None
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Detect faces
            faces = self.face_app.get(img)
            
            if not faces:
                logger.warning(f"No faces detected in {image_path}")
                return None
            
            # Get the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Store face data
            face_data = {
                'embedding': largest_face.embedding,
                'bbox': largest_face.bbox,
                'kps': largest_face.kps,
                'det_score': largest_face.det_score,
                'gender': largest_face.gender,
                'age': largest_face.age
            }
            
            logger.info(f"Extracted face embedding: age={face_data['age']}, gender={face_data['gender']}")
            return face_data
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None
    
    def swap_face_in_image(self, source_img: np.ndarray, target_face_data: Dict, 
                          reference_face_data: Dict) -> np.ndarray:
        """
        Swap a face in an image with reference face
        
        Args:
            source_img: Source image array
            target_face_data: Face data from target image
            reference_face_data: Face data from reference image
            
        Returns:
            Image with swapped face
        """
        try:
            # Prepare face data for swapping
            target_face = insightface.app.common.Face(d={
                'embedding': target_face_data['embedding'],
                'bbox': target_face_data['bbox'],
                'kps': target_face_data['kps'],
                'det_score': target_face_data['det_score']
            })
            
            source_face = insightface.app.common.Face(d={
                'embedding': reference_face_data['embedding'],
                'bbox': reference_face_data['bbox'],
                'kps': reference_face_data['kps'],
                'det_score': reference_face_data['det_score']
            })
            
            # Perform face swap
            result = self.face_swapper.get(source_img, target_face, source_face, paste_back=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error swapping face: {str(e)}")
            return source_img
    
    def swap_faces_in_video(self, video_path: str, reference_image_path: str, 
                           output_path: Optional[str] = None, 
                           skip_frames: int = 1,
                           similarity_threshold: float = 0.5) -> str:
        """
        Swap faces in video with reference face
        
        Args:
            video_path: Path to input video
            reference_image_path: Path to reference face image
            output_path: Path to save output video
            skip_frames: Process every Nth frame (1 = all frames)
            similarity_threshold: Minimum similarity to perform swap
            
        Returns:
            Path to output video
        """
        if not self.model_loaded:
            logger.error("Models not loaded, returning original video")
            return video_path
        
        # Extract reference face
        reference_face_data = self.extract_face_embedding(reference_image_path)
        if not reference_face_data:
            logger.error("Failed to extract reference face")
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
                output_path = video_path.replace('.mp4', '_faceswapped.mp4')
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            logger.info(f"Processing {total_frames} frames...")
            
            # Process frames
            frame_count = 0
            swapped_count = 0
            
            with tqdm(total=total_frames, desc="Swapping faces") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame if not skipping
                    if frame_count % skip_frames == 0:
                        # Detect faces in current frame
                        faces = self.face_app.get(frame)
                        
                        if faces:
                            # Process each face
                            frame_modified = False
                            result_frame = frame.copy()
                            
                            for face in faces:
                                # Calculate similarity with reference
                                similarity = self._calculate_face_similarity(
                                    face.embedding, 
                                    reference_face_data['embedding']
                                )
                                
                                # Swap if similarity is above threshold
                                if similarity >= similarity_threshold:
                                    face_data = {
                                        'embedding': face.embedding,
                                        'bbox': face.bbox,
                                        'kps': face.kps,
                                        'det_score': face.det_score
                                    }
                                    
                                    result_frame = self.swap_face_in_image(
                                        result_frame, face_data, reference_face_data
                                    )
                                    frame_modified = True
                                    swapped_count += 1
                            
                            if frame_modified:
                                frame = result_frame
                    
                    # Write frame
                    out.write(frame)
                    frame_count += 1
                    pbar.update(1)
            
            # Clean up
            cap.release()
            out.release()
            safe_cv2_cleanup()  # Safely handle OpenCV GUI cleanup
            
            logger.info(f"Face swapping complete. Swapped {swapped_count} faces in {frame_count} frames")
            logger.info(f"Output saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return video_path
    
    def _calculate_face_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between face embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def batch_swap_faces_in_video_chain(self, video_paths: List[str], 
                                      reference_image_path: str,
                                      output_dir: Optional[str] = None) -> List[str]:
        """
        Process multiple videos in a chain with face swapping
        
        Args:
            video_paths: List of video paths
            reference_image_path: Path to reference face image
            output_dir: Directory to save output videos
            
        Returns:
            List of output video paths
        """
        if output_dir is None:
            output_dir = os.path.dirname(video_paths[0])
        
        output_paths = []
        
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}")
            
            output_path = os.path.join(
                output_dir,
                f"faceswapped_chain_{i+1:02d}.mp4"
            )
            
            swapped_path = self.swap_faces_in_video(
                video_path, 
                reference_image_path, 
                output_path
            )
            
            output_paths.append(swapped_path)
        
        return output_paths
    
    def create_face_consistency_report(self, video_paths: List[str], 
                                     reference_image_path: str) -> Dict:
        """
        Analyze face consistency across video chain
        
        Args:
            video_paths: List of video paths
            reference_image_path: Reference image path
            
        Returns:
            Consistency report dictionary
        """
        reference_face_data = self.extract_face_embedding(reference_image_path)
        if not reference_face_data:
            return {"error": "Failed to extract reference face"}
        
        report = {
            "reference_age": reference_face_data['age'],
            "reference_gender": reference_face_data['gender'],
            "video_analysis": [],
            "average_similarity": 0.0,
            "consistency_score": 0.0
        }
        
        all_similarities = []
        
        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            
            # Sample frames from video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample 5 frames evenly
            sample_indices = np.linspace(0, total_frames-1, min(5, total_frames), dtype=int)
            
            video_similarities = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    faces = self.face_app.get(frame)
                    if faces:
                        # Get main face
                        main_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        
                        similarity = self._calculate_face_similarity(
                            main_face.embedding,
                            reference_face_data['embedding']
                        )
                        
                        video_similarities.append(similarity)
            
            cap.release()
            safe_cv2_cleanup()  # Safely handle OpenCV GUI cleanup
            
            if video_similarities:
                avg_similarity = np.mean(video_similarities)
                all_similarities.extend(video_similarities)
                
                report["video_analysis"].append({
                    "video": video_name,
                    "average_similarity": float(avg_similarity),
                    "min_similarity": float(min(video_similarities)),
                    "max_similarity": float(max(video_similarities))
                })
        
        if all_similarities:
            report["average_similarity"] = float(np.mean(all_similarities))
            report["consistency_score"] = float(np.std(all_similarities))  # Lower is better
        
        return report


# Enhanced face enhancement module with face swapping
class EnhancedVideoFaceConsistencyEnhancer:
    """Enhanced face consistency enhancer with face swapping capabilities"""
    
    def __init__(self, reference_image_path: str):
        """
        Initialize with reference image
        
        Args:
            reference_image_path: Path to reference image
        """
        self.reference_image_path = reference_image_path
        self.face_swapper = FaceSwapper()
        
        # Import the original face enhancer if available
        try:
            from utils.face_enhancement import FaceEnhancer
            self.face_enhancer = FaceEnhancer()
            self.enhancement_available = True
        except:
            self.enhancement_available = False
            logger.warning("Face enhancement not available, using swapping only")
    
    def enhance_video_with_reference(self, video_path: str, output_path: str,
                                   enhance_quality: bool = True,
                                   swap_faces: bool = True) -> str:
        """
        Enhance video using reference face for consistency
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            enhance_quality: Whether to enhance face quality
            swap_faces: Whether to perform face swapping
            
        Returns:
            Path to processed video
        """
        temp_path = video_path
        
        # Step 1: Face swapping for consistency
        if swap_faces:
            logger.info("Performing face swapping for consistency...")
            temp_path = self.face_swapper.swap_faces_in_video(
                temp_path,
                self.reference_image_path,
                output_path.replace('.mp4', '_swapped.mp4')
            )
        
        # Step 2: Face enhancement for quality
        if enhance_quality and self.enhancement_available:
            logger.info("Enhancing face quality...")
            final_path = self.face_enhancer.enhance_video(
                temp_path,
                output_path
            )
            
            # Clean up temporary file
            if temp_path != video_path and os.path.exists(temp_path):
                os.remove(temp_path)
            
            return final_path
        else:
            # Rename to final output if no enhancement
            if temp_path != output_path:
                shutil.move(temp_path, output_path)
            return output_path
    
    def process_video_chain(self, video_paths: List[str], output_dir: str,
                          enhance_quality: bool = True,
                          swap_faces: bool = True,
                          create_report: bool = True) -> Tuple[List[str], Optional[Dict]]:
        """
        Process entire video chain with face swapping and enhancement
        
        Args:
            video_paths: List of video paths
            output_dir: Output directory
            enhance_quality: Whether to enhance quality
            swap_faces: Whether to swap faces
            create_report: Whether to create consistency report
            
        Returns:
            Tuple of (processed_paths, consistency_report)
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_paths = []
        
        # Process each video
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}")
            
            output_path = os.path.join(
                output_dir,
                f"processed_chain_{i+1:02d}.mp4"
            )
            
            processed_path = self.enhance_video_with_reference(
                video_path,
                output_path,
                enhance_quality=enhance_quality,
                swap_faces=swap_faces
            )
            
            processed_paths.append(processed_path)
        
        # Create consistency report
        report = None
        if create_report:
            logger.info("Creating consistency report...")
            report = self.face_swapper.create_face_consistency_report(
                processed_paths,
                self.reference_image_path
            )
            
            # Save report
            report_path = os.path.join(output_dir, "face_consistency_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Consistency report saved to: {report_path}")
        
        return processed_paths, report