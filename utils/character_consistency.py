# utils/character_consistency.py
"""
Character Consistency Module
Handles character validation, tracking, and enhancement across video chains
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json
import face_recognition
from deepface import DeepFace
from PIL import Image
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

logger = logging.getLogger(__name__)

class CharacterConsistencyManager:
    """Manages character consistency across video chains"""
    
    def __init__(self, reference_image_path: str, output_dir: str = "./outputs"):
        """
        Initialize the character consistency manager
        
        Args:
            reference_image_path: Path to the reference image containing the character
            output_dir: Directory to save analysis outputs
        """
        self.reference_image_path = reference_image_path
        self.output_dir = output_dir
        self.character_profile = None
        self.face_landmarks = None
        self.reference_encoding = None
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Initialize character profile
        self._initialize_character_profile()
    
    def _initialize_character_profile(self):
        """Extract and store character features from reference image"""
        try:
            # Load reference image
            image = cv2.imread(self.reference_image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract face encoding using face_recognition
            face_locations = face_recognition.face_locations(rgb_image)
            if face_locations:
                self.reference_encoding = face_recognition.face_encodings(
                    rgb_image, face_locations
                )[0]
            
            # Extract detailed features using DeepFace
            try:
                analysis = DeepFace.analyze(
                    self.reference_image_path,
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False
                )
                
                # Handle different result formats
                if isinstance(analysis, list) and len(analysis) > 0:
                    analysis_data = analysis[0]
                elif isinstance(analysis, dict):
                    analysis_data = analysis
                else:
                    raise ValueError("Unexpected analysis result format")
                
                # Store character profile
                self.character_profile = {
                    'age': analysis_data.get('age', 25),
                    'gender': analysis_data.get('dominant_gender', analysis_data.get('gender', 'unknown')),
                    'dominant_race': analysis_data.get('dominant_race', 'unknown'),
                    'dominant_emotion': analysis_data.get('dominant_emotion', 'neutral'),
                    'face_region': analysis_data.get('region', {})
                }
            except Exception as e:
                logger.warning(f"DeepFace analysis failed: {str(e)}")
                self.character_profile = self._fallback_character_analysis(image)
            
            # Extract face landmarks using MediaPipe
            results = self.face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                self.face_landmarks = self._extract_landmark_features(
                    results.multi_face_landmarks[0], image.shape
                )
            
            # Extract additional visual features
            self.character_profile.update(self._extract_visual_features(image))
            
            # Save character profile
            profile_path = os.path.join(self.output_dir, "character_profile.json")
            os.makedirs(self.output_dir, exist_ok=True)
            with open(profile_path, 'w') as f:
                json.dump(self.character_profile, f, indent=2)
            
            logger.info(f"Character profile initialized: {self.character_profile}")
            
        except Exception as e:
            logger.error(f"Error initializing character profile: {str(e)}")
            raise
    
    def _fallback_character_analysis(self, image: np.ndarray) -> Dict:
        """Fallback character analysis using basic CV techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return {
                'face_region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'detection_method': 'cascade_classifier'
            }
        return {'face_region': None, 'detection_method': 'none'}
    
    def _extract_landmark_features(self, landmarks, image_shape) -> Dict:
        """Extract key facial landmark positions"""
        h, w = image_shape[:2]
        landmark_points = {}
        
        # Key landmark indices for face recognition
        key_indices = {
            'left_eye': [33, 133, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 263, 387, 388, 389, 390, 391, 466],
            'nose_tip': [1, 2, 4, 5, 6],
            'mouth': [13, 14, 269, 270, 267, 271, 272],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464, 234]
        }
        
        for feature, indices in key_indices.items():
            points = []
            for idx in indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    points.append([landmark.x * w, landmark.y * h])
            landmark_points[feature] = points
        
        return landmark_points
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract additional visual features like hair color, clothing color"""
        features = {}
        
        # Extract dominant colors from different regions
        h, w = image.shape[:2]
        
        # Hair region (top of head)
        hair_region = image[0:int(h*0.3), int(w*0.2):int(w*0.8)]
        features['hair_color'] = self._get_dominant_color(hair_region)
        
        # Clothing region (bottom part)
        clothing_region = image[int(h*0.7):h, int(w*0.2):int(w*0.8)]
        features['clothing_color'] = self._get_dominant_color(clothing_region)
        
        # Skin tone (from face region if available)
        if self.character_profile and self.character_profile.get('face_region'):
            face = self.character_profile['face_region']
            if face:
                face_img = image[face['y']:face['y']+face['h'], 
                                face['x']:face['x']+face['w']]
                features['skin_tone'] = self._get_dominant_color(face_img)
        
        return features
    
    def _get_dominant_color(self, image_region: np.ndarray) -> List[int]:
        """Get dominant color from image region"""
        if image_region.size == 0:
            return [0, 0, 0]
        
        # Reshape and cluster colors
        pixels = image_region.reshape(-1, 3)
        # Use simple mean for efficiency
        dominant = np.mean(pixels, axis=0).astype(int)
        return dominant.tolist()
    
    def validate_character_consistency(self, frame_path: str, threshold: float = 0.6) -> Tuple[float, Dict]:
        """
        Validate if the character in the frame matches the reference
        
        Args:
            frame_path: Path to the frame to validate
            threshold: Similarity threshold (0-1)
            
        Returns:
            Tuple of (similarity_score, validation_details)
        """
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            validation_results = {
                'face_match': 0.0,
                'landmark_similarity': 0.0,
                'color_consistency': 0.0,
                'overall_score': 0.0,
                'is_consistent': False,
                'details': {}
            }
            
            # Face recognition matching
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations and self.reference_encoding is not None:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    # Calculate face similarity
                    face_distances = face_recognition.face_distance(
                        face_encodings, self.reference_encoding
                    )
                    face_match = 1 - min(face_distances)  # Convert distance to similarity
                    validation_results['face_match'] = float(face_match)
                    validation_results['details']['face_distance'] = float(min(face_distances))
            
            # Landmark similarity
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks and self.face_landmarks:
                current_landmarks = self._extract_landmark_features(
                    results.multi_face_landmarks[0], frame.shape
                )
                landmark_sim = self._calculate_landmark_similarity(
                    self.face_landmarks, current_landmarks
                )
                validation_results['landmark_similarity'] = landmark_sim
            
            # Color consistency
            current_features = self._extract_visual_features(frame)
            color_sim = self._calculate_color_similarity(
                self.character_profile, current_features
            )
            validation_results['color_consistency'] = color_sim
            
            # Calculate overall score
            weights = {
                'face_match': 0.5,
                'landmark_similarity': 0.3,
                'color_consistency': 0.2
            }
            
            overall_score = sum(
                validation_results[key] * weight 
                for key, weight in weights.items()
            )
            
            validation_results['overall_score'] = overall_score
            validation_results['is_consistent'] = overall_score >= threshold
            
            # Log validation results
            logger.info(f"Character validation for {frame_path}: {validation_results}")
            
            return overall_score, validation_results
            
        except Exception as e:
            logger.error(f"Error validating character consistency: {str(e)}")
            return 0.0, {'error': str(e)}
    
    def _calculate_landmark_similarity(self, ref_landmarks: Dict, curr_landmarks: Dict) -> float:
        """Calculate similarity between facial landmarks"""
        similarities = []
        
        for feature in ref_landmarks:
            if feature in curr_landmarks:
                ref_points = np.array(ref_landmarks[feature])
                curr_points = np.array(curr_landmarks[feature])
                
                if len(ref_points) == len(curr_points) and len(ref_points) > 0:
                    # Normalize by face size
                    ref_norm = ref_points - np.mean(ref_points, axis=0)
                    curr_norm = curr_points - np.mean(curr_points, axis=0)
                    
                    # Calculate similarity
                    diff = np.mean(np.abs(ref_norm - curr_norm))
                    similarity = max(0, 1 - diff / 100)  # Normalize to 0-1
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_color_similarity(self, ref_features: Dict, curr_features: Dict) -> float:
        """Calculate color similarity between features"""
        similarities = []
        
        color_keys = ['hair_color', 'clothing_color', 'skin_tone']
        for key in color_keys:
            if key in ref_features and key in curr_features:
                ref_color = np.array(ref_features[key])
                curr_color = np.array(curr_features[key])
                
                # Calculate color distance
                distance = np.linalg.norm(ref_color - curr_color)
                similarity = max(0, 1 - distance / 441.67)  # Max RGB distance
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def generate_character_description(self) -> Dict[str, str]:
        """Generate detailed character description for prompts"""
        description = {
            'appearance': "",
            'facial_features': "",
            'clothing': "",
            'distinctive_features': ""
        }
        
        # Log what we actually detected for debugging
        logger.info(f"🔍 CHARACTER ANALYSIS DEBUG:")
        logger.info(f"Character profile: {self.character_profile}")
        
        # Try to use OpenAI analysis as fallback for more accurate description
        try:
            openai_description = self._get_openai_character_description()
            if openai_description:
                logger.info(f"✅ Using OpenAI character analysis: {openai_description}")
                return openai_description
        except Exception as e:
            logger.warning(f"OpenAI character analysis failed: {str(e)}")
        
        # Fallback to original method with improved logic
        if self.character_profile:
            # Build appearance description
            appearance_parts = []
            if 'age' in self.character_profile:
                appearance_parts.append(f"{self.character_profile['age']} years old")
            if 'gender' in self.character_profile:
                appearance_parts.append(self.character_profile['gender'])
            if 'dominant_race' in self.character_profile:
                appearance_parts.append(self.character_profile['dominant_race'])
            
            description['appearance'] = ", ".join(appearance_parts)
            
            # Hair color
            if 'hair_color' in self.character_profile:
                rgb = self.character_profile['hair_color']
                hair_desc = self._rgb_to_description(rgb, "hair")
                description['distinctive_features'] += f"{hair_desc} hair, "
            
            # Clothing
            if 'clothing_color' in self.character_profile:
                rgb = self.character_profile['clothing_color']
                clothing_desc = self._rgb_to_description(rgb, "clothing")
                description['clothing'] = f"wearing {clothing_desc} colored clothing"
        
        logger.warning(f"⚠️ Using fallback character analysis: {description}")
        return description
    
    def _get_openai_character_description(self) -> Optional[Dict[str, str]]:
        """Use OpenAI to get accurate character description"""
        try:
            import base64
            from openai import OpenAI
            from config import OPENAI_API_KEY
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Encode the image
            with open(self.reference_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Get detailed description from OpenAI with custom prompt
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": """Analyze this image and provide a detailed character description. 
                                Focus on: age (approximate), gender, hair color/style, clothing description, and distinctive features.
                                Be very specific about clothing (e.g., 'silver metallic bikini' not just 'clothing').
                                Be specific about hair color and style.
                                Describe exactly what you see in the image."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            full_description = response.choices[0].message.content
            logger.info(f"🎯 OpenAI character analysis: {full_description}")
            
            # Parse into components
            description = {
                'appearance': "",
                'facial_features': "",
                'clothing': "",
                'distinctive_features': ""
            }
            
            # Extract basic info from full description
            lower_desc = full_description.lower()
            
            # Gender and age
            if 'woman' in lower_desc or 'female' in lower_desc:
                if 'young' in lower_desc:
                    description['appearance'] = "young woman"
                else:
                    description['appearance'] = "woman"
            elif 'man' in lower_desc or 'male' in lower_desc:
                if 'young' in lower_desc:
                    description['appearance'] = "young man"
                else:
                    description['appearance'] = "man"
            
            # Clothing - look for specific terms
            if 'bikini' in lower_desc:
                if 'silver' in lower_desc or 'metallic' in lower_desc:
                    description['clothing'] = "wearing a silver metallic bikini"
                elif 'reflective' in lower_desc:
                    description['clothing'] = "wearing a reflective bikini"
                else:
                    description['clothing'] = "wearing a bikini"
            elif 'swimsuit' in lower_desc:
                description['clothing'] = "wearing a swimsuit"
            elif 'dress' in lower_desc:
                description['clothing'] = "wearing a dress"
            elif 'shirt' in lower_desc:
                description['clothing'] = "wearing a shirt"
            
            # Hair - more comprehensive search
            hair_colors = ['blonde', 'brown', 'black', 'red', 'silver', 'gray', 'white', 'dark', 'light']
            hair_styles = ['long', 'short', 'curly', 'straight', 'wavy']
            
            hair_parts = []
            for color in hair_colors:
                if color in lower_desc and 'hair' in lower_desc:
                    hair_parts.append(color)
                    break
            
            for style in hair_styles:
                if style in lower_desc and 'hair' in lower_desc:
                    hair_parts.append(style)
                    break
            
            if hair_parts:
                description['distinctive_features'] = f"{' '.join(hair_parts)} hair"
            
            # If we found some details, use this analysis
            if description['clothing'] or description['distinctive_features'] or description['appearance']:
                logger.info(f"✅ Parsed OpenAI description: {description}")
                return description
                
        except Exception as e:
            logger.error(f"Error getting OpenAI character description: {str(e)}")
        
        return None
    
    def _rgb_to_description(self, rgb: List[int], context: str) -> str:
        """Convert RGB values to human-readable color description"""
        r, g, b = rgb
        
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return "white" if context == "clothing" else "light/blonde"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "reddish" if context == "hair" else "red"
        elif g > r and g > b:
            return "greenish"
        elif b > r and b > g:
            return "bluish"
        elif r > 150 and g > 100 and b < 100:
            return "brown" if context == "hair" else "brown/tan"
        else:
            return "medium-toned"
    
    def create_validation_report(self, video_paths: List[str]) -> Dict:
        """Create a comprehensive validation report for all videos in the chain"""
        report = {
            'total_videos': len(video_paths),
            'validation_scores': [],
            'average_consistency': 0.0,
            'failed_frames': [],
            'recommendations': []
        }
        
        all_scores = []
        
        for video_path in video_paths:
            # Extract last frame from each video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame temporarily
                    temp_frame = os.path.join(self.output_dir, "temp_validation_frame.jpg")
                    cv2.imwrite(temp_frame, frame)
                    
                    # Validate
                    score, details = self.validate_character_consistency(temp_frame)
                    
                    video_result = {
                        'video': os.path.basename(video_path),
                        'score': score,
                        'details': details
                    }
                    
                    report['validation_scores'].append(video_result)
                    all_scores.append(score)
                    
                    if score < 0.6:
                        report['failed_frames'].append(video_path)
                    
                    # Clean up
                    os.remove(temp_frame)
            
            cap.release()
        
        # Calculate statistics
        if all_scores:
            report['average_consistency'] = np.mean(all_scores)
            report['min_score'] = min(all_scores)
            report['max_score'] = max(all_scores)
            
            # Generate recommendations
            if report['average_consistency'] < 0.7:
                report['recommendations'].append(
                    "Consider regenerating videos with stronger character descriptions"
                )
            if report['failed_frames']:
                report['recommendations'].append(
                    f"Regenerate {len(report['failed_frames'])} videos with low consistency"
                )
        
        # Save report with JSON serialization fix
        report_path = os.path.join(self.output_dir, "character_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        return report
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)