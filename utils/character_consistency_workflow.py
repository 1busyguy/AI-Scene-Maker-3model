import os
import logging
from typing import List
from utils.character_consistency import CharacterConsistencyManager
from utils.face_enhancement import FaceEnhancer
from utils import video_processing
from utils import fal_client
from config import OUTPUT_DIR, DEFAULT_RESOLUTION

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_chain_videos(prompt: str, image_path: str, num_chains: int = 3,
                           model: str = "wan", resolution: str = DEFAULT_RESOLUTION) -> List[str]:
    """Generate a chain of videos while keeping character consistency."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    character_manager = CharacterConsistencyManager(image_path, OUTPUT_DIR)
    face_enhancer = FaceEnhancer()

    # Upload initial image
    logger.info("Uploading initial image...")
    current_image_url = fal_client.upload_file(image_path, high_quality=True)

    video_paths = []
    for chain in range(num_chains):
        chain_num = chain + 1
        logger.info(f"Generating video chain {chain_num} using model {model}...")
        video_url = fal_client.generate_video_from_image(
            prompt=prompt,
            image_url=current_image_url,
            resolution=resolution,
            model=model
        )
        video_path = os.path.join(OUTPUT_DIR, f"chain_{chain_num:02d}.mp4")
        fal_client.download_video(video_url, video_path)
        video_paths.append(video_path)

        if chain < num_chains - 1:
            # Extract final frame and trim video
            frame_path, trimmed_path = video_processing.extract_simple_last_frame(
                video_path,
                OUTPUT_DIR,
                chain_number=chain_num,
                avoid_over_enhancement=True
            )
            # Replace with trimmed video for smooth flow
            if os.path.exists(trimmed_path):
                video_paths[-1] = trimmed_path

            # Enhance and validate
            enhanced_frame = face_enhancer.enhance_face(frame_path, light_mode=True)
            score, _ = character_manager.validate_character_consistency(enhanced_frame)
            if score < 0.6:
                logger.warning(f"Chain {chain_num}: low consistency score {score:.2f}")
            current_image_url = fal_client.upload_file(enhanced_frame, high_quality=False)

    return video_paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate chained videos with character consistency")
    parser.add_argument("image", help="Path to the initial reference image")
    parser.add_argument("prompt", help="Base prompt for the videos")
    parser.add_argument("--chains", type=int, default=3, help="Number of video chains")
    parser.add_argument("--model", default="wan", help="Model type (wan, pixverse, luma, kling)")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="Video resolution")
    args = parser.parse_args()

    videos = generate_chain_videos(args.prompt, args.image, args.chains, args.model, args.resolution)
    print("Generated videos:")
    for path in videos:
        print(path)
