from openai import OpenAI
import base64
import os
import logging
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path):
    """Encode an image to base64 for sending to the OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_text(image_path):
    """Get a description of an image using GPT-4.1-mini Vision API"""
    logger.info(f"Getting description for image: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Encode the image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Call the OpenAI API with the GPT-4.1-mini model
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # Updated to use gpt-4o-mini
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
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
        
        # Extract the description from the response
        description = response.choices[0].message.content
        
        # *** CRITICAL FIX: Handle OpenAI content policy rejections ***
        if description is None:
            logger.warning("🚨 OpenAI content policy rejection: Image description blocked")
            return "[Content policy violation - using fallback description]"
        
        logger.info(f"Got image description: {description[:50]}...")
        return description
    except Exception as e:
        logger.exception(f"Error getting image description: {str(e)}")
        raise

def generate_scene_vision(action_direction, image_description):
    prompt = f"Combine the following into a cohesive scene vision:\nAction Direction: {action_direction}\nImage Description: {image_description}"
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",  # Updated to use gpt-4.1-mini
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.exception(f"Error generating scene vision: {str(e)}")
        raise

def determine_optimal_chain_count(scene_vision, action_direction):
    """
    Determine the optimal number of video chains based on the complexity of the scene vision
    and action direction.
    
    Args:
        scene_vision: The combined scene vision
        action_direction: The user's action direction
        
    Returns:
        int: Recommended number of chains (1-10)
    """
    prompt = """
    Analyze the following scene vision and action direction, and determine how many video chains 
    (1-10) would be optimal to create a fluid, cohesive video sequence. Consider:
    
    1. Complexity of the described transformation
    2. Number of distinct stages in the action
    3. Granularity needed to show the progression smoothly
    
    Return ONLY a single number between 1 and 10.
    
    Scene Vision: {scene_vision}
    
    Action Direction: {action_direction}
    """.format(scene_vision=scene_vision, action_direction=action_direction)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        
        # Extract the number from the response
        response = completion.choices[0].message.content.strip()
        # Try to parse as int, with fallback to default
        try:
            num_chains = int(''.join(filter(str.isdigit, response)))
            return max(1, min(10, num_chains))  # Ensure within bounds
        except:
            return 3  # Default fallback
    except Exception as e:
        logger.exception(f"Error determining optimal chain count: {str(e)}")
        raise

def create_chain_prompt(action_direction, scene_vision, last_frame_description, current_chain, total_chains):
    """
    Create a concise cinematic logline prompt for the next chain in the sequence.
    
    Args:
        action_direction: Overall movement/action directive
        scene_vision: Overall scene vision
        last_frame_description: Description of the last frame from previous chain
        current_chain: Current chain number (0-indexed)
        total_chains: Total number of chains
        
    Returns:
        str: A concise cinematic logline prompt
    """
    # Calculate progress percentage and narrative position
    progress_percent = (current_chain / total_chains) * 100
    
    # Define narrative positions based on chain progression
    if current_chain == 0:
        narrative_position = "ESTABLISHING"
    elif current_chain < total_chains / 3:
        narrative_position = "SETUP"
    elif current_chain < total_chains * 2/3:
        narrative_position = "DEVELOPMENT"
    else:
        narrative_position = "RESOLUTION"
    
    # Define cinematic shot types appropriate for the current narrative position
    cinematic_elements = {
        "ESTABLISHING": ["wide establishing shot", "aerial view", "slow tracking shot"],
        "SETUP": ["medium shot", "dolly in", "steadicam follow"],
        "DEVELOPMENT": ["close-up", "over-the-shoulder", "dutch angle", "handheld camera"],
        "RESOLUTION": ["dramatic reveal", "pull back shot", "slow motion", "intense close-up"]
    }
    
    # Select a cinematic element based on chain position
    shot_options = cinematic_elements[narrative_position]
    shot_type = shot_options[current_chain % len(shot_options)]
    
    # Construct a specific prompt that focuses on continuity
    prompt = f"""
    Create a single concise cinematic logline for a video generation AI that will continue EXACTLY from the last frame.
    
    NARRATIVE ARC: {narrative_position} ({progress_percent:.0f}% complete)
    
    ACTION DIRECTIVE: {action_direction}
    
    LAST FRAME: {last_frame_description}
    
    CURRENT SCENE VISION: Take the essence of "{scene_vision}" but focus on the CONTINUITY of visual elements.
    
    CINEMATIC TECHNIQUE: {shot_type}
    
    You MUST create a SINGLE CONCISE CINEMATIC LOGLINE that:
    1. Acts as a precise instruction for video generation
    2. Maintains PERFECT VISUAL CONTINUITY with the last frame
    3. Respects the overall action directive
    4. Incorporates the specified cinematic technique
    5. Emphasizes the exact same visual style, lighting, and character appearance
    6. Contains NO MORE THAN 50 WORDS
    7. Is formatted as a simple, clear, directive statement
    
    Format your response as a SINGLE PARAGRAPH with no preamble or explanation.
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        
        # Get the generated logline
        logline = completion.choices[0].message.content.strip()
        
        # Clean up any quotation marks that might have been added
        logline = logline.strip('"\'')
        
        # Log the generated logline
        logger.info(f"Generated logline: {logline}")
        
        return logline
    except Exception as e:
        logger.exception(f"Error creating chain prompt: {str(e)}")
        # Provide a simple fallback
        return f"Continue directly from the previous frame using a {shot_type}, maintaining the exact same visual appearance while {action_direction}"

def score_frame_consistency(frame_description, scene_vision, action_direction):
    """
    Score a frame based on how well it matches the scene vision and action direction.
    
    Args:
        frame_description: Description of the frame
        scene_vision: Overall scene vision
        action_direction: User's action direction
        
    Returns:
        float: A score from 0-10
    """
    prompt = f"""
    Score how well this frame description matches the overall scene vision and action direction.

    FRAME DESCRIPTION: {frame_description}
    
    SCENE VISION: {scene_vision}
    
    ACTION DIRECTION: {action_direction}
    
    Score from 0-10 based on:
    1. Visual quality (clarity, detail, composition)
    2. Consistency with scene vision
    3. Progression of the action direction
    4. Potential for continued development
    5. Absence of artifacts or visual issues
    
    Return ONLY a number from 0-10.
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        
        # Extract the score from the response
        response = completion.choices[0].message.content.strip()
        try:
            # Try to extract just the number
            score = float(''.join(filter(lambda x: x.isdigit() or x == '.', response)))
            return min(10, max(0, score))  # Ensure within bounds
        except:
            # Default middle score if parsing fails
            return 5.0
    except Exception as e:
        logger.exception(f"Error scoring frame consistency: {str(e)}")
        # Return a default middle score
        return 5.0

def extract_field(text, field_name):
    """Helper function to extract fields from text if JSON parsing fails"""
    import re
    
    # Try different patterns to match field content
    patterns = [
        # Pattern 1: Field name followed by colon and content until next field or end
        rf"{field_name}:?\s*(.*?)(?=(?:Theme|Background|Main Subject|Tone and Color|Action Direction):|$)",
        # Pattern 2: Number + field name followed by content
        rf"\d+\.\s+{field_name}:?\s*(.*?)(?=\d+\.|$)",
        # Pattern 3: Field name in quotes followed by content
        rf'"{field_name}":\s*"(.*?)"',
        # Pattern 4: Just look for the field name and take what follows
        rf"{field_name}[:\s]+(.*?)(?:\n\n|\n[A-Z]|\Z)"
    ]
    
    # Try each pattern until we find a match
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).strip()
            if result:
                # Clean up any trailing quotes, commas, etc.
                result = result.strip('",\'')
                return result
    
    # If we got here, none of the patterns matched
    return f"Unable to identify {field_name}"

def analyze_image_structured(image_path):
    """
    Analyze an image and return structured data about it.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: A dictionary containing:
            - theme: The overall theme of the image
            - background: Description of the background elements
            - main_subject: Description of the main subject(s)
            - tone_and_color: Description of tone, mood, and color palette
            - action_direction: Suggested action direction for the scene
    """
    import base64
    import os
    
    from openai import OpenAI
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    client = OpenAI()
    
    system_message = """
    Analyze the image and provide a structured analysis with the following components:
    
    1. Theme: The overall theme or concept of the image (1 sentence)
    2. Background Description: Detailed description of the background elements (2-3 sentences)
    3. Main Subject Description: Detailed description of the main subject(s) (2-3 sentences)
    4. Tone and Color: Description of the mood, tone, and color palette (2-3 sentences)
    5. Action Direction: A clear direction for what action or movement should happen next (1-2 sentences)
    
    Format your response as a valid JSON object with these exact keys: 
    {
        "theme": "...",
        "background": "...",
        "main_subject": "...",
        "tone_and_color": "...",
        "action_direction": "..."
    }
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000,
        response_format={"type": "json_object"}  # Force JSON response format
    )
    
    try:
        import json
        # Parse the response as JSON
        content = response.choices[0].message.content
        
        # *** CRITICAL FIX: Handle OpenAI content policy rejections ***
        if content is None:
            logger.warning("🚨 OpenAI content policy rejection: Image analysis blocked")
            return _create_fallback_analysis("Content policy violation - manual entry required")
        
        logger.info(f"Raw analysis response: {content[:100]}...")  # Log the first 100 chars for debugging
        
        analysis = json.loads(content)
        
        # Check that all required fields are present
        required_fields = ["theme", "background", "main_subject", "tone_and_color", "action_direction"]
        missing_fields = [field for field in required_fields if field not in analysis]
        
        if missing_fields:
            logger.warning(f"Missing fields in JSON response: {missing_fields}")
            # Try to extract missing fields from the text content
            for field in missing_fields:
                analysis[field] = extract_field(content, field)
        
        return analysis
        
    except json.JSONDecodeError as e:
        # If the response isn't valid JSON, extract the key components manually
        logger.warning(f"Failed to parse JSON: {str(e)}")
        content = response.choices[0].message.content
        
        # *** CRITICAL FIX: Handle content policy rejection in JSON parsing ***
        if content is None:
            logger.warning("🚨 OpenAI content policy rejection during JSON parsing")
            return _create_fallback_analysis("Content policy violation - manual entry required")
        
        # Fallback extraction method
        analysis = {
            "theme": extract_field(content, "Theme"),
            "background": extract_field(content, "Background Description"),
            "main_subject": extract_field(content, "Main Subject Description"),
            "tone_and_color": extract_field(content, "Tone and Color"),
            "action_direction": extract_field(content, "Action Direction")
        }
        
        # Log what we found
        for key, value in analysis.items():
            logger.info(f"{key}: {value}")
            
        return analysis
    
    except Exception as e:
        # *** CRITICAL FIX: Handle all other OpenAI API errors ***
        logger.warning(f"🚨 OpenAI API error during image analysis: {str(e)}")
        return _create_fallback_analysis(f"API error - manual entry required: {str(e)}")

def _create_fallback_analysis(reason: str) -> dict:
    """
    Create a fallback analysis structure when OpenAI analysis fails.
    This allows the application to continue with manual entry.
    
    Args:
        reason: Reason for the fallback
        
    Returns:
        dict: Fallback analysis structure with clear indicators for manual entry
    """
    logger.info(f"Creating fallback analysis: {reason}")
    
    return {
        "theme": f"[MANUAL ENTRY REQUIRED] - {reason}",
        "background": "[Please describe the background elements manually]",
        "main_subject": "[Please describe the main subject manually]", 
        "tone_and_color": "[Please describe the tone and color manually]",
        "action_direction": "[Please enter the desired action direction manually]",
        "_fallback": True,  # Flag to indicate this is a fallback response
        "_reason": reason   # Store the reason for debugging
    }