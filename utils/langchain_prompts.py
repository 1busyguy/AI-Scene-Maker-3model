import os
import logging
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize model with API key
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Fixed model name
    temperature=0.7
)

def generate_cinematic_prompt(action_direction, scene_vision, frame_description, image_description, theme, background, main_subject, tone_and_color, current_chain, total_chains):
    progress_percent = int((current_chain / total_chains) * 100)
    story_phase = (
        "Establishing" if current_chain == 0 else
        "Setup" if current_chain < total_chains / 3 else
        "Development" if current_chain < total_chains * 2 / 3 else
        "Resolution"
    )

    # 🎯 CRITICAL FIX: Motion Continuity Focus
    if current_chain == 0:
        # First chain - establish the scene
        prompt_template = f"""
You are an expert cinematographer creating the OPENING shot of a cinematic sequence.

Scene Setup:
- Theme: {theme}
- Background: {background}
- Main Subject: {main_subject}
- Tone and Color: {tone_and_color}
- Action Direction: {action_direction}

Overall Vision:
{scene_vision}

Instructions:
1. Create an ESTABLISHING shot that introduces the scene naturally
2. Focus on smooth, natural movement that will continue into the next segment
3. End the movement in a way that flows naturally (avoid abrupt stops)
4. Use cinematic techniques appropriate for opening scenes
5. Write ONE concise paragraph (under 75 words) as a direct instruction

Final Cinematic Prompt:
"""
    else:
        # Subsequent chains - CRITICAL MOTION CONTINUITY
        prompt_template = f"""
You are an expert cinematographer ensuring SEAMLESS MOTION CONTINUITY between video segments.

🎯 CRITICAL: This video segment must begin EXACTLY where the previous segment ended.

Previous Frame Analysis:
{frame_description}

Motion Continuity Requirements:
- Theme: {theme} (MUST REMAIN IDENTICAL)
- Background: {background} (MUST REMAIN CONSISTENT)
- Main Subject: {main_subject} (MUST MAINTAIN EXACT APPEARANCE)
- Tone and Color: {tone_and_color} (MUST MATCH EXACTLY)
- Next Action: {action_direction}

Story Phase: {story_phase} ({progress_percent}% complete)

🎬 SEAMLESS TRANSITION INSTRUCTIONS:
1. BEGIN exactly where the previous frame ended - NO scene resets or jumps
2. CONTINUE the existing motion smoothly - if something was moving, keep it moving in the same direction initially
3. MAINTAIN identical lighting, camera position, and subject appearance
4. GRADUALLY introduce the new action while preserving motion flow
5. Think of this as one continuous shot, not separate scenes
6. NO establishing shots - we're already established
7. NO scene changes - continue the existing scene

Example Flow:
- Previous: "Character walking forward"
- Current: "Character continues walking forward, then begins to turn left"
- NOT: "Character standing in new pose" ❌

Write ONE paragraph (under 75 words) focusing on MOTION CONTINUITY:
"""

    try:
        response = llm.invoke(prompt_template)
        generated_prompt = response.content.strip()
        
        # 🎯 POST-PROCESS: Ensure motion continuity keywords are present
        if current_chain > 0:
            motion_keywords = ["continues", "smoothly", "flowing", "seamlessly"]
            if not any(keyword in generated_prompt.lower() for keyword in motion_keywords):
                # Add motion continuity if missing
                generated_prompt = f"Seamlessly continuing from the previous moment, {generated_prompt.lower()}"
        
        logger.info(f"🎬 Chain {current_chain}: Motion continuity prompt generated")
        logger.debug(f"Prompt: {generated_prompt}")
        
        return generated_prompt
        
    except Exception as e:
        logger.exception(f"Error generating prompt with LangChain: {str(e)}")
        
        # 🎯 FALLBACK: Motion-aware fallback
        if current_chain == 0:
            return f"Cinematic establishing shot showing {main_subject} in {background} with {tone_and_color} lighting, beginning {action_direction}"
        else:
            return f"Seamlessly continuing the motion from the previous frame, {main_subject} smoothly transitions into {action_direction} while maintaining the {background} setting and {tone_and_color} visual style"


def generate_motion_aware_prompt(action_direction, scene_vision, frame_description, 
                                image_description, theme, background, main_subject, 
                                tone_and_color, current_chain, total_chains, 
                                previous_motion_state=None):
    """
    🎯 NEW FUNCTION: Advanced motion-aware prompt generation
    
    Args:
        previous_motion_state: Description of the motion at the end of the previous video
    """
    
    progress_percent = int((current_chain / total_chains) * 100)
    
    if current_chain == 0:
        # First chain - establish and set motion state
        prompt_template = f"""
You are creating the opening of a cinematic sequence that must flow seamlessly into subsequent segments.

Scene Elements:
- Theme: {theme}
- Background: {background}
- Main Subject: {main_subject}
- Tone and Color: {tone_and_color}
- Opening Action: {action_direction}

Vision: {scene_vision}

Create an opening that:
1. Establishes the scene cinematically
2. Begins natural movement that can continue smoothly
3. Ends with clear directional motion or positioning
4. Uses professional cinematography

Write a concise cinematic prompt (under 75 words):
"""
        
    else:
        # Subsequent chains - MOTION CONTINUITY FOCUS
        motion_context = previous_motion_state or "continuing from the previous movement"
        
        prompt_template = f"""
You are ensuring PERFECT MOTION CONTINUITY in a cinematic sequence.

CRITICAL CONTINUITY REQUIREMENTS:
- Previous Motion State: {motion_context}
- Current Frame: {frame_description}
- Theme: {theme} (IDENTICAL)
- Background: {background} (CONSISTENT)
- Subject: {main_subject} (SAME APPEARANCE)
- Visual Style: {tone_and_color} (EXACT MATCH)

Next Motion Phase: {action_direction}
Progress: {progress_percent}% through sequence

SEAMLESS TRANSITION RULES:
1. Start EXACTLY where previous motion ended
2. Continue existing movement direction initially
3. Smoothly blend into new action direction
4. NO positional jumps or scene resets
5. Maintain identical visual elements
6. Think: "one continuous camera shot"

Create a motion-continuous prompt (under 75 words):
"""
    
    try:
        response = llm.invoke(prompt_template)
        generated_prompt = response.content.strip()
        
        # Enhance with motion continuity if needed
        if current_chain > 0 and "seamlessly" not in generated_prompt.lower():
            generated_prompt = f"Seamlessly {generated_prompt.lower()}"
        
        logger.info(f"🎬 Motion-aware prompt generated for chain {current_chain}")
        return generated_prompt
        
    except Exception as e:
        logger.exception(f"Error in motion-aware prompt generation: {str(e)}")
        
        # Motion-aware fallback
        if current_chain == 0:
            return f"{main_subject} in {background}, {action_direction}, cinematic {tone_and_color} lighting"
        else:
            return f"Seamlessly continuing the previous motion, {main_subject} flows into {action_direction}, maintaining {background} and {tone_and_color} consistency"


def extract_motion_state_from_prompt(prompt_text):
    """
    🎯 NEW FUNCTION: Extract the ending motion state from a prompt
    This helps the next chain understand how to continue smoothly
    """
    
    # Look for motion-indicating words and phrases
    motion_indicators = [
        "moving", "walking", "running", "turning", "rotating", "flowing",
        "ascending", "descending", "approaching", "receding", "tilting",
        "panning", "zooming", "tracking", "following", "circling"
    ]
    
    # Look for directional words
    directional_words = [
        "left", "right", "forward", "backward", "up", "down", "toward", "away",
        "closer", "further", "inward", "outward", "clockwise", "counterclockwise"
    ]
    
    prompt_lower = prompt_text.lower()
    
    # Find motion and direction context
    motion_context = []
    
    for indicator in motion_indicators:
        if indicator in prompt_lower:
            # Find the sentence containing this motion
            sentences = prompt_text.split('.')
            for sentence in sentences:
                if indicator in sentence.lower():
                    motion_context.append(sentence.strip())
                    break
    
    # If we found motion context, return the most relevant one
    if motion_context:
        return motion_context[-1]  # Use the last/most recent motion described
    
    return "continuing with the established movement"


def generate_continuation_with_motion_analysis(previous_prompt, action_direction, 
                                             scene_vision, frame_description, 
                                             theme, background, main_subject, 
                                             tone_and_color, current_chain, total_chains):
    """
    🎯 NEW FUNCTION: Generate continuation with automatic motion analysis
    """
    
    # Extract motion state from previous prompt
    motion_state = extract_motion_state_from_prompt(previous_prompt)
    
    # Generate motion-aware prompt
    return generate_motion_aware_prompt(
        action_direction, scene_vision, frame_description, 
        image_description="", theme=theme, background=background, 
        main_subject=main_subject, tone_and_color=tone_and_color,
        current_chain=current_chain, total_chains=total_chains,
        previous_motion_state=motion_state
    )


# 🎯 BACKWARDS COMPATIBILITY: Keep the original function name
# but now it uses motion-aware logic
def generate_cinematic_prompt_with_motion_fix(action_direction, scene_vision, frame_description, image_description, theme, background, main_subject, tone_and_color, current_chain, total_chains):
    """
    Enhanced version of the original function with motion continuity fixes
    """
    return generate_motion_aware_prompt(
        action_direction, scene_vision, frame_description, 
        image_description, theme, background, main_subject, 
        tone_and_color, current_chain, total_chains
    )