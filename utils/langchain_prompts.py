import os
import logging
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize model with API key
llm = ChatOpenAI(
    model="gpt-4.1-mini",
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

    prompt_template = f"""
You are an expert cinematographer creating a concise, visually continuous prompt for an AI video generator.

Previous Frame Description:
{frame_description}

Scene Analysis:
- Theme: {theme}
- Background Description: {background}
- Main Subject Description: {main_subject}
- Tone and Color: {tone_and_color}
- Action Direction: {action_direction}

Overall Scene Vision:
{scene_vision}

Story Phase: {story_phase} ({progress_percent}% complete)

Instructions:
1. MAINTAIN STRICT VISUAL CONTINUITY by explicitly referencing the Main Subject, Background, and Tone/Color from the analysis.
2. The Theme and Main Subject must remain consistent throughout all frames.
3. The Background elements should remain consistent with only small logical changes to support the action.
4. Progress ONLY the action while keeping visual elements consistent with previous frames.
5. Suggest specific cinematography techniques (camera angles, movements, framing) suitable for the current story phase.
6. Write exactly ONE concise paragraph (under 75 words) as a direct instruction without phrases like "I want" or "The video should".

Final Cinematic Prompt:
"""

    try:
        response = llm.invoke(prompt_template)
        return response.content.strip()
    except Exception as e:
        logger.exception(f"Error generating prompt with LangChain: {str(e)}")
        return f"Continue from the previous frame, showing {action_direction} in a {story_phase.lower()} shot while maintaining the {main_subject} against the {background} with {tone_and_color} visual style." 