# utils/enhanced_prompts.py
"""
Enhanced Prompt Engineering Module
Implements character locking and consistency in prompts
"""

import logging
from typing import Dict, Optional, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

logger = logging.getLogger(__name__)

class CharacterLockedPromptGenerator:
    """Generate prompts with strong character consistency constraints"""
    
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.7):
        """Initialize the prompt generator"""
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.character_description = None
        self.character_lock_template = None
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize prompt templates with character locking"""
        
        # Character lock template - to be inserted into every prompt
        self.character_lock_template = """
CRITICAL CHARACTER CONSISTENCY REQUIREMENTS:
The main character MUST maintain EXACT consistency with:
- Physical Appearance: {appearance}
- Facial Features: {facial_features}
- Clothing: {clothing}
- Distinctive Features: {distinctive_features}
- Body Type and Posture: {body_type}
- Hair Style and Color: {hair_details}

ABSOLUTE RULES:
1. The character's face, body proportions, and clothing MUST remain IDENTICAL
2. NO changes to hair color, style, or length
3. NO changes to clothing items or colors
4. NO changes to skin tone or facial structure
5. The character should appear as if filmed in continuous footage

REFERENCE ANCHORS:
- This is the SAME person continuing the SAME action
- Think of this as the next few seconds of footage of the EXACT same person
- Any deviation from the character's appearance is a FAILURE
"""

        # Main prompt template with character locking
        self.main_prompt_template = PromptTemplate(
            input_variables=[
                "action_direction", "scene_vision", "frame_description",
                "character_lock", "story_phase", "previous_actions",
                "continuity_notes"
            ],
            template="""You are a cinematographer creating a prompt for the next shot in a continuous sequence.

{character_lock}

CURRENT SCENE STATE:
Previous Frame: {frame_description}
Story Phase: {story_phase}
Previous Actions: {previous_actions}

CONTINUITY NOTES: {continuity_notes}

REQUIRED ACTION: {action_direction}

SCENE VISION: {scene_vision}

Create a concise, single-paragraph prompt (max 75 words) that:
1. EXPLICITLY mentions the character's unchanged appearance details
2. Describes ONLY the next immediate action/movement
3. Maintains perfect visual continuity
4. Uses specific character descriptors from the Character Lock

Format: Direct instruction starting with the character description, then the action.
Example: "The [exact character description] continues to [specific action]..."

Prompt:"""
        )
        
        # Validation prompt to check generated prompts
        self.validation_prompt_template = PromptTemplate(
            input_variables=["generated_prompt", "character_description"],
            template="""Analyze this prompt for character consistency:

PROMPT: {generated_prompt}

REQUIRED CHARACTER: {character_description}

Check if the prompt:
1. Explicitly mentions key character features
2. Avoids any language that could change appearance
3. Maintains continuity

Return JSON:
{{
    "has_character_description": true/false,
    "consistency_score": 0-10,
    "issues": ["list of issues"],
    "improved_prompt": "enhanced version if needed"
}}"""
        )
    
    def set_character_description(self, character_details: Dict[str, str]):
        """Set the character description to be locked across all prompts"""
        self.character_description = character_details
        
        # Create the character lock section
        self.character_lock = self.character_lock_template.format(
            appearance=character_details.get('appearance', 'consistent appearance'),
            facial_features=character_details.get('facial_features', 'consistent facial features'),
            clothing=character_details.get('clothing', 'same clothing'),
            distinctive_features=character_details.get('distinctive_features', 'same distinctive features'),
            body_type=character_details.get('body_type', 'same build and posture'),
            hair_details=character_details.get('hair_details', 'same hairstyle and color')
        )
        
        logger.info(f"Character lock set with details: {character_details}")
    
    def generate_locked_prompt(
        self,
        action_direction: str,
        scene_vision: str,
        frame_description: str,
        story_phase: str,
        previous_actions: List[str] = None,
        current_chain: int = 0,
        total_chains: int = 1
    ) -> str:
        """Generate a prompt with character locking enforced"""
        
        if not self.character_description:
            raise ValueError("Character description must be set before generating prompts")
        
        # Build continuity notes based on chain position
        continuity_notes = self._generate_continuity_notes(current_chain, total_chains)
        
        # Format previous actions
        previous_actions_str = ""
        if previous_actions:
            previous_actions_str = " → ".join(previous_actions[-3:])  # Last 3 actions
        
        # Generate the prompt
        prompt_chain = LLMChain(llm=self.llm, prompt=self.main_prompt_template)
        
        generated_prompt = prompt_chain.run(
            action_direction=action_direction,
            scene_vision=scene_vision,
            frame_description=frame_description,
            character_lock=self.character_lock,
            story_phase=story_phase,
            previous_actions=previous_actions_str,
            continuity_notes=continuity_notes
        ).strip()
        
        # Validate and enhance the prompt
        validated_prompt = self._validate_and_enhance_prompt(generated_prompt)
        
        logger.info(f"Generated character-locked prompt: {validated_prompt}")
        
        return validated_prompt
    
    def _generate_continuity_notes(self, current_chain: int, total_chains: int) -> str:
        """Generate specific continuity notes based on chain position"""
        
        progress = (current_chain / total_chains) * 100
        
        if current_chain == 0:
            return "First shot - establish character clearly with all details visible"
        elif current_chain == total_chains - 1:
            return "Final shot - maintain exact character appearance for closure"
        elif progress < 33:
            return "Early sequence - reinforce character appearance details"
        elif progress < 66:
            return "Mid sequence - maintain established character look precisely"
        else:
            return "Late sequence - ensure character consistency for impact"
    
    def _validate_and_enhance_prompt(self, prompt: str) -> str:
        """Validate the prompt has character details and enhance if needed"""
        
        validation_chain = LLMChain(llm=self.llm, prompt=self.validation_prompt_template)
        
        try:
            validation_result = validation_chain.run(
                generated_prompt=prompt,
                character_description=json.dumps(self.character_description)
            )
            
            # Parse the validation result
            result = json.loads(validation_result)
            
            if result['consistency_score'] < 8 or not result['has_character_description']:
                # Use the improved prompt if provided
                if result.get('improved_prompt'):
                    return result['improved_prompt']
                else:
                    # Manually enhance the prompt
                    return self._manually_enhance_prompt(prompt)
            
            return prompt
            
        except Exception as e:
            logger.warning(f"Validation failed, using original prompt: {str(e)}")
            return prompt
    
    def _manually_enhance_prompt(self, prompt: str) -> str:
        """Manually add character details to the prompt"""
        
        # Extract key character details
        char_desc_parts = []
        
        if self.character_description.get('appearance'):
            char_desc_parts.append(self.character_description['appearance'])
        
        if self.character_description.get('clothing'):
            char_desc_parts.append(self.character_description['clothing'])
        
        if self.character_description.get('distinctive_features'):
            char_desc_parts.append(f"with {self.character_description['distinctive_features']}")
        
        character_string = ", ".join(char_desc_parts)
        
        # Inject character description at the beginning
        if not any(desc in prompt.lower() for desc in ['person', 'character', 'individual']):
            enhanced = f"The {character_string} {prompt}"
        else:
            # Replace generic terms with specific description
            enhanced = prompt
            for generic in ['person', 'character', 'individual', 'figure']:
                if generic in enhanced.lower():
                    enhanced = enhanced.lower().replace(generic, character_string)
                    break
        
        return enhanced
    
    def create_chain_transition_prompt(
        self,
        previous_prompt: str,
        next_action: str,
        transition_type: str = "smooth"
    ) -> str:
        """Create a prompt specifically for transitions between chains"""
        
        transition_template = PromptTemplate(
            input_variables=["character_lock", "previous_prompt", "next_action", "transition_type"],
            template="""Create a transition prompt that bridges two video segments.

{character_lock}

PREVIOUS SEGMENT ENDED WITH: {previous_prompt}
NEXT ACTION NEEDED: {next_action}
TRANSITION TYPE: {transition_type}

Generate a prompt that:
1. Creates a {transition_type} transition
2. Maintains EXACT character appearance
3. Naturally progresses the action
4. Is under 50 words

Transition Prompt:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=transition_template)
        
        transition_prompt = chain.run(
            character_lock=self.character_lock,
            previous_prompt=previous_prompt,
            next_action=next_action,
            transition_type=transition_type
        ).strip()
        
        return transition_prompt
    
    def generate_character_emphasis_modifiers(self) -> List[str]:
        """Generate a list of emphasis modifiers to strengthen character consistency"""
        
        modifiers = []
        
        # Base modifiers
        modifiers.extend([
            "maintaining exact appearance",
            "with unchanged features",
            "preserving all visual details",
            "keeping consistent look"
        ])
        
        # Add specific modifiers based on character description
        if self.character_description:
            if self.character_description.get('clothing'):
                modifiers.append(f"still wearing {self.character_description['clothing']}")
            
            if self.character_description.get('hair_details'):
                modifiers.append(f"with same {self.character_description['hair_details']}")
            
            if self.character_description.get('distinctive_features'):
                modifiers.append(f"showing {self.character_description['distinctive_features']}")
        
        return modifiers
    
    def create_negative_prompt(self, existing_negative: str = "") -> str:
        """Create a negative prompt to prevent character changes"""
        
        # Core character consistency negatives (shorter list)
        negative_elements = [
            "different person",
            "changed appearance", 
            "different clothes",
            "altered features",
            "inconsistent character"
        ]
        
        # Add specific negatives based on character (but keep it short)
        if self.character_description:
            if 'hair_color' in str(self.character_description):
                # Add only one opposite hair color to keep it concise
                if 'blonde' in str(self.character_description).lower():
                    negative_elements.append("dark hair")
                elif 'dark' in str(self.character_description).lower():
                    negative_elements.append("blonde hair")
        
        # Create the character consistency negative prompt
        character_negative = ", ".join(negative_elements)
        
        # Combine with existing negative prompt, avoiding duplicates
        if existing_negative:
            # Split existing negative into individual terms
            existing_terms = [term.strip() for term in existing_negative.split(",")]
            new_terms = [term.strip() for term in character_negative.split(",")]
            
            # Only add terms that aren't already present
            unique_new_terms = [term for term in new_terms if term not in existing_terms]
            
            if unique_new_terms:
                combined = existing_negative + ", " + ", ".join(unique_new_terms)
            else:
                combined = existing_negative
            
            # Ensure the total length doesn't exceed 500 characters
            if len(combined) > 500:
                # Keep existing and add only the most important character terms
                essential_terms = ["different person", "changed appearance", "inconsistent character"]
                available_terms = [term for term in essential_terms if term not in existing_negative]
                if available_terms:
                    combined = existing_negative + ", " + ", ".join(available_terms[:2])
                else:
                    combined = existing_negative
            
            return combined
        else:
            return character_negative


# Integration with existing langchain_prompts.py
def generate_character_locked_cinematic_prompt(
    action_direction: str,
    scene_vision: str,
    frame_description: str,
    image_description: str,
    theme: str,
    background: str,
    main_subject: str,
    tone_and_color: str,
    current_chain: int,
    total_chains: int,
    character_manager: Optional['CharacterConsistencyManager'] = None
) -> str:
    """
    Enhanced version of generate_cinematic_prompt with character locking
    """
    
    # Initialize the character-locked prompt generator
    prompt_generator = CharacterLockedPromptGenerator()
    
    # If character manager is provided, use it to set character description
    if character_manager:
        char_desc = character_manager.generate_character_description()
        
        # Build body_type and hair_details from character manager data only
        # Don't use potentially corrupted main_subject parameter
        appearance = char_desc.get('appearance', '')
        if appearance:
            char_desc['body_type'] = f"consistent build and posture of {appearance}"
        else:
            char_desc['body_type'] = "consistent build and posture"
            
        # Extract hair details from distinctive_features if available
        distinctive = char_desc.get('distinctive_features', '')
        if distinctive and 'hair' in distinctive.lower():
            # Extract hair-related terms
            hair_terms = [term.strip() for term in distinctive.split(',') if 'hair' in term.lower()]
            if hair_terms:
                char_desc['hair_details'] = hair_terms[0]
            else:
                char_desc['hair_details'] = "consistent hairstyle and color"
        else:
            char_desc['hair_details'] = "consistent hairstyle and color"
        
        prompt_generator.set_character_description(char_desc)
    else:
        # Create character description from provided details (fallback only)
        char_desc = {
            'appearance': "the person in the image",
            'facial_features': "consistent facial features",
            'clothing': "clothing visible in the scene",
            'distinctive_features': "distinctive features from the image",
            'body_type': "consistent build and posture",
            'hair_details': "hairstyle and color from the original image"
        }
        prompt_generator.set_character_description(char_desc)
    
    # Determine story phase
    progress_percent = int((current_chain / total_chains) * 100)
    story_phase = (
        "Establishing" if current_chain == 0 else
        "Setup" if current_chain < total_chains / 3 else
        "Development" if current_chain < total_chains * 2 / 3 else
        "Resolution"
    )
    
    # Generate the character-locked prompt
    locked_prompt = prompt_generator.generate_locked_prompt(
        action_direction=action_direction,
        scene_vision=scene_vision,
        frame_description=frame_description,
        story_phase=story_phase,
        current_chain=current_chain,
        total_chains=total_chains
    )
    
    return locked_prompt