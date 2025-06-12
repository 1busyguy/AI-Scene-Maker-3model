# utils/enhanced_prompts.py - Modernized for LangChain v0.1+
"""
Enhanced prompt generation using modern LangChain LCEL (LangChain Expression Language)
Replaces deprecated LLMChain with RunnableSequence (prompt | llm pattern)
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any, Union
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
import config

logger = logging.getLogger(__name__)

class ModernEnhancedPromptGenerator:
    """
    Modern prompt generator using LangChain LCEL instead of deprecated LLMChain
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize with modern LangChain components
        
        Args:
            model_name: OpenAI model to use
            temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize modern LangChain LLM
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=config.OPENAI_API_KEY,
                max_tokens=500,  # Reasonable limit for prompts
                timeout=30       # Prevent hanging
            )
            logger.info(f"Modern LangChain LLM initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {str(e)}")
            self.llm = None
        
        # Create modern prompt templates
        self._initialize_modern_templates()
        
        # Create modern chains using LCEL
        self._create_modern_chains()
    
    def _initialize_modern_templates(self):
        """Initialize modern prompt templates"""
        
        # Main prompt template using ChatPromptTemplate
        self.main_chat_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI video prompt engineer specializing in creating cinematic, detailed prompts for AI video generation.

Your task is to transform basic input into compelling, cinematic video prompts that will produce high-quality, visually stunning videos.

REQUIREMENTS:
- Create vivid, cinematic descriptions
- Include specific visual details (lighting, camera angles, atmosphere)
- Maintain consistency with the provided theme and tone
- Keep prompts concise but descriptive (max 150 words)
- Focus on visual elements that translate well to video
- Include motion and dynamic elements when appropriate

QUALITY FOCUS:
- Emphasize "high quality", "detailed", "cinematic" elements
- Specify lighting conditions (golden hour, dramatic lighting, etc.)
- Include camera movements or angles when relevant
- Add atmospheric details (mist, particles, weather, etc.)"""),
            
            ("human", """Transform this into a cinematic video prompt:

Action/Direction: {action_direction}
Theme: {theme}
Background/Setting: {background}
Main Subject: {main_subject}
Tone and Color: {tone_and_color}
Scene Vision: {scene_vision}

Create a compelling video prompt that captures the essence of this vision with cinematic quality.""")
        ])
        
        # Style-specific templates
        self.style_templates = {
            "cinematic": ChatPromptTemplate.from_messages([
                ("system", "You specialize in cinematic video prompts with film-like quality, dramatic lighting, and professional cinematography."),
                ("human", "Create a cinematic video prompt: {input_text}")
            ]),
            
            "artistic": ChatPromptTemplate.from_messages([
                ("system", "You create artistic, visually striking video prompts with unique visual styles, creative compositions, and artistic flair."),
                ("human", "Create an artistic video prompt: {input_text}")
            ]),
            
            "natural": ChatPromptTemplate.from_messages([
                ("system", "You create natural, realistic video prompts focusing on authentic lighting, natural movements, and realistic scenarios."),
                ("human", "Create a natural, realistic video prompt: {input_text}")
            ]),
            
            "dynamic": ChatPromptTemplate.from_messages([
                ("system", "You specialize in dynamic, action-packed video prompts with movement, energy, and visual excitement."),
                ("human", "Create a dynamic video prompt with movement: {input_text}")
            ])
        }
        
        # Continuation prompt template
        self.continuation_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating video prompt continuations that maintain visual consistency while progressing the narrative.

REQUIREMENTS:
- Maintain visual consistency with the previous prompt
- Create smooth narrative progression
- Keep the same tone, lighting, and atmosphere
- Introduce subtle changes or developments
- Ensure the continuation feels natural and connected"""),
            
            ("human", """Create a continuation prompt based on:

Previous prompt: {previous_prompt}
Continuation direction: {continuation_direction}
Maintain consistency in: {consistency_elements}

Generate a prompt that continues the story while maintaining visual consistency.""")
        ])
    
    def _create_modern_chains(self):
        """Create modern LangChain chains using LCEL"""
        if not self.llm:
            logger.error("Cannot create chains without LLM")
            return
        
        try:
            # Output parser
            output_parser = StrOutputParser()
            
            # Main chain using LCEL (prompt | llm | parser)
            self.main_chain = self.main_chat_template | self.llm | output_parser
            
            # Style-specific chains
            self.style_chains = {}
            for style, template in self.style_templates.items():
                self.style_chains[style] = template | self.llm | output_parser
            
            # Continuation chain
            self.continuation_chain = self.continuation_template | self.llm | output_parser
            
            logger.info("Modern LangChain LCEL chains created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create modern chains: {str(e)}")
            self.main_chain = None
            self.style_chains = {}
            self.continuation_chain = None
    
    def generate_enhanced_prompt(self, action_direction: str, theme: str = "", 
                                background: str = "", main_subject: str = "", 
                                tone_and_color: str = "", scene_vision: str = "",
                                style: str = "cinematic") -> str:
        """
        Generate enhanced prompt using modern LangChain LCEL
        
        Args:
            action_direction: Main action or direction
            theme: Overall theme
            background: Background/setting description
            main_subject: Main subject description
            tone_and_color: Tone and color preferences
            scene_vision: Overall scene vision
            style: Style preference (cinematic, artistic, natural, dynamic)
            
        Returns:
            Enhanced prompt string
        """
        if not self.main_chain:
            logger.warning("Modern chain not available, using fallback")
            return self._fallback_prompt_generation(action_direction, theme, background, 
                                                  main_subject, tone_and_color, scene_vision)
        
        try:
            # Prepare input data
            input_data = {
                "action_direction": action_direction or "cinematic scene",
                "theme": theme or "dramatic",
                "background": background or "atmospheric setting",
                "main_subject": main_subject or "the main focus",
                "tone_and_color": tone_and_color or "rich, cinematic colors",
                "scene_vision": scene_vision or "visually compelling"
            }
            
            # Use modern invoke method instead of deprecated run
            logger.info(f"Generating enhanced prompt with style: {style}")
            start_time = time.time()
            
            # Generate using main chain
            enhanced_prompt = self.main_chain.invoke(input_data)
            
            # Apply style-specific enhancement if requested and available
            if style in self.style_chains and style != "cinematic":
                style_input = {"input_text": enhanced_prompt}
                enhanced_prompt = self.style_chains[style].invoke(style_input)
            
            generation_time = time.time() - start_time
            logger.info(f"Enhanced prompt generated in {generation_time:.2f}s")
            
            # Post-process the prompt
            enhanced_prompt = self._post_process_prompt(enhanced_prompt)
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Enhanced prompt generation failed: {str(e)}")
            return self._fallback_prompt_generation(action_direction, theme, background, 
                                                  main_subject, tone_and_color, scene_vision)
    
    def generate_continuation_prompt(self, previous_prompt: str, continuation_direction: str,
                                   consistency_elements: str = "lighting, atmosphere, style") -> str:
        """
        Generate continuation prompt using modern LangChain LCEL
        
        Args:
            previous_prompt: The previous prompt to continue from
            continuation_direction: Direction for the continuation
            consistency_elements: Elements to maintain consistency in
            
        Returns:
            Continuation prompt string
        """
        if not self.continuation_chain:
            logger.warning("Continuation chain not available, using fallback")
            return self._fallback_continuation(previous_prompt, continuation_direction)
        
        try:
            input_data = {
                "previous_prompt": previous_prompt,
                "continuation_direction": continuation_direction,
                "consistency_elements": consistency_elements
            }
            
            logger.info("Generating continuation prompt")
            start_time = time.time()
            
            # Use modern invoke method
            continuation_prompt = self.continuation_chain.invoke(input_data)
            
            generation_time = time.time() - start_time
            logger.info(f"Continuation prompt generated in {generation_time:.2f}s")
            
            # Post-process
            continuation_prompt = self._post_process_prompt(continuation_prompt)
            
            return continuation_prompt
            
        except Exception as e:
            logger.error(f"Continuation prompt generation failed: {str(e)}")
            return self._fallback_continuation(previous_prompt, continuation_direction)
    
    def _post_process_prompt(self, prompt: str) -> str:
        """Post-process generated prompt for quality and consistency"""
        if not prompt:
            return prompt
        
        # Clean up the prompt
        prompt = prompt.strip()
        
        # Ensure prompt isn't too long (most video models have limits)
        if len(prompt) > 400:
            # Try to truncate at a sentence boundary
            sentences = prompt.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= 400:
                    truncated += sentence + '. '
                else:
                    break
            if truncated:
                prompt = truncated.rstrip('. ') + '.'
            else:
                prompt = prompt[:400] + "..."
        
        # Ensure quality keywords are present
        quality_keywords = ["high quality", "detailed", "cinematic", "professional"]
        prompt_lower = prompt.lower()
        
        if not any(keyword in prompt_lower for keyword in quality_keywords):
            prompt = f"{prompt}, high quality, detailed"
        
        return prompt
    
    def _fallback_prompt_generation(self, action_direction: str, theme: str, 
                                   background: str, main_subject: str, 
                                   tone_and_color: str, scene_vision: str) -> str:
        """Fallback prompt generation without LLM"""
        logger.info("Using fallback prompt generation")
        
        # Combine elements intelligently
        elements = []
        
        if main_subject:
            elements.append(main_subject)
        
        if action_direction:
            elements.append(f"performing {action_direction}")
        
        if background:
            elements.append(f"in {background}")
        
        if theme:
            elements.append(f"with {theme} theme")
        
        if tone_and_color:
            elements.append(f"featuring {tone_and_color}")
        
        if scene_vision:
            elements.append(f"creating {scene_vision}")
        
        # Add quality enhancements
        quality_elements = [
            "high quality",
            "detailed",
            "cinematic lighting",
            "professional cinematography"
        ]
        
        # Combine everything
        base_prompt = ", ".join(elements) if elements else "cinematic scene"
        quality_suffix = ", ".join(quality_elements)
        
        return f"{base_prompt}, {quality_suffix}"
    
    def _fallback_continuation(self, previous_prompt: str, continuation_direction: str) -> str:
        """Fallback continuation without LLM"""
        logger.info("Using fallback continuation generation")
        
        # Extract key elements from previous prompt
        base_elements = previous_prompt.split(',')[:3]  # Take first 3 elements
        base = ', '.join(base_elements)
        
        # Add continuation
        continuation = f"{base}, then {continuation_direction}, maintaining consistent lighting and atmosphere, high quality, detailed"
        
        return continuation
    
    def batch_generate_prompts(self, prompt_requests: List[Dict[str, Any]]) -> List[str]:
        """
        Generate multiple prompts efficiently using modern batch processing
        
        Args:
            prompt_requests: List of dictionaries with prompt generation parameters
            
        Returns:
            List of generated prompts
        """
        if not self.main_chain:
            logger.warning("Chain not available, using fallback for batch generation")
            return [self._fallback_prompt_generation(**req) for req in prompt_requests]
        
        results = []
        
        try:
            logger.info(f"Batch generating {len(prompt_requests)} prompts")
            start_time = time.time()
            
            # Process in batches to avoid rate limits
            batch_size = 5
            for i in range(0, len(prompt_requests), batch_size):
                batch = prompt_requests[i:i + batch_size]
                
                batch_results = []
                for req in batch:
                    try:
                        # Use the regular generation method
                        result = self.generate_enhanced_prompt(**req)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to generate prompt in batch: {str(e)}")
                        # Use fallback for failed requests
                        fallback = self._fallback_prompt_generation(**req)
                        batch_results.append(fallback)
                
                results.extend(batch_results)
                
                # Small delay between batches to respect rate limits
                if i + batch_size < len(prompt_requests):
                    time.sleep(1)
            
            total_time = time.time() - start_time
            logger.info(f"Batch generation completed in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            # Fallback to individual generation
            return [self._fallback_prompt_generation(**req) for req in prompt_requests]
    
    def is_available(self) -> bool:
        """Check if the modern prompt generator is fully functional"""
        return self.llm is not None and self.main_chain is not None

# Global instance
_modern_prompt_generator = None

def get_modern_prompt_generator() -> ModernEnhancedPromptGenerator:
    """Get global modern prompt generator instance"""
    global _modern_prompt_generator
    
    if _modern_prompt_generator is None:
        _modern_prompt_generator = ModernEnhancedPromptGenerator()
    
    return _modern_prompt_generator

# Backwards compatibility functions
def generate_enhanced_prompt(action_direction: str, theme: str = "", background: str = "", 
                           main_subject: str = "", tone_and_color: str = "", 
                           scene_vision: str = "", style: str = "cinematic") -> str:
    """Backwards compatibility wrapper"""
    generator = get_modern_prompt_generator()
    return generator.generate_enhanced_prompt(
        action_direction, theme, background, main_subject, 
        tone_and_color, scene_vision, style
    )

def generate_continuation_prompt(previous_prompt: str, continuation_direction: str) -> str:
    """Backwards compatibility wrapper"""
    generator = get_modern_prompt_generator()
    return generator.generate_continuation_prompt(previous_prompt, continuation_direction)

# Migration helper function
def migrate_from_old_enhanced_prompts():
    """
    Helper function to migrate from old LLMChain-based implementation
    This can be called during initialization to ensure smooth transition
    """
    try:
        generator = get_modern_prompt_generator()
        if generator.is_available():
            logger.info("✅ Modern enhanced prompts successfully migrated from deprecated LLMChain")
            return True
        else:
            logger.warning("⚠️ Modern enhanced prompts initialized but LLM unavailable")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to migrate enhanced prompts: {str(e)}")
        return False

# Auto-migration on import
if __name__ != "__main__":
    migrate_from_old_enhanced_prompts()