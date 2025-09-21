"""
Documentary styles system for AI Documentary Composer.

Provides different narrative styles with specialized prompts and configurations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class StyleType(Enum):
    """Available documentary styles."""
    DOCUMENTARY = "documentary"
    TRAVEL_VLOG = "travel_vlog"
    PERSONAL_MEMORIES = "personal_memories"


@dataclass
class StyleConfig:
    """Configuration for a documentary style."""
    name: str
    display_name: str
    tone: str
    master_prompt_template: str
    planning_prompt_template: str
    
    def get_planning_prompt(self, clips_info: str, personal_context: str = "") -> str:
        """Generate planning prompt with personal context integration."""
        context_section = ""
        if personal_context.strip():
            context_section = f"\n\nPersonal Context:\n{personal_context.strip()}\n"
        
        return self.planning_prompt_template.format(
            clips_info=clips_info,
            personal_context=context_section
        )


class DocumentaryStyles:
    """Documentary style definitions and utilities."""
    
    STYLES: Dict[StyleType, StyleConfig] = {
        StyleType.DOCUMENTARY: StyleConfig(
            name="documentary",
            display_name="Documentary",
            tone="authoritative, educational, scientifically accurate with natural storytelling",
            master_prompt_template="""Create National Geographic-style documentary narration combining scientific expertise with compelling storytelling. Deliver authoritative, educational content about the natural world, cultures, and geography. Include scientific facts, measurements, dates, and expert observations. Use specific terminology and connect local observations to global patterns. Write like a National Geographic narrator - authoritative yet accessible.""",
            planning_prompt_template="""Create National Geographic-style documentary narration that educates viewers with scientific depth and cultural insights.

Clips: {clips_info}
{personal_context}

NATIONAL GEOGRAPHIC NARRATION GUIDELINES:
- Lead with compelling scientific or historical facts
- Include specific measurements, elevations, dates, and data
- Reference geological formations, ecosystems, or cultural phenomena
- Use proper scientific nomenclature where appropriate
- Connect observations to broader environmental or societal patterns
- Explain the 'why' behind what viewers see - the science, history, or culture
- Incorporate conservation or sustainability angles when relevant

EXCELLENT EXAMPLES (National Geographic style):
- "The Amazon rainforest generates 20% of Earth's oxygen through photosynthesis. Here, a single hectare contains more tree species than all of North America, supporting an estimated 390 billion individual trees."
- "At 8,849 meters, Mount Everest grows 4 millimeters annually as the Indian tectonic plate continues its 50-million-year collision with Asia. This ongoing geological process has created Earth's highest mountain range."
- "The African savanna operates on an ancient rhythm - the annual migration of 1.5 million wildebeest following the rains, a journey that has repeated for over a million years across these grasslands."
- "Iceland's geothermal activity powers 90% of its buildings through a network of volcanic heat extraction. Here, where the North American and Eurasian plates diverge at 2.5 centimeters per year, the Earth literally tears itself apart."
- "The Great Barrier Reef, visible from space at 2,300 kilometers long, houses 25% of all marine species despite covering just 0.1% of the ocean surface. Rising temperatures have triggered five mass bleaching events since 1998."

AVOID:
- Tourist guidebook language ("beautiful", "stunning", "must-see")
- Vague descriptions without scientific backing
- Emotional or poetic language over factual content
- Surface-level observations without depth

Return JSON with authoritative, educational narration that enlightens and informs."""
        ),
        
        StyleType.TRAVEL_VLOG: StyleConfig(
            name="travel_vlog",
            display_name="Travel Vlog",
            tone="casual, first-person, friendly, conversational with specific details",
            master_prompt_template="""Create an enthusiastic travel vlog that combines personal excitement with specific location details: 'So we just arrived in Barcelona - Spain's second-largest city with 1.6 million people - and these Gaudi buildings from the early 1900s are literally everywhere! Let me show you around the Gothic Quarter.'""",
            planning_prompt_template="""Create a balanced travel vlog narrative that combines friendly tone with specific location and activity details.

Clips: {clips_info}
{personal_context}
Example tone: "We're exploring the historic center of Porto today. These narrow streets lead to the most amazing viewpoints. You can see the whole city from here, including the famous port wine cellars across the river."

IMPORTANT RULES:
- Always mention specific locations or landmarks when visible
- Describe what's actually happening in each scene
- Use variety in your expressions - avoid repeating the same excitement words
- Mix enthusiasm with informative details about what viewers are seeing
- Use "we", "I", or address viewers directly, but vary your approach

Order clips as a personal journey. Return JSON with balanced, informative yet friendly narration."""
        ),
        
        StyleType.PERSONAL_MEMORIES: StyleConfig(
            name="personal_memories",
            display_name="Personal Memories",
            tone="warm, conversational, genuine, relaxed",
            master_prompt_template="""Share personal memories naturally while including specific details when relevant. Example: 'Remember when we finally made it to Rome after that 12-hour flight? The kids spent ages trying to count all 135 steps on the Spanish Steps.'""",
            planning_prompt_template="""Create a natural, conversational memories narrative.

Clips: {clips_info}
{personal_context}
Write like you're telling a friend about your trip. Example: "That first morning in Porto was something else. We walked out of the hotel and immediately got lost in those narrow streets, but honestly, that's when the adventure really started."

Use complete, natural sentences. Be warm and genuine, not overly poetic. Share the real moments. Return JSON with conversational narration."""
        )
    }
    
    @classmethod
    def get_style(cls, style_type: StyleType) -> StyleConfig:
        """Get style configuration by type."""
        return cls.STYLES[style_type]
    
    @classmethod
    def get_style_by_name(cls, name: str) -> StyleConfig:
        """Get style configuration by name string with backward compatibility."""
        # Handle backward compatibility
        if name == "family_trip":
            name = "personal_memories"
        elif name == "memories":
            name = "personal_memories"

        for style_type in StyleType:
            if style_type.value == name:
                return cls.STYLES[style_type]
        raise ValueError(f"Unknown style: {name}")
    
    @classmethod
    def list_styles(cls) -> Dict[str, str]:
        """List all available styles with display names."""
        return {
            style_type.value: config.display_name 
            for style_type, config in cls.STYLES.items()
        }
    
    @classmethod
    def get_default_style(cls) -> StyleType:
        """Get the default style for backward compatibility."""
        return StyleType.DOCUMENTARY


def get_project_metadata_schema() -> Dict[str, Any]:
    """Get the extended project metadata schema."""
    return {
        "style": "documentary",  # Default to documentary for backward compatibility
        "project_context": "",   # Optional project-level context
        "ordering_strategy": "llm"  # "llm" or "exif"
    }


def get_clip_metadata_schema() -> Dict[str, Any]:
    """Get the extended clip metadata schema.""" 
    return {
        "filename": "",
        "personal_context": "",  # Per-clip personal context
        "exif_timestamp": None   # Optional EXIF timestamp
    }