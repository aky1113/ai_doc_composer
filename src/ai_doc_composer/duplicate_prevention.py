"""
Utility functions for preventing duplicate phrases in documentary narration.
"""

import re
from difflib import SequenceMatcher


def detect_duplicate_phrases(current_text: str, all_previous_texts: list[str], threshold: float = 0.5) -> tuple[bool, str]:
    """
    Detect if current text contains significant duplicate phrases from previous texts.

    Args:
        current_text: The new narration to check
        all_previous_texts: List of all previous narration texts
        threshold: Similarity threshold (0.5 = 50% similar)

    Returns:
        (has_duplicate, duplicate_info) - True if duplicate found, and info about the duplicate
    """
    if not all_previous_texts or not current_text:
        return False, ""

    current_text_lower = current_text.lower().strip()

    # Split into meaningful phrases (at least 4 words each)
    def extract_phrases(text: str, min_words: int = 4) -> set[str]:
        """Extract significant phrases from text."""
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text.lower())
        phrases = set()

        for sentence in sentences:
            words = sentence.strip().split()
            # Extract all subphrases of at least min_words
            for i in range(len(words) - min_words + 1):
                for j in range(i + min_words, min(i + 15, len(words) + 1)):  # Max phrase length of 15 words
                    phrase = ' '.join(words[i:j])
                    phrases.add(phrase)

        return phrases

    current_phrases = extract_phrases(current_text)

    for idx, prev_text in enumerate(all_previous_texts):
        prev_text_lower = prev_text.lower().strip()

        # Check for exact duplicate (or near-exact)
        similarity = SequenceMatcher(None, current_text_lower, prev_text_lower).ratio()
        if similarity > 0.8:  # 80% similar = basically duplicate
            return True, f"Text is {similarity*100:.0f}% similar to clip {idx+1}"

        # Check for significant phrase overlap
        prev_phrases = extract_phrases(prev_text)
        common_phrases = current_phrases & prev_phrases

        # Filter to only meaningful overlaps (not just common phrases)
        significant_overlaps = [p for p in common_phrases if len(p.split()) >= 5]

        if significant_overlaps:
            longest_overlap = max(significant_overlaps, key=len)
            if len(longest_overlap.split()) >= 7:  # 7+ word exact match is suspicious
                return True, f"Contains duplicate phrase from clip {idx+1}: '{longest_overlap[:50]}...'"

    return False, ""


def remove_duplicate_sentences(current_text: str, previous_text: str) -> str:
    """
    Remove sentences from current text that appear in previous text.

    Args:
        current_text: The new narration
        previous_text: The previous narration

    Returns:
        Current text with duplicate sentences removed
    """
    if not previous_text or not current_text:
        return current_text

    # Split into sentences
    current_sentences = re.split(r'(?<=[.!?])\s+', current_text.strip())
    prev_sentences = re.split(r'(?<=[.!?])\s+', previous_text.strip())

    # Normalize for comparison
    prev_normalized = {s.lower().strip() for s in prev_sentences if s.strip()}

    # Keep only non-duplicate sentences
    filtered = []
    for sent in current_sentences:
        if sent.strip() and sent.lower().strip() not in prev_normalized:
            filtered.append(sent)

    result = ' '.join(filtered).strip()
    return result if result else current_text  # Return original if everything was filtered


def check_phrase_diversity(all_texts: list[str]) -> dict[str, list[int]]:
    """
    Check for repeated phrases across all narration texts.

    Args:
        all_texts: List of all narration texts

    Returns:
        Dictionary of repeated phrases and which clips they appear in
    """
    phrase_locations = {}

    for idx, text in enumerate(all_texts):
        if not text:
            continue

        # Extract phrases of 4-8 words
        words = text.lower().split()
        for length in range(4, min(9, len(words) + 1)):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i+length])
                # Skip very common phrases
                if phrase.startswith(('the', 'a', 'an', 'this', 'that', 'it')):
                    if phrase not in phrase_locations:
                        phrase_locations[phrase] = []
                    phrase_locations[phrase].append(idx + 1)

    # Filter to only repeated phrases
    repeated = {phrase: clips for phrase, clips in phrase_locations.items() if len(clips) > 1}

    return repeated


def create_duplicate_prevention_prompt(previous_narrations: list[str]) -> str:
    """
    Create a prompt section that explicitly lists phrases to avoid.

    Args:
        previous_narrations: List of all previous narration texts

    Returns:
        Prompt text warning about phrases to avoid
    """
    if not previous_narrations:
        return ""

    # Extract key phrases from previous narrations
    avoid_phrases = set()

    for text in previous_narrations[-3:]:  # Focus on recent clips
        if not text:
            continue

        # Extract distinctive phrases (5-10 words)
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= 5:
                # Add the opening phrase
                avoid_phrases.add(' '.join(words[:7]))
                # Add any distinctive middle phrases
                if len(words) > 10:
                    avoid_phrases.add(' '.join(words[3:10]))

    if not avoid_phrases:
        return ""

    # Format as a warning
    phrases_list = list(avoid_phrases)[:5]  # Limit to 5 most important
    phrases_formatted = '\n'.join(f"  - \"{p}\"" for p in phrases_list)

    return f"""
CRITICAL: You must create ORIGINAL narration. Do NOT repeat these phrases from previous clips:
{phrases_formatted}

Create fresh, unique description even if the scenes are similar. Use different words and structure.
"""