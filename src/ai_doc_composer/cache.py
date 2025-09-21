"""Response caching system for AI Documentary Composer.

Implements intelligent caching to avoid redundant API calls and model inference:
- Vision captioning cache (avoid re-processing identical frames)
- LLM response cache (avoid re-generating same prompts)  
- TTS synthesis cache (avoid re-synthesizing identical text)
- Persistent disk-based storage with TTL expiration

Significantly improves development iteration speed and reduces API costs.
"""

from __future__ import annotations

import json
import hashlib
import time
import statistics
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def age_hours(self) -> float:
        """Get age of cache entry in hours."""
        return (time.time() - self.timestamp) / 3600


class ResponseCache:
    """Intelligent response caching system for AI pipeline components."""
    
    def __init__(self, cache_dir: Union[str, Path] = "cache", 
                 default_ttl: Optional[float] = None):
        """Initialize cache with persistent storage.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (None = no expiration)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.default_ttl = default_ttl
        
        # Separate cache files for different components
        self.vision_cache_file = self.cache_dir / "vision_cache.json"
        self.llm_cache_file = self.cache_dir / "llm_cache.json"
        self.tts_cache_file = self.cache_dir / "tts_cache.json"
        
        # Load existing caches
        self.vision_cache = self._load_cache(self.vision_cache_file)
        self.llm_cache = self._load_cache(self.llm_cache_file)
        self.tts_cache = self._load_cache(self.tts_cache_file)
    
    def _load_cache(self, cache_file: Path) -> Dict[str, CacheEntry]:
        """Load cache from disk."""
        if not cache_file.exists():
            return {}
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Convert to CacheEntry objects
            cache = {}
            for key, entry_data in cache_data.items():
                cache[key] = CacheEntry(
                    key=entry_data['key'],
                    value=entry_data['value'],
                    timestamp=entry_data['timestamp'],
                    ttl_seconds=entry_data.get('ttl_seconds'),
                    metadata=entry_data.get('metadata', {})
                )
            
            return cache
            
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_file}: {e}")
            return {}
    
    def _save_cache(self, cache: Dict[str, CacheEntry], cache_file: Path):
        """Save cache to disk."""
        try:
            # Convert CacheEntry objects to serializable format
            cache_data = {}
            for key, entry in cache.items():
                if not entry.is_expired():  # Only save non-expired entries
                    cache_data[key] = {
                        'key': entry.key,
                        'value': entry.value,
                        'timestamp': entry.timestamp,
                        'ttl_seconds': entry.ttl_seconds,
                        'metadata': entry.metadata or {}
                    }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_file}: {e}")
    
    def _generate_key(self, *args) -> str:
        """Generate cache key from arguments."""
        # Create deterministic hash from all arguments
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def cache_vision_response(self, image_path: str, prompt: str, provider: str, 
                            response: str, ttl_seconds: Optional[float] = None) -> str:
        """Cache vision captioning response.
        
        Args:
            image_path: Path to the image file
            prompt: Caption prompt used
            provider: Vision provider (blip, gemini, etc.)
            response: Generated caption
            ttl_seconds: Cache expiration time
            
        Returns:
            Cache key for the stored response
        """
        # Include file modification time to detect changes
        try:
            mtime = Path(image_path).stat().st_mtime
        except Exception:
            mtime = 0
            
        key = self._generate_key("vision", image_path, mtime, prompt, provider)
        
        entry = CacheEntry(
            key=key,
            value=response,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
            metadata={
                'image_path': image_path,
                'prompt': prompt,
                'provider': provider,
                'file_mtime': mtime
            }
        )
        
        self.vision_cache[key] = entry
        self._save_cache(self.vision_cache, self.vision_cache_file)
        
        return key
    
    def get_vision_response(self, image_path: str, prompt: str, provider: str) -> Optional[str]:
        """Retrieve cached vision response.
        
        Args:
            image_path: Path to the image file
            prompt: Caption prompt used
            provider: Vision provider
            
        Returns:
            Cached response if available and valid, None otherwise
        """
        try:
            mtime = Path(image_path).stat().st_mtime
        except Exception:
            mtime = 0
            
        key = self._generate_key("vision", image_path, mtime, prompt, provider)
        
        if key in self.vision_cache:
            entry = self.vision_cache[key]
            if not entry.is_expired():
                return entry.value
            else:
                # Remove expired entry
                del self.vision_cache[key]
        
        return None
    
    def cache_llm_response(self, prompt: str, provider: str, model: str, 
                          temperature: float, response: Any, 
                          ttl_seconds: Optional[float] = None) -> str:
        """Cache LLM response.
        
        Args:
            prompt: Input prompt
            provider: LLM provider (ollama, gemini, etc.)
            model: Model name/ID
            temperature: Sampling temperature
            response: Generated response
            ttl_seconds: Cache expiration time
            
        Returns:
            Cache key for the stored response
        """
        key = self._generate_key("llm", prompt, provider, model, temperature)
        
        entry = CacheEntry(
            key=key,
            value=response,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
            metadata={
                'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'provider': provider,
                'model': model,
                'temperature': temperature
            }
        )
        
        self.llm_cache[key] = entry
        self._save_cache(self.llm_cache, self.llm_cache_file)
        
        return key
    
    def get_llm_response(self, prompt: str, provider: str, model: str, 
                        temperature: float) -> Optional[Any]:
        """Retrieve cached LLM response.
        
        Args:
            prompt: Input prompt
            provider: LLM provider
            model: Model name/ID
            temperature: Sampling temperature
            
        Returns:
            Cached response if available and valid, None otherwise
        """
        key = self._generate_key("llm", prompt, provider, model, temperature)
        
        if key in self.llm_cache:
            entry = self.llm_cache[key]
            if not entry.is_expired():
                return entry.value
            else:
                del self.llm_cache[key]
        
        return None
    
    def cache_tts_response(self, text: str, provider: str, language: str, 
                          speaker: str, audio_data: bytes,
                          ttl_seconds: Optional[float] = None) -> str:
        """Cache TTS synthesis result.
        
        Args:
            text: Input text to synthesize
            provider: TTS provider (xtts, gemini, etc.)
            language: Language code
            speaker: Speaker voice identifier
            audio_data: Generated audio bytes
            ttl_seconds: Cache expiration time
            
        Returns:
            Cache key for the stored audio
        """
        key = self._generate_key("tts", text, provider, language, speaker)
        
        # Save audio data to separate file
        audio_file = self.cache_dir / f"tts_{key}.wav"
        try:
            with open(audio_file, 'wb') as f:
                f.write(audio_data)
            audio_path = str(audio_file)
        except Exception as e:
            print(f"Warning: Failed to save cached audio: {e}")
            audio_path = None
        
        entry = CacheEntry(
            key=key,
            value=audio_path,  # Store path to audio file
            timestamp=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
            metadata={
                'text': text[:50] + "..." if len(text) > 50 else text,
                'provider': provider,
                'language': language,
                'speaker': speaker,
                'audio_size_bytes': len(audio_data)
            }
        )
        
        self.tts_cache[key] = entry
        self._save_cache(self.tts_cache, self.tts_cache_file)
        
        return key
    
    def get_tts_response(self, text: str, provider: str, language: str, 
                        speaker: str) -> Optional[bytes]:
        """Retrieve cached TTS audio.
        
        Args:
            text: Input text
            provider: TTS provider
            language: Language code
            speaker: Speaker voice identifier
            
        Returns:
            Cached audio bytes if available and valid, None otherwise
        """
        key = self._generate_key("tts", text, provider, language, speaker)
        
        if key in self.tts_cache:
            entry = self.tts_cache[key]
            if not entry.is_expired() and entry.value:
                try:
                    with open(entry.value, 'rb') as f:
                        return f.read()
                except Exception:
                    # Remove invalid cache entry
                    del self.tts_cache[key]
            else:
                del self.tts_cache[key]
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'vision_cache': {
                'total_entries': len(self.vision_cache),
                'expired_entries': sum(1 for e in self.vision_cache.values() if e.is_expired()),
                'avg_age_hours': statistics.mean([e.age_hours() for e in self.vision_cache.values()]) if self.vision_cache else 0
            },
            'llm_cache': {
                'total_entries': len(self.llm_cache),
                'expired_entries': sum(1 for e in self.llm_cache.values() if e.is_expired()),
                'avg_age_hours': statistics.mean([e.age_hours() for e in self.llm_cache.values()]) if self.llm_cache else 0
            },
            'tts_cache': {
                'total_entries': len(self.tts_cache),
                'expired_entries': sum(1 for e in self.tts_cache.values() if e.is_expired()),
                'avg_age_hours': statistics.mean([e.age_hours() for e in self.tts_cache.values()]) if self.tts_cache else 0
            }
        }
        
        # Calculate total cache size on disk
        total_size = 0
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        stats['total_cache_size_mb'] = total_size / (1024 * 1024)
        
        return stats
    
    def cleanup_expired(self):
        """Remove expired cache entries and orphaned files."""
        for cache_name, cache in [
            ('vision', self.vision_cache), 
            ('llm', self.llm_cache), 
            ('tts', self.tts_cache)
        ]:
            expired_keys = [key for key, entry in cache.items() if entry.is_expired()]
            for key in expired_keys:
                del cache[key]
            
            if expired_keys:
                print(f"Cleaned up {len(expired_keys)} expired {cache_name} cache entries")
        
        # Save cleaned caches
        self._save_cache(self.vision_cache, self.vision_cache_file)
        self._save_cache(self.llm_cache, self.llm_cache_file)
        self._save_cache(self.tts_cache, self.tts_cache_file)


# Global cache instance
_global_cache = None

def get_cache(cache_dir: str = "cache", default_ttl: Optional[float] = 86400) -> ResponseCache:
    """Get global cache instance.
    
    Args:
        cache_dir: Cache directory (default: 'cache')
        default_ttl: Default TTL in seconds (default: 24 hours)
        
    Returns:
        Global ResponseCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache(cache_dir, default_ttl)
    return _global_cache


if __name__ == "__main__":
    # Simple cache testing
    cache = get_cache()
    
    print("🗄️  Testing response cache system...")
    
    # Test vision caching
    cache.cache_vision_response(
        "test_image.jpg", "Describe this image", "gemini",
        "A beautiful landscape with mountains and lakes", 3600
    )
    
    result = cache.get_vision_response("test_image.jpg", "Describe this image", "gemini")
    print(f"✓ Vision cache test: {result[:50]}...")
    
    # Test LLM caching
    cache.cache_llm_response(
        "Tell me about Switzerland", "gemini", "gemini-pro", 0.3,
        {"response": "Switzerland is a beautiful country..."}
    )
    
    result = cache.get_llm_response("Tell me about Switzerland", "gemini", "gemini-pro", 0.3)
    print(f"✓ LLM cache test: {result}")
    
    # Show cache stats
    stats = cache.get_cache_stats()
    print(f"📊 Cache stats: {stats['total_cache_size_mb']:.2f} MB total")
    
    print("✅ Cache system working correctly!")