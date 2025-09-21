"""Gemini API Quota Management System."""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import typer

# Updated quotas based on actual free tier limits (as of 2025)
QUOTAS = {
    "gemini-2.5-flash-lite": {"rpm": 15, "daily": 1000},  # New model with good quota!
    "gemini-2.5-flash": {"rpm": 10, "daily": 250},        # New model with moderate quota
    "gemini-2.0-flash": {"rpm": 10, "daily": 200},        # Confirmed from error message
    "gemini-2.0-flash-exp": {"rpm": 10, "daily": 200},    # Experimental version
    "gemini-1.5-flash": {"rpm": 15, "daily": 1500},       # Legacy with higher limits
    "gemini-1.5-pro": {"rpm": 2, "daily": 50},            # Premium model, lower limits
}

# Model priority order (prefer models with higher quotas first)
MODEL_PRIORITY = [
    "gemini-1.5-flash",       # Highest daily quota (1500)
    "gemini-2.5-flash-lite",  # Second highest (1000) - NEW!
    "gemini-2.5-flash",       # Moderate quota (250) - NEW!
    "gemini-2.0-flash",       # Lower quota (200)
    "gemini-2.0-flash-exp",   # Experimental fallback (200)
    "gemini-1.5-pro",         # Premium model as last resort (50)
]

class QuotaManager:
    """Manages API quotas for Gemini models."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize quota manager with optional cache directory."""
        self.cache_dir = cache_dir or Path.home() / ".ai_doc_composer" / "quota_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quota_file = self.cache_dir / "gemini_quota.json"
        self.usage = self._load_usage()

    def _load_usage(self) -> Dict:
        """Load quota usage from cache."""
        if self.quota_file.exists():
            try:
                with open(self.quota_file) as f:
                    data = json.load(f)
                    # Check if data is from today
                    today = datetime.now().strftime("%Y-%m-%d")
                    if data.get("date") != today:
                        # Reset daily counters
                        return self._reset_daily()
                    return data
            except Exception:
                return self._reset_daily()
        return self._reset_daily()

    def _reset_daily(self) -> Dict:
        """Reset daily usage counters."""
        today = datetime.now().strftime("%Y-%m-%d")
        data = {
            "date": today,
            "models": {
                model: {
                    "daily_used": 0,
                    "last_request_time": 0,
                    "rpm_window": []  # Track request times for RPM limiting
                }
                for model in QUOTAS.keys()
            }
        }
        self._save_usage(data)
        return data

    def _save_usage(self, data: Dict) -> None:
        """Save quota usage to cache."""
        try:
            with open(self.quota_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            typer.echo(f"Warning: Could not save quota cache: {e}")

    def _clean_rpm_window(self, model: str) -> None:
        """Remove requests older than 1 minute from RPM window."""
        if model not in self.usage["models"]:
            return

        current_time = time.time()
        minute_ago = current_time - 60

        # Keep only requests within the last minute
        self.usage["models"][model]["rpm_window"] = [
            t for t in self.usage["models"][model]["rpm_window"]
            if t > minute_ago
        ]

    def check_quota(self, model: str) -> Tuple[bool, str, float]:
        """
        Check if model has available quota.
        Returns: (can_use, reason, wait_time_seconds)
        """
        if model not in QUOTAS:
            return False, f"Unknown model: {model}", 0

        model_data = self.usage["models"].get(model, {})
        quota = QUOTAS[model]

        # Check daily limit
        if model_data.get("daily_used", 0) >= quota["daily"]:
            # Calculate time until midnight for reset
            now = datetime.now()
            midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            wait_seconds = (midnight - now).total_seconds()
            return False, f"Daily quota exceeded ({quota['daily']} requests)", wait_seconds

        # Clean and check RPM window
        self._clean_rpm_window(model)
        rpm_count = len(model_data.get("rpm_window", []))

        if rpm_count >= quota["rpm"]:
            # Calculate wait time until oldest request expires
            oldest_request = min(model_data["rpm_window"])
            wait_seconds = max(0, 60 - (time.time() - oldest_request))
            return False, f"Rate limit exceeded ({quota['rpm']} requests per minute)", wait_seconds

        return True, "OK", 0

    def record_usage(self, model: str, count: int = 1) -> None:
        """Record API usage for a model."""
        if model not in self.usage["models"]:
            self.usage["models"][model] = {
                "daily_used": 0,
                "last_request_time": 0,
                "rpm_window": []
            }

        current_time = time.time()
        self.usage["models"][model]["daily_used"] += count
        self.usage["models"][model]["last_request_time"] = current_time
        self.usage["models"][model]["rpm_window"].append(current_time)

        # Clean old entries from RPM window
        self._clean_rpm_window(model)

        # Save updated usage
        self._save_usage(self.usage)

    def mark_exhausted(self, model: str) -> None:
        """Mark a model as having exhausted its daily quota."""
        if model not in self.usage["models"]:
            self.usage["models"][model] = {
                "daily_used": 0,
                "last_request_time": 0,
                "rpm_window": []
            }

        # Set daily usage to the quota limit
        quota_limit = QUOTAS.get(model, {}).get("daily", 200)
        self.usage["models"][model]["daily_used"] = quota_limit
        self._save_usage(self.usage)

    def get_available_model(self, preferred_model: Optional[str] = None) -> Tuple[Optional[str], str]:
        """
        Get an available model, trying preferred first, then fallbacks.
        Returns: (model_name, status_message)
        """
        models_to_try = []

        # Add preferred model first if specified
        if preferred_model and preferred_model in QUOTAS:
            models_to_try.append(preferred_model)

        # Add priority models
        models_to_try.extend([m for m in MODEL_PRIORITY if m not in models_to_try])

        # Try each model
        for model in models_to_try:
            can_use, reason, wait_time = self.check_quota(model)
            if can_use:
                return model, f"Using model: {model}"

        # No models available
        return None, "All models have exceeded their quotas. Please wait or upgrade to a paid tier."

    def get_usage_summary(self) -> str:
        """Get a summary of current quota usage."""
        lines = ["📊 Gemini API Quota Usage:\n"]

        for model in MODEL_PRIORITY:
            if model not in QUOTAS:
                continue

            quota = QUOTAS[model]
            model_data = self.usage["models"].get(model, {})
            daily_used = model_data.get("daily_used", 0)

            # Clean RPM window for accurate count
            self._clean_rpm_window(model)
            rpm_used = len(model_data.get("rpm_window", []))

            daily_pct = (daily_used / quota["daily"] * 100) if quota["daily"] > 0 else 0
            rpm_pct = (rpm_used / quota["rpm"] * 100) if quota["rpm"] > 0 else 0

            status = "✅" if daily_pct < 80 else "⚠️" if daily_pct < 100 else "❌"

            lines.append(f"{status} **{model}**")
            lines.append(f"   Daily: {daily_used}/{quota['daily']} ({daily_pct:.0f}%)")
            lines.append(f"   RPM: {rpm_used}/{quota['rpm']} ({rpm_pct:.0f}%)")

        return "\n".join(lines)

    def reset_model_quota(self, model: str) -> None:
        """Manually reset quota for a specific model (for testing)."""
        if model in self.usage["models"]:
            self.usage["models"][model] = {
                "daily_used": 0,
                "last_request_time": 0,
                "rpm_window": []
            }
            self._save_usage(self.usage)