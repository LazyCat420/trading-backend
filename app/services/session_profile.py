"""
Local Session Profile Memory.

Manages persistent profile and session state via a local JSON file on disk.
This functions as the agent's "long term memory" for preferences and
remembering the context of the last cycle.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_FILE = Path("data") / "session_profile.json"


class LocalProfileMemory:
    """Manages the persistent disk-based JSON memory for the agent."""

    @staticmethod
    def _ensure_file():
        if not PROFILE_FILE.parent.exists():
            PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)

        if not PROFILE_FILE.exists():
            default_state = {
                "user_preferences": {},
                "last_trade_context": {},
                "agent_notes": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(PROFILE_FILE, "w", encoding="utf-8") as f:
                json.dump(default_state, f, indent=4)

    @classmethod
    def get_profile(cls) -> dict:
        """Read the entire profile from disk."""
        cls._ensure_file()
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read profile memory: %s", e)
            return {}

    @classmethod
    def save_profile(cls, data: dict):
        """Save the entire profile to disk."""
        cls._ensure_file()
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(PROFILE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error("Failed to save profile memory: %s", e)

    @classmethod
    def update_preferences(cls, key: str, value: any):
        """Update a specific user preference."""
        profile = cls.get_profile()
        prefs = profile.get("user_preferences", {})
        prefs[key] = value
        profile["user_preferences"] = prefs
        cls.save_profile(profile)

    @classmethod
    def add_agent_note(cls, note: str):
        """Add a general note or memory for the agent."""
        profile = cls.get_profile()
        notes = profile.get("agent_notes", [])
        notes.append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "note": note}
        )
        profile["agent_notes"] = notes
        cls.save_profile(profile)

    @classmethod
    def set_last_trade_context(cls, context: dict):
        """Save the context of the last completed trading cycle."""
        profile = cls.get_profile()
        profile["last_trade_context"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
        }
        cls.save_profile(profile)

    @classmethod
    def get_last_trade_context(cls) -> dict:
        """Retrieve the last trading cycle context."""
        profile = cls.get_profile()
        return profile.get("last_trade_context", {})


profile_memory = LocalProfileMemory()
