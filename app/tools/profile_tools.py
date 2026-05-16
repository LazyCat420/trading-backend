"""
Profile Memory Tools — Read/Write Persistent Agentic Session Memory
"""

from pydantic import BaseModel, Field
from app.tools.registry import registry, PermissionLevel
from app.services.session_profile import profile_memory


class UpdatePreferenceInput(BaseModel):
    key: str = Field(
        description="The name of the preference to update (e.g. 'risk_tolerance', 'favorite_sectors')."
    )
    value: str = Field(description="The new value for the preference.")


class AddNoteInput(BaseModel):
    note: str = Field(description="The note to save to the agent's persistent memory.")


@registry.register(
    name="read_profile",
    description="Read the agent's persistent profile, including user preferences, agent notes, and the context of the last trading cycle. Use this to remember past sessions.",
    tier=1,
    permission=PermissionLevel.READ_ONLY,
)
async def read_profile_tool() -> dict:
    """Retrieve the persistent local profile memory."""
    return profile_memory.get_profile()


@registry.register(
    name="update_preference",
    description="Update a specific user preference in the persistent profile.",
    tier=1,
    permission=PermissionLevel.WRITE,
    input_model=UpdatePreferenceInput,
)
async def update_preference_tool(key: str, value: str) -> dict:
    """Update a user preference."""
    profile_memory.update_preferences(key, value)
    return {"status": "success", "message": f"Preference '{key}' updated successfully."}


@registry.register(
    name="add_agent_note",
    description="Add a persistent note or memory to the agent's profile. This will be remembered across sessions.",
    tier=1,
    permission=PermissionLevel.WRITE,
    input_model=AddNoteInput,
)
async def add_agent_note_tool(note: str) -> dict:
    """Add a note to the agent's persistent profile."""
    profile_memory.add_agent_note(note)
    return {
        "status": "success",
        "message": "Note added to persistent memory successfully.",
    }
