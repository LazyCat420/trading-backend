import os
import pytest
from app.config.config import settings

def test_production_db_config():
    """BENCHMARK: The database should point to the NAS, never localhost in prod."""
    # This ensures the .env file is correctly parsing the NAS IP
    assert "localhost" not in settings.DATABASE_URL, "DATABASE_URL is falling back to localhost!"
    assert "10.0.0.16" in settings.DATABASE_URL, "DATABASE_URL is not pointing to the NAS instance."
