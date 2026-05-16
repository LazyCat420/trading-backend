"""
Shared test fixtures for trading-cycle-backend tests.

Provides:
  - Mocked settings that don't require .env
  - Isolated ToolRegistry for testing tool provisioning
"""
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
