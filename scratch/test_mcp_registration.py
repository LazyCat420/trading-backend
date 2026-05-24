import sys
import os

# Add app to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variable to make it read correct env values
os.environ["ENV_FILE"] = os.path.join(project_root, ".env")

import logging
logging.basicConfig(level=logging.INFO)

from app.services.boot_service import BootService

print("Running MCP server registration...")
BootService._register_mcp_servers()
print("Done.")
