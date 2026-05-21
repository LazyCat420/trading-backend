import sys
import os
import json

# Set up paths for importing app and shared client codebase
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
shared_code = os.path.abspath(os.path.join(project_root, "..", "trading-client"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if shared_code not in sys.path:
    sys.path.insert(0, shared_code)

# Import the registry to trigger registration of all tools
from app.tools import registry

def main():
    snapshot = registry.get_registry_snapshot()
    # Write directly to stdout as a clean JSON block
    print(json.dumps(snapshot, indent=2))

if __name__ == "__main__":
    main()
