import sys
import os
import json
import asyncio

# Set up paths for importing app and shared client codebase
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # /home/lazycat/github/rods-project/sun/trading-service
shared_code = os.path.abspath(os.path.join(project_root, "..", "trading-client"))

if shared_code not in sys.path:
    sys.path.append(shared_code)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

from app.tools import registry

async def execute(tool_name: str, arguments_json: str):
    tool_call = {
        "id": "call_local_exec",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": arguments_json
        }
    }
    # Run the tool call using the registry
    result = await registry.execute_tool_call(
        tool_call,
        skip_permission_check=True
    )
    # Output the result as JSON
    print(json.dumps(result))

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: execute_tool.py <tool_name> <arguments_json>"}))
        sys.exit(1)
        
    tool_name = sys.argv[1]
    arguments_json = sys.argv[2]
    
    asyncio.run(execute(tool_name, arguments_json))

if __name__ == "__main__":
    main()
