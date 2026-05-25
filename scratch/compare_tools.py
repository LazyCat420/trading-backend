import json
import re

lazy_tools_file = "/home/lazycat/github/rods-project/sun/lazy-tool-service/tool_schemas.json"
tools_service_file = "/home/lazycat/github/rods-project/sun/tools-service/tool_names.txt"

with open(lazy_tools_file) as f:
    lazy_schemas = json.load(f)

lazy_names = [t["name"] for t in lazy_schemas]

tools_service_names = []
with open(tools_service_file) as f:
    for line in f:
        # Match lines like: 2187:    name: "get_macro_data",
        match = re.search(r'name:\s*"([^"]+)"', line)
        if match:
            tools_service_names.append(match.group(1))

print("Lazy Tools Names (total {}):".format(len(lazy_names)))
print(sorted(lazy_names))
print("\nTools Service Names (total {}):".format(len(tools_service_names)))
print(sorted(tools_service_names))

duplicates = set(lazy_names).intersection(set(tools_service_names))
print("\nDuplicate tool names:")
print(sorted(list(duplicates)))
