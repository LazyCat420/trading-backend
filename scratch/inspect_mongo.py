import pymongo
import json
from datetime import datetime

class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

uri = "mongodb://sun:sun@10.0.0.16:27017/?directConnection=true&authSource=admin"
client = pymongo.MongoClient(uri)
db = client.get_database("prism")

print("--- CUSTOM AGENTS ---")
try:
    for a in db.custom_agents.find():
        a["_id"] = str(a["_id"])
        print(json.dumps(a, indent=2, cls=MongoEncoder))
except Exception as e:
    print("Error reading custom_agents:", e)

print("\n--- MCP SERVERS ---")
try:
    for s in db.mcp_servers.find():
        s["_id"] = str(s["_id"])
        print(json.dumps(s, indent=2, cls=MongoEncoder))
except Exception as e:
    print("Error reading mcp_servers:", e)
