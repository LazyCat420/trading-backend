import pymongo
import json
from datetime import datetime

class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__str__"):
            return str(obj)
        return super().default(obj)

uri = "mongodb://sun:sun@10.0.0.16:27017/?directConnection=true&authSource=admin"
client = pymongo.MongoClient(uri)
db = client.get_database("prism")

print("--- RECENT AGENT CONVERSATIONS ---")
if "agent_conversations" in db.list_collection_names():
    for c in db.agent_conversations.find().sort("updatedAt", -1).limit(5):
        c["_id"] = str(c["_id"])
        # truncate messages if they are too long
        if "messages" in c:
            c["messages"] = c["messages"][-2:] # just print the last 2 messages
        print(json.dumps(c, indent=2, cls=MongoEncoder))
        print("="*60)
