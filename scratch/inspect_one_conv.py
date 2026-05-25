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

c = db.agent_conversations.find_one(sort=[("updatedAt", -1)])
if c:
    c["_id"] = str(c["_id"])
    print(json.dumps(c, indent=2, cls=MongoEncoder))
else:
    print("No conversations found")
