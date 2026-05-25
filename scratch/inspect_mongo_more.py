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

print("Collections:")
print(db.list_collection_names())

# Inspect some recent request logs or chats
print("\n--- RECENT REQUESTS ---")
if "requests" in db.list_collection_names():
    for r in db.requests.find().sort("createdAt", -1).limit(5):
        r["_id"] = str(r["_id"])
        print(json.dumps(r, indent=2, cls=MongoEncoder))
        print("="*60)

print("\n--- RECENT CONVERSATIONS ---")
if "conversations" in db.list_collection_names():
    for c in db.conversations.find().sort("updatedAt", -1).limit(3):
        c["_id"] = str(c["_id"])
        print(json.dumps(c, indent=2, cls=MongoEncoder))
        print("="*60)
