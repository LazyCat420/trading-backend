import pymongo
import json

uri = "mongodb://sun:sun@10.0.0.16:27017/?directConnection=true&authSource=admin"
client = pymongo.MongoClient(uri)
db = client["prism"]

def default_serializer(obj):
    import datetime
    if isinstance(obj, datetime.datetime) or isinstance(obj, datetime.date):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

for col_name in ["mcp_servers", "custom_tools", "settings", "workspaces"]:
    col = db[col_name]
    print(f"--- Documents in {col_name} ---")
    for doc in col.find():
        doc["_id"] = str(doc["_id"])
        print(json.dumps(doc, indent=2, default=default_serializer))
