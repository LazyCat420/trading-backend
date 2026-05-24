import pymongo

uri = "mongodb://sun:sun@10.0.0.16:27017/?directConnection=true&authSource=admin"
client = pymongo.MongoClient(uri)
print("Databases:", client.list_database_names())

for db_name in client.list_database_names():
    db = client[db_name]
    print(f"Collections in {db_name}:", db.list_collection_names())
