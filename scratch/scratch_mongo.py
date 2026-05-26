from pymongo import MongoClient
import json

uri = "mongodb://rodrigo:jLhNbFA3kt9k7BnwL-sW@192.168.86.2:27017/?directConnection=true&authSource=admin"
client = MongoClient(uri)
db = client["prism"]
collection = db["custom_agents"]

agents = list(collection.find({}))
for agent in agents:
    agent["_id"] = str(agent["_id"])
    print(f"--- Agent: {agent.get('name')} ({agent.get('agentId')}) ---")
    print(f"Description: {agent.get('description')}")
    print(f"Identity: {agent.get('identity')[:200]}...")
    print(f"Enabled Tools: {agent.get('enabledTools')}")
    print("-" * 40)
