import pymongo
from app.config import settings

def main():
    uri = settings.PRISM_MONGO_URI
    db_name = settings.PRISM_MONGO_DB or "prism"
    print(f"Connecting to MongoDB at {uri} (db: {db_name})...")
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    col = db["custom_agents"]
    
    # Find all agents
    agents = list(col.find({}))
    if not agents:
        print("No custom agents found.")
        return
        
    print(f"Found {len(agents)} custom agents. Updating project field to 'vllm-trading-bot'...")
    updated_count = 0
    for agent in agents:
        old_project = agent.get("project")
        if old_project != "vllm-trading-bot":
            col.update_one({"_id": agent["_id"]}, {"$set": {"project": "vllm-trading-bot"}})
            print(f"Updated '{agent.get('name')}' ({agent.get('agentId')}): project '{old_project}' -> 'vllm-trading-bot'")
            updated_count += 1
        else:
            print(f"Skipped '{agent.get('name')}' ({agent.get('agentId')}): already set to 'vllm-trading-bot'")
            
    print(f"Update complete. Successfully updated {updated_count} agents.")

if __name__ == "__main__":
    main()
