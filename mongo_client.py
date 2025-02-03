from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_database():
    """Get MongoDB database connection"""
    try:
        # Get MongoDB connection string from environment variable
        connection_string = os.getenv('MONGODB_URI')
        
        # Create a connection using MongoClient
        client = MongoClient(connection_string)
        
        # Test connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB")
        

        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None 