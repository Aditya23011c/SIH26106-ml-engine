"""
Quick standalone test to verify MongoDB Atlas connection works.
Run this from the project root (with venv activated):
    python test_mongo_connection.py
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sih26106")

if not MONGO_URI:
    raise SystemExit("MONGO_URI not found in .env - check that it was added correctly.")

print("Connecting to MongoDB Atlas...")
print("Database name:", MONGO_DB_NAME)

try:
    client = MongoClient(MONGO_URI, server_api=ServerApi("1"))

    client.admin.command("ping")
    print("Connection successful. Pinged your deployment.")

    db = client[MONGO_DB_NAME]

    test_collection = db["connection_test"]
    result = test_collection.insert_one({"status": "connected", "test": True})
    print("Test write successful. Inserted document id:", result.inserted_id)

    doc = test_collection.find_one({"_id": result.inserted_id})
    print("Test read successful. Document:", doc)

    test_collection.delete_one({"_id": result.inserted_id})
    print("Test document cleaned up.")

    print("\nAll checks passed. MongoDB Atlas is ready to use.")

except Exception as e:
    print("\nConnection FAILED:", e)
    print("\nCommon causes:")
    print("- Password has special characters that need URL-encoding")
    print("- IP address not whitelisted in Network Access")
    print("- Typo in the connection string or database user name")