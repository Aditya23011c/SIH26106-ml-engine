"""
analytics/mongo_analytics.py

MongoDB connection singleton for the analytics layer.
Replaces the earlier Snowflake-based snowflake_client.py.

Since analytics now reads directly from the same MongoDB cluster that
the rest of the platform writes to, there is no separate sync job —
trend_queries.py runs aggregation pipelines straight against the
`cases` collection.
"""

import os
import time
from functools import wraps

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sih26106")
CASES_COLLECTION = os.getenv("CASES_COLLECTION_NAME", "cases")

_client = None
_db = None

# Simple in-memory TTL cache so repeated dashboard hits inside the same
# process don't re-run the same aggregation every request.
_query_cache = {}
DEFAULT_CACHE_TTL_SECONDS = 60


def get_client():
    """Returns a singleton MongoClient, creating it on first use."""
    global _client
    if _client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI is not set in the environment (.env).")
        _client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
    return _client


def get_db():
    """Returns the analytics database handle (same DB as the rest of the app)."""
    global _db
    if _db is None:
        _db = get_client()[MONGO_DB_NAME]
    return _db


def get_cases_collection():
    """Returns the `cases` collection that analytics queries run against."""
    return get_db()[CASES_COLLECTION]


def cached_query(ttl_seconds=DEFAULT_CACHE_TTL_SECONDS):
    """
    Decorator that caches a query function's result in memory for ttl_seconds.
    Cache key is the function name + its args/kwargs.
    Good enough for a hackathon dashboard; not meant for multi-process deployment.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()

            if cache_key in _query_cache:
                cached_result, cached_at = _query_cache[cache_key]
                if now - cached_at < ttl_seconds:
                    return cached_result

            result = func(*args, **kwargs)
            _query_cache[cache_key] = (result, now)
            return result
        return wrapper
    return decorator


def run_aggregation(pipeline):
    """
    Runs an aggregation pipeline against the cases collection and
    returns the results as a list of dicts.
    """
    collection = get_cases_collection()
    return list(collection.aggregate(pipeline))