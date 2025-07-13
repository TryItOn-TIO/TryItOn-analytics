import redis
import json
import os

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")

r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

def get_recommendation_from_redis(key: str):
    value = r.get(key)
    if value:
        return json.loads(value)
    return []

def set_recommendation_to_redis(key: str, value):
    r.set(key, json.dumps(value)) 