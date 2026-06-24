"""Database connections for MongoDB and Redis."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from redis import Redis

from src.config.settings import settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client with connection management."""

    def __init__(self) -> None:
        """Initialize MongoDB client."""
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.sync_client: Optional[MongoClient] = None

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.client[settings.mongodb_db_name]
            
            # Test connection
            await self.client.admin.command("ping")
            logger.info(f"Connected to MongoDB: {settings.mongodb_db_name}")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def _create_indexes(self) -> None:
        """Create necessary indexes for the requests collection."""
        try:
            # Index for date-based queries (duplicate detection lookback)
            await self.db.requests.create_index([("created_at", -1)])
            
            # Index for config version queries
            await self.db.requests.create_index([("config_version", 1)])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    def get_sync_client(self) -> MongoClient:
        """
        Get synchronous MongoDB client for vector search operations.
        
        Motor (async) doesn't support all MongoDB operations yet,
        so we use sync client for vector search.
        """
        if not self.sync_client:
            self.sync_client = MongoClient(settings.mongodb_url)
        return self.sync_client


class RedisClient:
    """Redis client for LangGraph checkpointing."""

    def __init__(self) -> None:
        """Initialize Redis client."""
        self.client: Optional[Redis] = None

    def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.client = Redis.from_url(settings.redis_url, decode_responses=True)
            
            # Test connection
            self.client.ping()
            logger.info("Connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")

    def get_client(self) -> Redis:
        """Get Redis client instance."""
        if not self.client:
            self.connect()
        return self.client


# Note: Instances should be created in main.py lifespan, not here
# This keeps initialization explicit and visible


async def save_request(
    mongodb_client: MongoDBClient,
    data: Dict[str, Any],
    embedding: List[float]
) -> str:
    """
    Save request to MongoDB with embedding.
    
    Args:
        mongodb_client: MongoDB client instance
        data: Request data dictionary
        embedding: Vector embedding for duplicate detection
        
    Returns:
        Inserted document ID as string
    """
    from src.config.settings import prompt_config
    
    document = {
        "config_version": prompt_config.config_version,
        "data": data,
        "embedding": embedding,
        "created_at": datetime.utcnow(),
    }
    
    result = await mongodb_client.db.requests.insert_one(document)
    logger.info(f"Saved request with ID: {result.inserted_id}")
    
    return str(result.inserted_id)


async def update_request(
    mongodb_client: MongoDBClient,
    request_id: str,
    data: Dict[str, Any],
    embedding: List[float]
) -> bool:
    """
    Update an existing request in MongoDB.
    
    Args:
        request_id: MongoDB document ID to update
        data: Updated request data dictionary
        embedding: Updated vector embedding
        
    Returns:
        True if update successful, False otherwise
    """
    from bson import ObjectId
    from src.config.settings import prompt_config
    
    try:
        result = await mongodb_client.db.requests.update_one(
            {"_id": ObjectId(request_id)},
            {
                "$set": {
                    "data": data,
                    "embedding": embedding,
                    "config_version": prompt_config.config_version,
                    "updated_at": datetime.utcnow(),
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated request with ID: {request_id}")
            return True
        else:
            logger.warning(f"No request found with ID: {request_id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update request: {e}")
        return False

async def find_exact_match_duplicates(
    mongodb_client: MongoDBClient,
    data: Dict[str, Any],
    lookback_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Find exact match duplicates based on non-PII fields.
    
    This is the first line of defense for duplicate detection.
    Checks if there's an exact match on all non-PII fields within the lookback period.
    
    Args:
        mongodb_client: MongoDB client instance
        data: Request data dictionary (without PII)
        lookback_days: Number of days to look back
        
    Returns:
        List of exact match duplicates
    """
    from src.config.settings import prompt_config
    
    # Get PII fields to exclude from matching
    pii_fields = set(prompt_config.privacy.get("pii_fields", []))
    
    # Build query for exact match on non-PII fields
    query = {}
    for key, value in data.items():
        if key not in pii_fields:
            query[f"data.{key}"] = value
    
    # Add date filter
    cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
    query["created_at"] = {"$gte": cutoff_date}
    
    # Find exact matches
    cursor = mongodb_client.db.requests.find(query).sort("created_at", -1).limit(5)
    
    exact_matches = []
    async for doc in cursor:
        # Add a flag to indicate this is an exact match
        doc["is_exact_match"] = True
        doc["similarity_score"] = 1.0  # Perfect match
        exact_matches.append(doc)
    
    if exact_matches:
        logger.info(f"Found {len(exact_matches)} exact match duplicate(s)")
    else:
        logger.info("No exact match duplicates found")
    
    return exact_matches



async def find_similar_requests(
    mongodb_client: MongoDBClient,
    embedding: List[float],
    threshold: float = 0.85,
    lookback_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Find similar requests using vector search or fallback method.
    
    Args:
        mongodb_client: MongoDB client instance
        embedding: Query embedding vector
        threshold: Similarity threshold (0-1)
        lookback_days: Number of days to look back
        
    Returns:
        List of similar requests with similarity scores
    """
    if settings.vector_search_provider == "atlas":
        try:
            return await _atlas_vector_search(mongodb_client, embedding, threshold, lookback_days)
        except Exception as e:
            logger.warning(f"Atlas vector search failed: {e}, falling back to local")
            if settings.duplicate_detection_enabled:
                return await _local_similarity_search(mongodb_client, embedding, threshold, lookback_days)
            return []
    else:
        return await _local_similarity_search(mongodb_client, embedding, threshold, lookback_days)


async def _atlas_vector_search(
    mongodb_client: MongoDBClient,
    embedding: List[float],
    threshold: float,
    lookback_days: int
) -> List[Dict[str, Any]]:
    """
    Use MongoDB Atlas vector search for similarity.
    
    Requires vector search index to be created in Atlas.
    """
    cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": 5,
            }
        },
        {
            "$match": {
                "created_at": {"$gte": cutoff_date}
            }
        },
        {
            "$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {
                "similarity_score": {"$gte": threshold}
            }
        }
    ]
    
    results = []
    async for doc in mongodb_client.db.requests.aggregate(pipeline):
        results.append(doc)
    
    logger.info(f"Atlas vector search found {len(results)} similar requests")
    return results


async def _local_similarity_search(
    mongodb_client: MongoDBClient,
    embedding: List[float],
    threshold: float,
    lookback_days: int
) -> List[Dict[str, Any]]:
    """
    Fallback: Manual cosine similarity calculation.
    
    Used when Atlas vector search is not available.
    Limited to recent requests to avoid memory issues.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
    
    # Fetch recent requests
    cursor = mongodb_client.db.requests.find(
        {"created_at": {"$gte": cutoff_date}},
        limit=100
    ).sort("created_at", -1)
    
    similar_requests = []
    query_embedding = np.array(embedding).reshape(1, -1)
    
    async for doc in cursor:
        if "embedding" in doc:
            doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            
            if similarity >= threshold:
                doc["similarity_score"] = float(similarity)
                similar_requests.append(doc)
    
    # Sort by similarity score
    similar_requests.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    logger.info(f"Local similarity search found {len(similar_requests)} similar requests")
    return similar_requests[:5]  # Return top 5

# Made with Bob
