import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from app.api import anomaly, workload, metrics, models, training, tasks
from app.core.logging import setup_logging
from app.services.metrics_service import metrics_service
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from fastapi_cache.coder import JsonCoder
import logging
import json
from typing import Union, Any
from app.api.training import setup_shutdown_handler as setup_training_shutdown_handler

# Load environment variables from .env file
load_dotenv()

# Setup logging first
setup_logging()

# Initialize logger
logger = logging.getLogger(__name__)

# Custom JsonCoder that handles both bytes and strings
class SafeJsonCoder(JsonCoder):
    @classmethod
    def decode(cls, value: Union[str, bytes]) -> Any:
        if isinstance(value, bytes):
            return json.loads(value.decode())
        return json.loads(value)

# Initialize Redis with connection test
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost')
try:
    redis = aioredis.from_url(REDIS_URL, encoding="utf8")
    # Test connection
    # await redis.ping()
except Exception as e:
    logger.error(f"Redis connection failed: {str(e)}")
    redis = None
    FastAPICache.init(RedisBackend(None), prefix="dummy-cache")  # Prevent cache errors

# Initialize FastAPI Cache BEFORE creating the app
FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache", coder=SafeJsonCoder)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Startup
        metrics_service.start_collection()
        # Initialize FastAPI cache
        if redis is not None:
            FastAPICache.init(
                RedisBackend(redis),
                prefix="fastapi-cache",
                key_builder=None,
                coder=SafeJsonCoder
            )
            logger.info("FastAPI cache initialized with Redis backend")
        else:
            logger.warning("FastAPI cache not initialized - Redis unavailable")
        yield
    finally:
        # Shutdown - ensure metrics collection is always stopped
        metrics_service.stop_collection()

app = FastAPI(    
    title="DBPecker API",
    description="API for distributed database anomaly detection and diagnosis",
    version="1.0.0",
    lifespan=lifespan
)

# Read allowed origins from environment variable, default to localhost
allowed_origins_str = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:8001,http://127.0.0.1:8001')
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

# Ensure at least localhost is allowed if the env var is empty or misconfigured
if not allowed_origins:
    allowed_origins = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:8001", 
        "http://127.0.0.1:8001",
        "http://10.101.168.212:3000"
    ]
# Also add the origin if the environment variable is set but doesn't contain it
elif "http://10.101.168.212:3000" not in allowed_origins:
    allowed_origins.append("http://10.101.168.212:3000")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, # Use the list from environment variable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["anomaly"])
app.include_router(workload.router, prefix="/api/workload", tags=["workload"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])

# Set up training service shutdown handler
setup_training_shutdown_handler(app)

# Create metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "DBPecker API is running"}

@app.on_event("shutdown")
async def shutdown_event():
    metrics_service.stop_collection()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', '8001'))
    uvicorn.run(app, host="0.0.0.0", port=port) 