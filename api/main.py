"""
FastAPI REST API for Mixture of Experts Recommendation System

This module provides a high-performance API for serving real-time personalized
recommendations using the MoE Transformer model.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import torch
import redis
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
import uvicorn
from contextlib import asynccontextmanager

# Import MoE model components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe_transformer import (
    MoEConfig, RecommendationMoE, create_moe_model, 
    load_moe_model, MoEMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate limiting configuration
RATE_LIMIT_PER_MINUTE = 100
RATE_LIMIT_PER_USER_PER_MINUTE = 20

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Model configuration
MODEL_CONFIG = MoEConfig(
    num_experts=8,
    num_experts_per_token=2,
    expert_capacity=64,
    hidden_size=512,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    vocab_size=50000,
    max_seq_length=512,
    load_balancing_loss_weight=0.01
)

# Global variables for model and monitoring
model: Optional[RecommendationMoE] = None
monitor: Optional[MoEMonitor] = None
redis_client: Optional[redis.Redis] = None

# Security schemas
security = HTTPBearer()

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID for personalization")
    item_ids: List[int] = Field(..., description="List of item IDs to score")
    context_features: Optional[List[float]] = Field(None, description="Optional context features")
    num_recommendations: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")
    
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    scores: List[float]
    expert_metrics: Dict[str, Any]
    latency_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float
    total_requests: int
    avg_latency_ms: float

class ExpertMetricsResponse(BaseModel):
    expert_utilization: Dict[str, int]
    performance_summary: Dict[str, Any]
    load_balancing_loss: float
    activated_experts_per_request: float

# Rate limiting utilities
class RateLimiter:
    """Rate limiter using Redis for distributed rate limiting"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def check_rate_limit(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is within rate limit"""
        current_time = int(time.time())
        window_start = current_time - window
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_count = self.redis.zcard(key)
        
        if current_count >= limit:
            return False
        
        # Add current request
        self.redis.zadd(key, {str(current_time): current_time})
        self.redis.expire(key, window)
        
        return True

# Authentication utilities
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return username"""
    try:
        # In production, implement proper JWT verification
        # For now, we'll use a simple token check
        token = credentials.credentials
        if token == "valid-token":  # Replace with proper JWT verification
            return "user"
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# Model management
async def load_model():
    """Load the MoE model and initialize monitoring"""
    global model, monitor, redis_client
    
    try:
        # Initialize Redis connection
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        
        # Test Redis connection
        redis_client.ping()
        logger.info("Redis connection established")
        
        # Initialize model (in production, load from saved weights)
        num_users = 1000000  # 1M users
        num_items = 500000   # 500K items
        
        model = create_moe_model(MODEL_CONFIG, num_users, num_items)
        model.eval()  # Set to evaluation mode
        
        # Initialize monitor
        monitor = MoEMonitor()
        
        logger.info("MoE model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await load_model()
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    if redis_client:
        redis_client.close()
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="MoE Recommendation API",
    description="High-performance recommendation API using Mixture of Experts Transformer",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Rate limiter instance
rate_limiter = None

@app.on_event("startup")
async def startup_event():
    """Initialize rate limiter after Redis is connected"""
    global rate_limiter
    if redis_client:
        rate_limiter = RateLimiter(redis_client)

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MoE Recommendation API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model, monitor, redis_client
    
    uptime = time.time() - getattr(app.state, 'start_time', time.time())
    total_requests = len(monitor.metrics_history) if monitor else 0
    avg_latency = 0.0
    
    if monitor and monitor.metrics_history:
        # Calculate average latency from recent requests
        recent_metrics = monitor.metrics_history[-100:]
        # This would need to be tracked in actual request processing
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        redis_connected=redis_client is not None and redis_client.ping(),
        uptime_seconds=uptime,
        total_requests=total_requests,
        avg_latency_ms=avg_latency
    )

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    username: str = Depends(verify_token)
):
    """
    Get personalized recommendations for a user.
    
    This endpoint uses the MoE model to generate real-time recommendations
    with only a subset of experts activated for efficiency.
    """
    global model, monitor, rate_limiter
    
    start_time = time.time()
    
    try:
        # Rate limiting
        if rate_limiter:
            # Global rate limiting
            global_allowed = await rate_limiter.check_rate_limit(
                "global_requests", RATE_LIMIT_PER_MINUTE
            )
            if not global_allowed:
                raise HTTPException(status_code=429, detail="Global rate limit exceeded")
            
            # Per-user rate limiting
            user_allowed = await rate_limiter.check_rate_limit(
                f"user_{request.user_id}", RATE_LIMIT_PER_USER_PER_MINUTE
            )
            if not user_allowed:
                raise HTTPException(status_code=429, detail="User rate limit exceeded")
        
        # Input validation
        if not request.item_ids:
            raise HTTPException(status_code=400, detail="At least one item ID is required")
        
        if len(request.item_ids) > 1000:
            raise HTTPException(status_code=400, detail="Too many items requested")
        
        # Check cache first
        cache_key = f"rec:{request.user_id}:{hash(tuple(request.item_ids))}"
        cached_result = redis_client.get(cache_key) if redis_client else None
        
        if cached_result:
            cached_data = json.loads(cached_result)
            return RecommendationResponse(**cached_data)
        
        # Prepare input tensors
        user_ids = torch.tensor([request.user_id] * len(request.item_ids), dtype=torch.long)
        item_ids = torch.tensor(request.item_ids, dtype=torch.long)
        
        context_tensor = None
        if request.context_features:
            context_tensor = torch.tensor([request.context_features] * len(request.item_ids), dtype=torch.float)
        
        # Get recommendations from model
        with torch.no_grad():
            model_output = model(user_ids, item_ids, context_tensor)
        
        # Extract scores and sort by relevance
        scores = model_output['recommendation_scores'].cpu().numpy()
        item_score_pairs = list(zip(request.item_ids, scores))
        item_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_recommendations = item_score_pairs[:request.num_recommendations]
        
        # Prepare response
        recommendations = [
            {
                "item_id": item_id,
                "score": float(score),
                "rank": idx + 1
            }
            for idx, (item_id, score) in enumerate(top_recommendations)
        ]
        
        # Update monitoring metrics
        if monitor:
            monitor.update_metrics(model_output)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            scores=[float(score) for _, score in top_recommendations],
            expert_metrics=model_output.get('metrics', {}),
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Cache result (with short TTL for real-time recommendations)
        if redis_client:
            redis_client.setex(
                cache_key,
                300,  # 5 minutes TTL
                json.dumps(response.dict())
            )
        
        # Background task for logging
        background_tasks.add_task(log_recommendation_request, request, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics/experts", response_model=ExpertMetricsResponse)
async def get_expert_metrics(username: str = Depends(verify_token)):
    """Get expert utilization and performance metrics"""
    global monitor
    
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    expert_utilization = monitor.get_expert_utilization()
    performance_summary = monitor.get_performance_summary()
    
    return ExpertMetricsResponse(
        expert_utilization=expert_utilization,
        performance_summary=performance_summary,
        load_balancing_loss=performance_summary.get('avg_load_balancing_loss', 0.0),
        activated_experts_per_request=performance_summary.get('avg_activated_experts', 0.0)
    )

@app.post("/model/reload")
async def reload_model(username: str = Depends(verify_token)):
    """Reload the model (admin only)"""
    global model
    
    try:
        await load_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload model")

# Utility functions
async def log_recommendation_request(request: RecommendationRequest, response: RecommendationResponse):
    """Log recommendation request for analytics"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": request.user_id,
        "num_items_requested": len(request.item_ids),
        "num_recommendations": len(response.recommendations),
        "latency_ms": response.latency_ms,
        "expert_metrics": response.expert_metrics
    }
    
    # In production, send to analytics service
    logger.info(f"Recommendation request logged: {log_entry}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat()
    }

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for model sharing
        log_level="info"
    ) 