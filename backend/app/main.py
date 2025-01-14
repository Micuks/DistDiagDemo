import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from app.api import anomaly, workload, metrics
from app.core.logging import setup_logging

# Load environment variables from .env file
load_dotenv()

# Setup logging first
setup_logging()

app = FastAPI(    
    title="DistDiagDemo API",
    description="API for distributed database anomaly detection and diagnosis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://10.101.168.97:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(anomaly.router, prefix="/api/anomaly", tags=["anomaly"])
app.include_router(workload.router, prefix="/api/workload", tags=["workload"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])

# Create metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', '8000'))
    uvicorn.run(app, host="0.0.0.0", port=port) 