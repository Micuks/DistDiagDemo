from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

app = FastAPI(
    title="DistDiagDemo API",
    description="API for distributed database anomaly detection and diagnosis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Import and include routers
from app.api import anomaly_detection, graph_analysis, metrics

app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])
app.include_router(anomaly_detection.router, prefix="/api/v1/anomalies", tags=["anomalies"])
app.include_router(graph_analysis.router, prefix="/api/v1/graph", tags=["graph"])

# Create metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 