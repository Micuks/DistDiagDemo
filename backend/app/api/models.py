from fastapi import APIRouter, HTTPException
from app.services.diagnosis_service import DiagnosisService
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
diagnosis_service = DiagnosisService()

@router.get("/list")
async def list_models():
    """Get list of available models"""
    try:
        models = diagnosis_service.get_available_models()
        return JSONResponse(
            content=models,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )

@router.get("/{model_name}/performance")
async def get_model_performance(model_name: str):
    """Get performance metrics for a specific model"""
    try:
        metrics = diagnosis_service.get_model_performance(model_name)
        if not metrics:
            raise HTTPException(status_code=404, detail="Model metrics not found")
        return metrics
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 