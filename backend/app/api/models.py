from fastapi import APIRouter, HTTPException
from app.services.diagnosis_service import DiagnosisService
from fastapi.responses import JSONResponse
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()
diagnosis_service = DiagnosisService()

class CompareModelsRequest(BaseModel):
    model_names: List[str] 

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