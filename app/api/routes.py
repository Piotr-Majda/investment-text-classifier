from fastapi import APIRouter, Depends, HTTPException, status

from app.models.prediction import TextClassificationInput, TextClassificationResult
from app.services.classifier import ClassifierService, get_classifier_service

router = APIRouter()


@router.post(
    "/classify",
    response_model=TextClassificationResult,
    status_code=status.HTTP_200_OK,
    summary="Classify investment text",
    description="Analyze investment text and provide classification based on risk level, "
    "investment horizon, and recommended action.",
)
async def classify_text(
    input_data: TextClassificationInput,
    classifier_service: ClassifierService = Depends(get_classifier_service),
) -> TextClassificationResult:
    try:
        result = classifier_service.predict(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification error: {str(e)}",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the API is running and the model is loaded.",
)
async def health_check(
    classifier_service: ClassifierService = Depends(get_classifier_service),
) -> dict:
    if classifier_service.is_ready():
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False} 