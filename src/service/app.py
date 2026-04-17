"""FastAPI application serving bank product recommendations."""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Response

from src.service.inference import RecommendationInferenceService
from src.service.monitoring import (
    EMPTY_RECOMMENDATIONS,
    RECOMMENDATION_SCORE,
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
    SUSPICIOUS_RECOMMENDATIONS,
    prometheus_payload,
)
from src.service.schemas import HealthResponse, PredictRequest, PredictResponse
from src.utils.config import ProjectConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
config = ProjectConfig()
app = FastAPI(
    title="Bank Product Recommendation Service",
    description="REST API that ranks new banking products for an existing client profile.",
    version="1.0.0",
)
inference_service = RecommendationInferenceService(
    config.api_model_path,
    preload_product_models=config.api_preload_product_models,
)


@app.on_event("startup")
def load_model() -> None:
    inference_service.load()
    logger.info("Loaded model from %s", config.api_model_path)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=inference_service.loaded, model_path=str(config.api_model_path))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    try:
        result = inference_service.predict(request)
    except Exception as exc:  # pragma: no cover - defensive handler for service runtime
        REQUEST_ERRORS.labels(endpoint="/predict").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - start)

    if not result.recommendations:
        EMPTY_RECOMMENDATIONS.inc()

    if result.suspicious_request:
        SUSPICIOUS_RECOMMENDATIONS.inc()

    for item in result.recommendations:
        RECOMMENDATION_SCORE.observe(item["score"])

    return PredictResponse(
        customer_id=request.customer_id,
        snapshot_date=request.snapshot_date,
        top_k=request.top_k,
        recommendations=result.recommendations,
        suspicious_request=result.suspicious_request,
        warnings=result.warnings,
    )


@app.get("/metrics")
def metrics() -> Response:
    REQUEST_COUNT.labels(endpoint="/metrics").inc()
    payload, content_type = prometheus_payload()
    return Response(content=payload, media_type=content_type)
