"""Model loading and inference helpers for the API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.constants import PRODUCT_COLUMNS
from src.models.predict import recommend_from_scores, score_dataframe
from src.service.monitoring import detect_suspicious_request
from src.service.schemas import PredictRequest


@dataclass(slots=True)
class InferenceResult:
    recommendations: list[dict[str, Any]]
    suspicious_request: bool
    warnings: list[str]


class RecommendationInferenceService:
    """Thin wrapper around the trained model bundle."""

    def __init__(self, model_path: Path, *, preload_product_models: bool = True) -> None:
        self.model_path = model_path
        self.preload_product_models = preload_product_models
        self.bundle: dict[str, Any] | None = None

    def load(self) -> None:
        self.bundle = joblib.load(self.model_path)
        model = self.bundle.get("model")
        if self.preload_product_models and hasattr(model, "enable_inference_cache"):
            model.enable_inference_cache(preload=True)

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    def build_feature_row(self, request: PredictRequest) -> dict[str, Any]:
        if self.bundle is None:
            raise RuntimeError("Model bundle is not loaded")

        profile = request.profile.model_dump()
        feature_row: dict[str, Any] = {}
        current_products = set(request.current_products)

        for column in PRODUCT_COLUMNS:
            feature_row[column] = int(column in current_products)

        for column in self.bundle["feature_columns"]:
            if column in PRODUCT_COLUMNS:
                feature_row[column] = int(column in current_products)
            else:
                feature_row[column] = 0

        feature_row.update(
            {
                "month_number": request.snapshot_date.month,
                "customer_since_months": profile["antiguedad"] or 0,
                "products_total": len(current_products),
                "prev_products_total": profile["prev_products_total"] if profile["prev_products_total"] is not None else len(current_products),
                "products_added_prev_month": profile["products_added_prev_month"] or 0,
                "products_dropped_prev_month": profile["products_dropped_prev_month"] or 0,
                "has_any_new_product": profile["has_any_new_product"] or 0,
                "age": profile["age"],
                "ind_nuevo": profile["ind_nuevo"],
                "antiguedad": profile["antiguedad"],
                "indrel": profile["indrel"],
                "tipodom": profile["tipodom"],
                "cod_prov": profile["cod_prov"],
                "ind_actividad_cliente": profile["ind_actividad_cliente"],
                "renta": profile["renta"],
                "ind_empleado": profile["ind_empleado"],
                "pais_residencia": profile["pais_residencia"],
                "sexo": profile["sexo"],
                "indrel_1mes": profile["indrel_1mes"],
                "tiprel_1mes": profile["tiprel_1mes"],
                "indresi": profile["indresi"],
                "indext": profile["indext"],
                "conyuemp": profile["conyuemp"],
                "canal_entrada": profile["canal_entrada"],
                "indfall": profile["indfall"],
                "nomprov": profile["nomprov"],
                "segmento": profile["segmento"],
            }
        )

        if request.profile.fecha_alta and request.profile.antiguedad is None:
            delta_days = (request.snapshot_date - request.profile.fecha_alta).days
            feature_row["customer_since_months"] = max(delta_days / 30.0, 0.0)

        feature_row["has_any_new_product"] = int((feature_row["products_added_prev_month"] or 0) > 0)
        return feature_row

    def predict(self, request: PredictRequest) -> InferenceResult:
        if self.bundle is None:
            raise RuntimeError("Model bundle is not loaded")

        feature_row = self.build_feature_row(request)
        warnings = detect_suspicious_request(feature_row, self.bundle.get("reference_stats", {}))

        frame = pd.DataFrame([feature_row])
        scores = score_dataframe(self.bundle, frame)
        recommendations = recommend_from_scores(
            scores=scores,
            current_products=frame[PRODUCT_COLUMNS].to_numpy(),
            k=request.top_k,
        )[0]

        return InferenceResult(
            recommendations=recommendations,
            suspicious_request=bool(warnings) or not recommendations,
            warnings=warnings,
        )
