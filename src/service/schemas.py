"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.constants import PRODUCT_COLUMNS


class ClientProfile(BaseModel):
    ind_empleado: str | None = "N"
    pais_residencia: str | None = "ES"
    sexo: str | None = "UNKNOWN"
    age: float | None = None
    fecha_alta: date | None = None
    ind_nuevo: float | None = 0
    antiguedad: float | None = None
    indrel: float | None = 1
    indrel_1mes: str | None = "1"
    tiprel_1mes: str | None = "A"
    indresi: str | None = "S"
    indext: str | None = "N"
    conyuemp: str | None = "UNKNOWN"
    canal_entrada: str | None = "KHE"
    indfall: str | None = "N"
    tipodom: float | None = 1
    cod_prov: float | None = None
    nomprov: str | None = "UNKNOWN"
    ind_actividad_cliente: float | None = 0
    renta: float | None = None
    segmento: str | None = "02 - PARTICULARES"
    prev_products_total: float | None = None
    products_added_prev_month: float | None = 0
    products_dropped_prev_month: float | None = 0
    has_any_new_product: float | None = 0


class PredictRequest(BaseModel):
    customer_id: int | None = None
    snapshot_date: date = Field(..., description="Current client snapshot date used for recommendation.")
    current_products: list[str] = Field(default_factory=list, description="Product codes already owned by the client.")
    top_k: int = Field(default=3, ge=1, le=10)
    profile: ClientProfile = Field(default_factory=ClientProfile)

    @field_validator("current_products")
    @classmethod
    def validate_products(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - set(PRODUCT_COLUMNS))
        if invalid:
            raise ValueError(f"Unknown product codes: {invalid}")
        return value


class RecommendationItem(BaseModel):
    product_code: str
    product_name: str
    score: float


class PredictResponse(BaseModel):
    customer_id: int | None
    snapshot_date: date
    top_k: int
    recommendations: list[RecommendationItem]
    suspicious_request: bool
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
    model_path: str
