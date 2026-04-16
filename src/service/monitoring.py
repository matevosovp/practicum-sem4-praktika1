"""Prometheus metrics and drift helpers for the API."""

from __future__ import annotations

from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


REQUEST_COUNT = Counter("bank_reco_request_total", "Total prediction requests", ["endpoint"])
REQUEST_ERRORS = Counter("bank_reco_error_total", "Total API errors", ["endpoint"])
REQUEST_LATENCY = Histogram(
    "bank_reco_latency_seconds",
    "Latency of API requests in seconds",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0),
)
RECOMMENDATION_SCORE = Histogram(
    "bank_reco_score_distribution",
    "Distribution of emitted recommendation scores",
    buckets=(0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
)
EMPTY_RECOMMENDATIONS = Counter("bank_reco_empty_total", "Prediction requests with no returned recommendations")
SUSPICIOUS_RECOMMENDATIONS = Counter("bank_reco_suspicious_total", "Requests flagged as suspicious or drifted")


def prometheus_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def detect_suspicious_request(feature_row: dict[str, Any], reference_stats: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    for feature_name, stats in reference_stats.get("numeric", {}).items():
        value = feature_row.get(feature_name)
        if value is None:
            continue
        if value < stats["p01"] or value > stats["p99"]:
            warnings.append(f"{feature_name} outside training 1%-99% interval")

    for feature_name, top_values in reference_stats.get("categorical", {}).items():
        value = str(feature_row.get(feature_name, "UNKNOWN"))
        if value not in top_values:
            warnings.append(f"{feature_name} has unseen or rare category")

    return warnings
