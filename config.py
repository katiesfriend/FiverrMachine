"""Centralized configuration for FiverrMachine pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class PipelineConfig:
    """Lightweight configuration container.

    Values can be overridden via environment variables so the pipeline stays
    predictable between runs while still being tunable per deployment.
    """

    max_jobs_total: int = 50
    max_jobs_per_site: int = 15
    min_score_for_high_fit: float = 70.0
    focus_shortlist_size: int = 8


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def load_config() -> PipelineConfig:
    """Load pipeline configuration from environment or defaults."""

    return PipelineConfig(
        max_jobs_total=_int_env("FIVERR_MAX_JOBS_TOTAL", PipelineConfig.max_jobs_total),
        max_jobs_per_site=_int_env(
            "FIVERR_MAX_JOBS_PER_SITE", PipelineConfig.max_jobs_per_site
        ),
        min_score_for_high_fit=_float_env(
            "FIVERR_MIN_SCORE_HIGH_FIT", PipelineConfig.min_score_for_high_fit
        ),
        focus_shortlist_size=_int_env(
            "FIVERR_FOCUS_SHORTLIST_SIZE", PipelineConfig.focus_shortlist_size
        ),
    )


# Default instance for convenience
DEFAULT_CONFIG = load_config()
