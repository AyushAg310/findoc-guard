"""Runtime configuration.

Values come from (in order of precedence): environment variables prefixed with
``FINDOC_``, a local ``.env`` file, then the defaults below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINDOC_", env_file=".env", extra="ignore")

    # General
    log_level: str = "INFO"
    output_dir: Path = Path("outputs")
    device: str = Field(default="auto", description="'auto', 'cpu' or 'cuda'")

    # Capture / quality (Phase 1)
    max_side_px: int = 2000
    blur_threshold: float = 100.0  # variance of Laplacian below this = blurry
    min_coverage: float = 0.25  # document area / frame area (cheques are long and thin)

    # Learned page detection (Phase 4): "auto" uses the model if the file exists, "off" never
    segmentation: str = "auto"
    segmenter_model: Path = Path("models/docseg.onnx")

    # Cheques (Phase 2)
    templates_dir: Path | None = None  # folder of custom templates; None = built-in banks
    feature_method: str = "sift"  # "sift" or "orb"

    # Video (Phase 3)
    video_max_frames: int = 180  # frames kept for tracking (evenly subsampled)
    video_track_side: int = 960  # longest side of frames used for tracking
    keyframe_min_view_change: float = 0.008  # projective shape change between keyframes
    keyframe_max: int = 25

    # Risk score bands (Phase 5)
    risk_medium: int = 30
    risk_high: int = 70


@lru_cache
def get_settings() -> Settings:
    return Settings()
