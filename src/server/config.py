"""Application settings, read from environment variables (or passed in for tests)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    data_dir: Path = Field(Path("data"), alias="SCANNER_DATA",
                           description="Uploads and the fingerprint index")
    examples_dir: Path = Field(REPO_ROOT / "samples", alias="SCANNER_EXAMPLES")
    max_upload_bytes: int = Field(15 * 1024 * 1024, alias="SCANNER_MAX_UPLOAD_BYTES", gt=0)
    max_scans: int = Field(200, alias="SCANNER_MAX_SCANS", gt=0,
                           description="Scans kept in memory; older uploads are deleted")

    trufor_dir: Path | None = Field(None, alias="TRUFOR_DIR")
    trufor_weights: Path | None = Field(None, alias="TRUFOR_WEIGHTS")
    dtd_dir: Path | None = Field(None, alias="DTD_DIR")
    dtd_weights: Path | None = Field(None, alias="DTD_WEIGHTS")
