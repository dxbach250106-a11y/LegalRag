from __future__ import annotations

import logging
from pathlib import Path

from core.config import AppConfig


class ModelDownloader:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_embedding_model(self) -> Path:
        section = self.config.section("embedding")
        return self._download(section["model_name"], self.config.root / section["local_model_dir"])

    def download_generation_model(self) -> Path | None:
        section = self.config.section("generation")
        if not section.get("enabled", False):
            return None
        return self._download(section["model_name"], self.config.root / section["local_model_dir"])

    def _download(self, model_name: str, target: Path) -> Path:
        from huggingface_hub import snapshot_download
        target.mkdir(parents=True, exist_ok=True)
        self.logger.info("Downloading %s to %s", model_name, target)
        snapshot_download(repo_id=model_name, local_dir=target, local_dir_use_symlinks=False)
        return target
