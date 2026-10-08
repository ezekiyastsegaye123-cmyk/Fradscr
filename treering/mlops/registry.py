"""
FRADSCR MLOps Pipeline — Model Registry & Lineage Tracker
=========================================================
Tracks model provenance, SHA-256 artifact checksums, training hyperparameters,
validation metrics, and promotion lifecycle (candidate -> champion -> archived).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger("fradscr.mlops.registry")


@dataclass
class RegisteredModelEntry:
    """Metadata entry for a single versioned model in the registry."""
    model_id: str
    version: str
    artifact_path: str
    artifact_sha256: str
    created_at: str
    lifecycle_status: Literal["candidate", "champion", "archived", "challenger"]
    model_type: str
    metrics: Dict[str, float]
    gating_decision: Literal["DEPLOY_APPROVED", "DEPLOY_REJECTED", "PENDING"]
    training_data_sources: List[str]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    description: Optional[str] = None


class ModelRegistry:
    """Manages versioned model artifacts and deployment lifecycle."""

    def __init__(self, registry_file: Optional[Path] = None) -> None:
        self.registry_file = registry_file or Path("models/model_registry.json")
        self._entries: Dict[str, RegisteredModelEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, item in data.get("models", {}).items():
                        self._entries[k] = RegisteredModelEntry(**item)
            except Exception as exc:
                logger.warning("Failed to parse registry file %s: %s", self.registry_file, exc)

    def save(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "models": {k: asdict(v) for k, v in self._entries.items()},
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Calculate SHA-256 cryptographic digest of a model artifact."""
        if not filepath.exists():
            return "FILE_NOT_FOUND"
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def register_model(
        self,
        model_id: str,
        version: str,
        artifact_path: Path,
        model_type: str,
        metrics: Dict[str, float],
        gating_decision: str,
        training_data_sources: List[str],
        hyperparameters: Dict[str, Any],
        feature_names: List[str],
        description: Optional[str] = None,
        promote_to_champion: bool = False,
    ) -> RegisteredModelEntry:
        """Register a new candidate model in the registry."""
        sha = self.compute_sha256(artifact_path)
        status = "champion" if promote_to_champion else "candidate"

        # If promoting to champion, archive previous champion of same type
        if promote_to_champion:
            for k, entry in self._entries.items():
                if entry.model_type == model_type and entry.lifecycle_status == "champion":
                    self._entries[k] = RegisteredModelEntry(
                        **{**asdict(entry), "lifecycle_status": "archived"}
                    )

        entry = RegisteredModelEntry(
            model_id=model_id,
            version=version,
            artifact_path=str(artifact_path),
            artifact_sha256=sha,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            lifecycle_status=status,
            model_type=model_type,
            metrics=metrics,
            gating_decision=gating_decision,
            training_data_sources=training_data_sources,
            hyperparameters=hyperparameters,
            feature_names=feature_names,
            description=description,
        )

        self._entries[model_id] = entry
        self.save()
        logger.info("Successfully registered model %s (%s) with status '%s'", model_id, version, status)
        return entry

    def get_champion(self, model_type: Optional[str] = None) -> Optional[RegisteredModelEntry]:
        """Retrieve the currently designated production champion model."""
        for entry in reversed(list(self._entries.values())):
            if entry.lifecycle_status == "champion":
                if model_type is None or entry.model_type == model_type:
                    return entry
        return None

    def list_models(self) -> List[RegisteredModelEntry]:
        """Return all registered model entries."""
        return list(self._entries.values())
