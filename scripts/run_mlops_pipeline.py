#!/usr/bin/env python3
"""
FRADSCR MLOps Pipeline Runner (CLI)
===================================
Orchestrates end-to-end data validation, feature engineering, model training,
operational gating, model registry, and canary verification.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from treering.mlops import MLOpsPipelineOrchestrator, ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fradscr.mlops.cli")


def main():
    parser = argparse.ArgumentParser(description="FRADSCR MLOps End-to-End Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Execute complete end-to-end MLOps pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "data_validation",
            "feature_pipeline",
            "model_training",
            "model_gating",
            "model_registration",
            "canary_verification",
        ],
        help="Execute single isolated pipeline stage",
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate pipeline stages without modifying registry")
    parser.add_argument("--list-models", action="store_true", help="Display all registered models in the registry")
    parser.add_argument("--champion", action="store_true", help="Show currently active production champion model")

    args = parser.parse_args()

    orchestrator = MLOpsPipelineOrchestrator(base_dir=PROJECT_ROOT)

    if args.list_models:
        registry = orchestrator.registry
        models = registry.list_models()
        print(f"\n{'='*70}\nFRADSCR MODEL REGISTRY ({len(models)} models registered)\n{'='*70}")
        for m in models:
            champ = " [CURRENT CHAMPION]" if m.lifecycle_status == "champion" else ""
            print(f"• ID: {m.model_id} (Version: {m.version}){champ}")
            print(f"  Type: {m.model_type} | Status: {m.lifecycle_status} | Gating: {m.gating_decision}")
            print(f"  Artifact: {m.artifact_path} | SHA-256: {m.artifact_sha256[:16]}...")
            print(f"  Key Metrics: {m.metrics}\n")
        return

    if args.champion:
        champ = orchestrator.registry.get_champion()
        if champ:
            print(f"\nActive Production Champion: {champ.model_id} (v{champ.version})")
            print(json.dumps(champ.__dict__, indent=2))
        else:
            print("No active production champion found in registry.")
        return

    # Execute pipeline
    stages = None
    if args.stage:
        stages = [args.stage]
    elif not args.all:
        stages = orchestrator.ORDERED_STAGES

    print(f"\n{'='*70}\nLAUNCHING FRADSCR MLOPS PIPELINE\n{'='*70}")
    print(f"Target Stages: {stages or 'FULL PIPELINE'}")
    print(f"Dry Run: {args.dry_run}\n")

    record = orchestrator.run_pipeline(stages=stages, dry_run=args.dry_run)

    print(f"\n{'='*70}\nPIPELINE RUN SUMMARY: {record.overall_status}\n{'='*70}")
    print(f"Run ID: {record.run_id}")
    print(f"Started: {record.started_at} | Completed: {record.completed_at}")
    print(f"Total Duration: {record.total_duration_seconds:.2f}s\n")

    for s in record.stages_executed:
        icon = "✅" if s.status == "SUCCESS" else "❌"
        print(f"{icon} Stage: {s.stage_name:<22} | Status: {s.status:<8} | Duration: {s.duration_seconds}s")
        if s.error_message:
            print(f"   Error: {s.error_message}")

    if record.model_registered:
        print(f"\n📦 Registered Champion Model: {record.model_registered}")
    if record.deployment_decision:
        print(f"🎯 Deployment Gate Decision: {record.deployment_decision}")

    if record.overall_status != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()
