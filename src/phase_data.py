import pandas as pd
import numpy as np
import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from core.memory import Memory
from utils.context import  Sample, Policy, RuntimeContext, Decision, PhaseResult
from utils.helper_function import estimate


class PhaseData:
    """
    1. Reads one dataset
    2. Builds a baseline data profile
    3. Applies `estimate()` to measure intrinsic data value
    4. Produces a Data Worthiness Contract
    """
    def __init__(self, dataset: str, policy: Policy, memory: Memory):
        self.dataset_path = Path(dataset)
        self.dataset_name = self.dataset_path.stem
        self.policy = policy
        self.memory = memory

        self.cache: List[np.ndarray] = []
        self.cache_size = 100

        self.stats = dict(total=0, kept=0, dropped=0, labeled=0)
        self.values: List[float] = []
        self.kept_samples: List[tuple] = []


    def reinforcement_update(self, reward: float):

        lr = 0.02  # learning rate (small, stable)

        self.policy.keep_threshold = float(np.clip(
            self.policy.keep_threshold - lr * reward,
            0.4, 0.9
        ))

        self.policy.label_threshold = float(np.clip(
            self.policy.label_threshold - lr * reward * 0.5,
            0.2, self.policy.keep_threshold - 0.05
        ))

    def build_reports(self, ctx, df, numeric, categorical, columns, schema_hash, signature):
        # Compute metrics
        total = max(1, self.stats["total"])
        keep_rate = self.stats["kept"] / total
        mean_value = float(np.mean(self.values)) if self.values else 0.0

        # Status
        is_stable = len(df) >= self.policy.min_rows
        passed = (
            is_stable and
            keep_rate >= self.policy.min_keep_rate and
            mean_value >= self.policy.min_mean_value
        )
        status = "PASS" if passed else "FAIL"

        # Generate IDs
        timestamp = datetime.utcnow()
        baseline_id = f"{ctx.goal}_{timestamp:%Y%m%d_%H%M%S}"
        contract_id = f"{ctx.goal}_{uuid.uuid4().hex[:6]}"
        now = timestamp.isoformat()

        # Profile
        profile = {
            "phase":ctx.goal,
            "baseline_id": baseline_id,
            "created_by": ctx.config["created_by"],
            "dataset": self.dataset_name,
            "schema_hash": schema_hash,
            "features": {
                "numeric": numeric,
                "categorical": categorical
            },
            "columns": columns,
            "stats": {
                "row_count": len(df),
                "stable": is_stable
            },
            "lineage": {"source": str(self.dataset_path)},
            "signature": {"algo": "sha256", "hash": signature},
            "created_at": now
        }

        # Contract
        contract = {
            "type": "DATA_WORTHINESS",
            "phase": ctx.goal,
            "status": status,
            "baseline_id": baseline_id,
            "contract_id": contract_id,
            "signals": {
                "rows": len(df),
                "keep_rate": keep_rate,
                "mean_value": mean_value,
                "distribution": self.stats
            },
            "cost": {
                "spent": ctx.cost_spent,
                "budget": self.policy.cost_budget
            },
            "next_allowed_phase": "ready" if passed else None,
            "created_at": now
        }

        # Build report
        report = {
            "summary": {
                "status": status,
                "total_samples": total,
                "kept": self.stats["kept"],
                "dropped": self.stats["dropped"],
                "labeled": self.stats["labeled"],
                "keep_rate": f"{keep_rate:.1%}",
                "mean_value": round(mean_value, 4)
            },
            "dataset_profile": {
                "total_rows": len(df),
                "numeric_features": len(numeric),
                "categorical_features": len(categorical),
                "stability": is_stable
            },
            "cost": {
                "budget": self.policy.cost_budget,
                "spent": ctx.cost_spent,
                "remaining": self.policy.cost_budget - ctx.cost_spent,
                "efficiency": ctx.cost_spent / total if total > 0 else 0
            },
            "timestamp": now
        }

        return {
            "profile": profile,
            "contract": contract,
            "report": report,
            "metrics": contract["signals"],
            "success": passed
        }


    def run(self, ctx: RuntimeContext):
        ctx.current_phase = "data"

        # Validate data
        if not self.dataset_path.exists():
            return PhaseResult(
                phase=ctx.goal,
                success=False,
                contract={"error": f"Dataset not found: {self.dataset_path}"}
            )

        # Load dataset
        df = pd.read_csv(self.dataset_path)
        print(f"[LOAD] {len(df)} rows × {len(df.columns)} cols")
        
        # Feature
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Column analysis
        columns = {}
        for col in df.columns:
            columns[col] = {
                "dtype": str(df[col].dtype),
                "null_ratio": float(df[col].isnull().mean()),
                "unique_count": int(df[col].nunique())
            }

        # Schema fingerprint
        schema_hash = hashlib.md5(
            json.dumps(sorted(columns.items())).encode()
        ).hexdigest()

        # Create signature
        payload = {
            "dataset": self.dataset_name,
            "schema_hash": schema_hash,
            "columns": columns,
        }
        signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode() 
        ).hexdigest()   

        # Early exit: insufficient data
        if len(df) < self.policy.min_rows:
            artifacts = self.build_reports(
                ctx, df, numeric, categorical, columns, schema_hash, signature
            )
            ctx.success = False
            ctx.halted_reason = (
                "Insufficient rows" if len(df) < self.policy.min_rows 
                else "No numeric features"
            )
            return PhaseResult(phase=ctx.goal, **artifacts)


        # Sample Value loop
        for i, row in df.iterrows():
            # Budget check
            if ctx.cost_spent + self.policy.cost_per_sample > ctx.cost_budget:
                print(f"[BUDGET] Halted at row {i}, budget exhausted")
                break

            ctx.cost_spent += self.policy.cost_per_sample
            
            # Create sample
            sample = Sample(
                id=f"s_{uuid.uuid4().hex[:8]}",
                features=np.nan_to_num(row[numeric].values.astype(float)),
                label=None,
                meta={"row": int(i)}
            )

            # Estimate intrinsic value
            est_result, self.cache, reward = estimate(
                sample=sample,
                policy=self.policy,
                cache=self.cache,
                cache_size=self.cache_size
            )
            
            # Update thresholds
            self.reinforcement_update(reward)
            
            # Track statistics
            self.stats["total"] += 1
            self.stats[est_result.decision.value] += 1
            self.values.append(est_result.value_score)

        # Build final artifacts
        artifacts = self.build_reports(
            ctx, df, numeric, categorical, columns, schema_hash, signature
        )
        
        # Update context
        ctx.success = artifacts["success"]
        if not ctx.success:
            ctx.halted_reason = "Data worthiness thresholds not met"

        # Log results
        print(f"[DONE] {self.stats}")
        print(f"[COST] ${ctx.cost_spent:.2f}/${self.policy.cost_budget:.2f}")
        print(f"[STATUS] {artifacts['contract']['status']}")

        return PhaseResult(phase=ctx.goal, **artifacts)
