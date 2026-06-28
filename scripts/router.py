"""Perian Phase 2 — Multi-model router with confidence escalation.

Routes each LLM step to the cheapest model that meets its capability requirement.
When the cheap model signals low confidence (via logprobs), escalates to the
strongest model for a second pass.

Key concepts:
- Capability-based routing: each step declares what it needs (extraction,
  conditional_reasoning, etc.) and the router picks the smallest model
  whose score for that capability meets the step's min_reliability threshold.
- Confidence escalation: after the cheap model runs, its logprob stats indicate
  how "sure" it was. Low mean_logprob → uncertain output → re-run on 32B.
- Circuit breaker: caps escalation at max_escalation_rate to prevent runaway costs
  when the cheap model is systematically too weak for the current data.
"""

import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FLEET_PATH = ROOT / "config" / "fleet.yaml"
ROUTER_CONFIG_PATH = ROOT / "config" / "router.yaml"


class Router:
    """Selects the cheapest qualifying model per step, manages escalation."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.fleet = self._load_fleet()
        self.escalation_cfg = self._load_escalation_config()

        # Per-run escalation tracking, keyed by step_id
        self.escalation_stats: dict[str, dict] = {}

        # Track which models are currently loaded (for keep-alive management)
        self.loaded_models: set[str] = set()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_fleet(self) -> dict:
        if not FLEET_PATH.exists():
            self.logger.warning(f"Fleet config not found at {FLEET_PATH}, using defaults")
            return {"models": [], "default_min_reliability": 0.80, "specialist_overrides": {}}
        with open(FLEET_PATH) as f:
            raw = yaml.safe_load(f)
        fleet = raw.get("fleet", {})
        # Index models by id for quick lookup
        fleet["_by_id"] = {m["id"]: m for m in fleet.get("models", [])}
        return fleet

    def _load_escalation_config(self) -> dict:
        defaults = {
            "mode": "logging_only",
            "metric": "mean_logprob",
            "threshold": -1.5,
            "max_escalation_rate": 0.25,
            "log_both_outputs": True,
            "escalation_target": "qwen3.5-27b-int4",
        }
        if not ROUTER_CONFIG_PATH.exists():
            self.logger.warning(f"Router config not found at {ROUTER_CONFIG_PATH}, using defaults")
            return defaults
        with open(ROUTER_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        cfg = raw.get("escalation", {})
        for k, v in defaults.items():
            if k not in cfg:
                cfg[k] = v
        return cfg

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def get_model(self, step: dict) -> dict:
        """Pick the cheapest model meeting the step's capability requirement.

        Returns a dict with at least: id, ollama_name, size_gb.
        If the step has a manual `model` override, that takes priority.
        """
        # Manual override — bypass routing entirely
        if "model" in step and "capability" not in step:
            return self._model_from_override(step["model"])

        capability = step.get("capability")
        if not capability:
            # No capability declared — fall back to largest model
            return self._largest_model(reason="no capability declared on step")

        min_rel = step.get("min_reliability", self.fleet.get("default_min_reliability", 0.80))
        models = self.fleet.get("models", [])

        # Check specialist overrides first
        overrides = self.fleet.get("specialist_overrides", {})
        if capability in overrides:
            preferred_id = overrides[capability]["prefer"]
            preferred = self.fleet["_by_id"].get(preferred_id)
            if preferred:
                score = preferred.get("capabilities", {}).get(capability, 0)
                if score >= min_rel:
                    self.logger.info(
                        f"Router: step '{step['id']}' → {preferred['id']} "
                        f"(specialist override for {capability}, score={score:.2f} ≥ {min_rel})"
                    )
                    return preferred

        # Find cheapest model (smallest size_gb) meeting the threshold
        candidates = []
        for m in models:
            score = m.get("capabilities", {}).get(capability, 0)
            if score >= min_rel:
                candidates.append((m, score))

        if not candidates:
            return self._largest_model(
                reason=f"no model meets {capability} ≥ {min_rel}"
            )

        # Sort by size ascending (cheapest first)
        candidates.sort(key=lambda x: x[0]["size_gb"])
        chosen, score = candidates[0]

        self.logger.info(
            f"Router: step '{step['id']}' → {chosen['id']} "
            f"({capability}={score:.2f} ≥ {min_rel}, size={chosen['size_gb']}GB)"
        )
        return chosen

    def _model_from_override(self, model_name: str) -> dict:
        """Look up a model by ollama_name or id from the fleet, or build a stub."""
        for m in self.fleet.get("models", []):
            if m["ollama_name"] == model_name or m["id"] == model_name:
                self.logger.info(f"Router: using manual override → {m['id']}")
                return m
        # Model not in fleet config — return a stub so runner still works
        self.logger.warning(f"Router: manual override '{model_name}' not in fleet config")
        return {"id": model_name, "ollama_name": model_name, "size_gb": 0}

    def _largest_model(self, reason: str = "") -> dict:
        """Fall back to the largest model in the fleet."""
        models = self.fleet.get("models", [])
        if not models:
            return {"id": "qwen3.5:27b-int4", "ollama_name": "qwen3.5:27b-int4", "size_gb": 16}
        largest = max(models, key=lambda m: m["size_gb"])
        self.logger.warning(f"Router: falling back to {largest['id']} ({reason})")
        return largest

    def get_escalation_target(self) -> dict:
        """Return the model config for the escalation target."""
        target_id = self.escalation_cfg.get("escalation_target", "qwen3.5-27b-int4")
        model = self.fleet["_by_id"].get(target_id)
        if model:
            return model
        return self._largest_model(reason=f"escalation target '{target_id}' not found")

    # ------------------------------------------------------------------
    # Confidence escalation
    # ------------------------------------------------------------------

    def should_escalate(self, step_id: str, logprob_stats: dict | None,
                        initial_model: dict) -> dict:
        """Decide whether to escalate an item based on logprob confidence.

        Returns a decision dict:
          {
            "escalate": bool,
            "reason": str,
            "mode": "logging_only" | "active",
            "metric_value": float | None,
            "threshold": float,
          }

        The caller checks decision["escalate"] and decision["mode"] to decide
        whether to actually re-run the item.
        """
        target = self.get_escalation_target()
        threshold = self.escalation_cfg["threshold"]
        mode = self.escalation_cfg["mode"]

        # Already on the strongest model — no point escalating
        if initial_model["id"] == target["id"]:
            return {
                "escalate": False,
                "reason": "already on escalation target",
                "mode": mode,
                "metric_value": None,
                "threshold": threshold,
            }

        # No logprob data available
        if not logprob_stats:
            return {
                "escalate": False,
                "reason": "no logprob data available",
                "mode": mode,
                "metric_value": None,
                "threshold": threshold,
            }

        metric = self.escalation_cfg.get("metric", "mean_logprob")
        value = logprob_stats.get(metric)
        if value is None:
            return {
                "escalate": False,
                "reason": f"metric '{metric}' not in logprob stats",
                "mode": mode,
                "metric_value": None,
                "threshold": threshold,
            }

        # Check circuit breaker
        stats = self._get_step_stats(step_id)
        max_rate = self.escalation_cfg["max_escalation_rate"]
        if stats["total"] > 0:
            current_rate = stats["escalated"] / stats["total"]
            if current_rate >= max_rate:
                return {
                    "escalate": False,
                    "reason": f"circuit breaker: escalation rate {current_rate:.2f} ≥ {max_rate}",
                    "mode": mode,
                    "metric_value": value,
                    "threshold": threshold,
                }

        # Check threshold
        if value < threshold:
            return {
                "escalate": True,
                "reason": f"{metric}={value:.4f} < threshold={threshold}",
                "mode": mode,
                "metric_value": value,
                "threshold": threshold,
            }

        return {
            "escalate": False,
            "reason": f"{metric}={value:.4f} ≥ threshold={threshold} (confident)",
            "mode": mode,
            "metric_value": value,
            "threshold": threshold,
        }

    def record_escalation_decision(self, step_id: str, escalated: bool,
                                   circuit_breaker_prevented: bool = False) -> None:
        """Track escalation counts for metrics and circuit breaker."""
        stats = self._get_step_stats(step_id)
        stats["total"] += 1
        if escalated:
            stats["escalated"] += 1
        if circuit_breaker_prevented:
            stats["prevented"] += 1

    def _get_step_stats(self, step_id: str) -> dict:
        if step_id not in self.escalation_stats:
            self.escalation_stats[step_id] = {
                "total": 0,
                "escalated": 0,
                "prevented": 0,
            }
        return self.escalation_stats[step_id]

    def get_escalation_metrics(self) -> dict:
        """Build the escalation section for metrics.json."""
        total_items = sum(s["total"] for s in self.escalation_stats.values())
        total_escalated = sum(s["escalated"] for s in self.escalation_stats.values())
        total_prevented = sum(s["prevented"] for s in self.escalation_stats.values())

        by_step = {}
        for step_id, stats in self.escalation_stats.items():
            entry = {
                "total": stats["total"],
                "escalated": stats["escalated"],
                "rate": round(stats["escalated"] / stats["total"], 3) if stats["total"] else 0,
            }
            if stats["prevented"]:
                entry["prevented_by_circuit_breaker"] = stats["prevented"]
            by_step[step_id] = entry

        return {
            "total_items": total_items,
            "escalated_count": total_escalated,
            "escalation_rate": round(total_escalated / total_items, 3) if total_items else 0,
            "circuit_breaker_triggered": total_prevented > 0,
            "prevented_escalations": total_prevented,
            "mode": self.escalation_cfg["mode"],
            "threshold": self.escalation_cfg["threshold"],
            "by_step": by_step,
        }

    # ------------------------------------------------------------------
    # Keep-alive orchestration
    # ------------------------------------------------------------------

    def get_models_for_steps(self, steps: list[dict]) -> list[tuple[dict, dict]]:
        """For a list of LLM steps, return [(step, model_config), ...].

        Use this to plan model loading order and minimize swaps.
        """
        return [(step, self.get_model(step)) for step in steps]

    def plan_model_batches(self, steps: list[dict]) -> list[dict]:
        """Group consecutive steps by model to minimize model swaps.

        Returns a list of batch dicts:
          [{"model": model_config, "steps": [step, ...], "is_last": bool}, ...]
        """
        step_models = self.get_models_for_steps(steps)
        if not step_models:
            return []

        batches = []
        current_model = step_models[0][1]
        current_steps = [step_models[0][0]]

        for step, model in step_models[1:]:
            if model["id"] == current_model["id"]:
                current_steps.append(step)
            else:
                batches.append({"model": current_model, "steps": current_steps})
                current_model = model
                current_steps = [step]

        batches.append({"model": current_model, "steps": current_steps})

        # Mark the last batch
        for i, batch in enumerate(batches):
            batch["is_last"] = (i == len(batches) - 1)

        return batches
