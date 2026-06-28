#!/usr/bin/env python3
"""Shared Ollama model management utilities.

Extracted from runner.py so that both the pipeline and Shadowfax (nightly
orchestrator) can manage model warm-up/unload without duplication.
"""

import logging

import requests


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        return False


def warm_up_model(model: str, config: dict, logger: logging.Logger) -> None:
    """Load model with keep_alive=-1 so it stays in memory for the entire run."""
    ollama_cfg = config.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    keep_alive = ollama_cfg.get("keep_alive", -1)

    logger.info(f"Warming up model {model} (keep_alive={keep_alive})")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": " "}],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        logger.info(f"Model {model} loaded and pinned in memory")
    except requests.RequestException as e:
        logger.warning(f"Model warm-up failed (non-fatal): {e}")


def unload_model(model: str, config: dict, logger: logging.Logger) -> None:
    """Immediately unload a model to free VRAM (keep_alive=0)."""
    ollama_cfg = config.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": " "}],
        "stream": False,
        "think": False,
        "keep_alive": "0",
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        logger.info(f"Model {model} unloaded (keep_alive=0)")
    except requests.RequestException as e:
        logger.warning(f"Model unload failed (non-fatal): {e}")


def cool_down_model(model: str, config: dict, logger: logging.Logger) -> None:
    """Reset keep_alive so model unloads after inactivity."""
    ollama_cfg = config.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    keep_alive_after = ollama_cfg.get("default_keep_alive_after", "5m")

    logger.info(f"Resetting model {model} keep_alive to {keep_alive_after}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": " "}],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive_after,
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        logger.info(f"Model {model} will unload after {keep_alive_after} of inactivity")
    except requests.RequestException as e:
        logger.warning(f"Model cool-down failed (non-fatal): {e}")
