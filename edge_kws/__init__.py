"""Compact keyword-spotting models for edge-device experiments."""

from .registry import MODEL_NAMES, KWSSystem, build_frontend, build_model, build_system

__all__ = ["MODEL_NAMES", "KWSSystem", "build_frontend", "build_model", "build_system"]

