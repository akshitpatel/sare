"""
MultimodalPerception — stub implementation.
Handles image/table/text → graph perception pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_HISTORY: list = []


class MultimodalPerception:
    """Minimal stub so web.py and Phase 5 imports don't crash."""

    def perceive(self, data: Any, modality: str = "text") -> Optional[object]:
        log.debug("[MultimodalPerception] perceive modality=%s (stub)", modality)
        _HISTORY.append({"modality": modality, "data_len": len(str(data))})
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total": len(_HISTORY),
            "by_modality": {m: sum(1 for h in _HISTORY if h["modality"] == m)
                            for m in {h["modality"] for h in _HISTORY}},
        }


_SINGLETON: Optional[MultimodalPerception] = None


def get_multimodal_perception() -> MultimodalPerception:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = MultimodalPerception()
    return _SINGLETON
