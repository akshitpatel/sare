"""
MultimodalPerception — functional implementation.
Handles image/table/text/code → graph perception pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sare.engine import Graph
from sare.perception.graph_builders import (
    CodeGraphBuilder,
    SentenceGraphBuilder,
    PlanGraphBuilder,
)

log = logging.getLogger(__name__)

_HISTORY: list[Dict[str, Any]] = []


class MultimodalPerception:
    """Perceive various modalities and produce a reasoning Graph."""

    def perceive(self, data: Any, modality: str = "text") -> Optional[Graph]:
        """
        Convert raw input into a Graph suitable for the reasoning core.

        Supported modalities:
            - "text": natural‑language sentences → SentenceGraphBuilder
            - "code": source‑code strings → CodeGraphBuilder
            - "plan": simple planning descriptions → PlanGraphBuilder
            - "image": currently unsupported (returns None)

        Returns:
            Graph instance on success, otherwise None.
        """
        log.debug("[MultimodalPerception] perceive modality=%s", modality)

        result: Optional[Graph] = None
        try:
            if modality == "text":
                result = self._build_from_text(str(data))
            elif modality == "code":
                result = self._build_from_code(str(data))
            elif modality == "plan":
                result = self._build_from_plan(str(data))
            elif modality == "image":
                result = self._build_from_image(data)
            else:
                log.warning("Unsupported modality %s; falling back to text", modality)
                result = self._build_from_text(str(data))
        except Exception as exc:  # pragma: no cover
            log.exception("Error during perception of modality %s: %s", modality, exc)
            result = None

        _HISTORY.append(
            {
                "modality": modality,
                "data_len": len(str(data)),
                "graph_nodes": result.node_count() if result else 0,
                "graph_edges": result.edge_count() if result else 0,
            }
        )
        return result

    # --------------------------------------------------------------------- #
    # Private builder helpers
    # --------------------------------------------------------------------- #
    def _build_from_text(self, text: str) -> Optional[Graph]:
        """Parse natural‑language text into a Graph using SentenceGraphBuilder."""
        builder = SentenceGraphBuilder()
        try:
            graph = builder.build(text)
            log.debug("SentenceGraphBuilder produced %d nodes, %d edges", graph.node_count(), graph.edge_count())
            return graph
        except Exception as exc:  # pragma: no cover
            log.error("SentenceGraphBuilder failed: %s", exc)
            return None

    def _build_from_code(self, code: str) -> Optional[Graph]:
        """Parse source‑code string into a Graph using CodeGraphBuilder."""
        builder = CodeGraphBuilder()
        try:
            graph = builder.build(code)
            log.debug("CodeGraphBuilder produced %d nodes, %d edges", graph.node_count(), graph.edge_count())
            return graph
        except Exception as exc:  # pragma: no cover
            log.error("CodeGraphBuilder failed: %s", exc)
            return None

    def _build_from_plan(self, plan: str) -> Optional[Graph]:
        """Parse simple planning description into a Graph using PlanGraphBuilder."""
        builder = PlanGraphBuilder()
        try:
            graph = builder.build(plan)
            log.debug("PlanGraphBuilder produced %d nodes, %d edges", graph.node_count(), graph.edge_count())
            return graph
        except Exception as exc:  # pragma: no cover
            log.error("PlanGraphBuilder failed: %s", exc)
            return None

    def _build_from_image(self, image_data: Any) -> Optional[Graph]:
        """
        Placeholder for image perception.
        Future work: integrate vision model → symbolic graph.
        """
        log.info("Image modality received but not yet implemented.")
        return None

    # --------------------------------------------------------------------- #
    # Public diagnostics
    # --------------------------------------------------------------------- #
    def get_stats(self) -> Dict[str, Any]:
        """Return simple usage statistics for debugging/monitoring."""
        by_modality: Dict[str, int] = {}
        for entry in _HISTORY:
            m = entry["modality"]
            by_modality[m] = by_modality.get(m, 0) + 1
        return {"total": len(_HISTORY), "by_modality": by_modality}


_SINGLETON: Optional[MultimodalPerception] = None


def get_multimodal_perception() -> MultimodalPerception:
    """Singleton accessor used throughout the system."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = MultimodalPerception()
    return _SINGLETON