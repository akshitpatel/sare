"""
SARE-HX World Grounder

Converts real-world data (CSV, text, JSON, URL) into a SARE-HX typed graph.
"""
from __future__ import annotations

import csv
import json
import re
import logging
import urllib.request
import io
from pathlib import Path
from typing import List, Optional, Dict, Any

log = logging.getLogger(__name__)


class RawPercept:
    def __init__(self, kind: str, payload: str, source: str = ""):
        self.kind = kind
        self.payload = payload
        self.source = source


class GraphDict:
    def __init__(self):
        self.nodes: List[dict] = []
        self.edges: List[dict] = []
        self._node_counter = 0

    def add_node(self, label: str, node_type: str, attributes: dict = None) -> int:
        nid = self._node_counter
        self._node_counter += 1
        self.nodes.append({"id": nid, "label": label, "type": node_type, "attributes": attributes or {}})
        return nid

    def add_edge(self, source: int, target: int, relation: str):
        self.edges.append({"source": source, "target": target, "relation": relation})

    def to_engine_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}


class CSVGrounder:
    def ground(self, csv_text: str, source: str = "") -> GraphDict:
        g = GraphDict()
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        if not rows:
            return g
        headers = rows[0]
        header_ids = [g.add_node(h, "attribute") for h in headers]
        for row in rows[1:]:
            row_id = g.add_node(source or "row", "entity")
            for i, val in enumerate(row):
                if i < len(header_ids):
                    val_id = g.add_node(val, "value")
                    g.add_edge(row_id, val_id, headers[i])
        return g


class TextGrounder:
    def ground(self, text: str) -> GraphDict:
        g = GraphDict()

        def get_or_create(label: str, ntype: str) -> int:
            for n in g.nodes:
                if n["label"] == label and n["type"] == ntype:
                    return n["id"]
            return g.add_node(label, ntype)

        # Simple subject-verb-object extraction via pattern
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            tokens = sent.split()
            if len(tokens) >= 3:
                subj_id = get_or_create(tokens[0].lower(), "entity")
                verb_id = get_or_create(tokens[1].lower(), "relation")
                obj_id = get_or_create(" ".join(tokens[2:]).lower(), "entity")
                g.add_edge(subj_id, obj_id, tokens[1].lower())
            elif len(tokens) >= 1:
                get_or_create(tokens[0].lower(), "entity")
        return g


class JSONGrounder:
    def ground(self, data: Any, parent_id: int = None, g: GraphDict = None, depth: int = 0) -> GraphDict:
        if g is None:
            g = GraphDict()
        if depth > 5:
            return g
        if isinstance(data, dict):
            node_id = g.add_node("object", "entity")
            if parent_id is not None:
                g.add_edge(parent_id, node_id, "contains")
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    child_id = g.add_node(str(k), "attribute")
                    g.add_edge(node_id, child_id, "has")
                    self.ground(v, child_id, g, depth + 1)
                else:
                    val_id = g.add_node(str(v), "value")
                    g.add_edge(node_id, val_id, str(k))
        elif isinstance(data, list):
            for item in data[:20]:  # cap at 20 items
                self.ground(item, parent_id, g, depth + 1)
        else:
            node_id = g.add_node(str(data), "value")
            if parent_id is not None:
                g.add_edge(parent_id, node_id, "item")
        return g


class ToyWorldGrounder:
    def ground(self, data: dict) -> GraphDict:
        g = GraphDict()

        def ensure_entity(entity: str) -> int:
            for n in g.nodes:
                if n["label"] == entity and n["type"] == "entity":
                    return n["id"]
            return g.add_node(entity, "entity")

        entities = data.get("entities", [])
        relations = data.get("relations", [])
        for e in entities:
            ensure_entity(str(e))
        for r in relations:
            if len(r) >= 3:
                src_id = ensure_entity(str(r[0]))
                tgt_id = ensure_entity(str(r[2]))
                g.add_edge(src_id, tgt_id, str(r[1]))
        return g


class URLGrounder:
    def ground(self, url: str, timeout: int = 10) -> GraphDict:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            text_g = TextGrounder()
            return text_g.ground(content[:2000])
        except Exception as e:
            g = GraphDict()
            g.add_node(f"error:{e}", "error")
            return g


class WorldGrounder:
    def __init__(self):
        self._csv = CSVGrounder()
        self._text = TextGrounder()
        self._json = JSONGrounder()
        self._toy = ToyWorldGrounder()
        self._url = URLGrounder()

    def ground(self, percept: RawPercept) -> GraphDict:
        kind = percept.kind.lower()
        payload = percept.payload
        if kind == "csv":
            return self._csv.ground(payload, percept.source)
        elif kind == "json":
            try:
                data = json.loads(payload)
            except Exception:
                data = {"raw": payload}
            return self._json.ground(data)
        elif kind == "url":
            return self._url.ground(payload)
        elif kind == "toy":
            try:
                data = json.loads(payload)
            except Exception:
                data = {}
            return self._toy.ground(data)
        else:  # text (default)
            return self._text.ground(payload)

    def ground_file(self, path: str) -> GraphDict:
        p = Path(path)
        if not p.exists():
            g = GraphDict()
            g.add_node("file_not_found", "error")
            return g
        content = p.read_text(errors="replace")
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return self._csv.ground(content, str(p))
        elif suffix == ".json":
            try:
                return self._json.ground(json.loads(content))
            except Exception:
                return self._text.ground(content)
        else:
            return self._text.ground(content)

    def to_engine_graph(self, gd: GraphDict):
        """Convert GraphDict to sare.engine.Graph"""
        from sare.engine import Graph
        g = Graph()
        id_map: Dict[int, int] = {}
        for n in gd.nodes:
            new_id = g.add_node(n["type"], n["label"])
            id_map[n["id"]] = new_id
        for e in gd.edges:
            src = id_map.get(e["source"])
            tgt = id_map.get(e["target"])
            if src is not None and tgt is not None:
                g.add_edge(src, tgt, e["relation"])
        return g


class WorldEvent:
    def __init__(self, event_type: str, subject: str, predicate: str, obj: str, metadata: dict = None):
        self.event_type = event_type
        self.subject = subject
        self.predicate = predicate
        self.obj = obj
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "metadata": self.metadata,
        }


class StreamingGrounder:
    def __init__(self):
        self._world: Dict[str, Any] = {}
        self._events: List[WorldEvent] = []
        self._beliefs: Dict[str, Dict[str, Any]] = {}

    def initialize(self, world_description: str):
        grounder = WorldGrounder()
        gd = grounder.ground(RawPercept("text", world_description))
        for n in gd.nodes:
            self._world[n["label"]] = {"type": n["type"]}

    def apply_event(self, event: WorldEvent):
        self._events.append(event)
        key = f"{event.subject}.{event.predicate}"
        self._world[key] = event.obj

    def query_belief(self, agent: str, about: str) -> Any:
        return self._beliefs.get(agent, {}).get(about)

    def get_world_state(self) -> dict:
        return dict(self._world)

    def replay(self, events: List[WorldEvent]):
        for e in events:
            self.apply_event(e)

    def to_dict(self) -> dict:
        return {
            "world": self._world,
            "events": [e.to_dict() for e in self._events],
        }


_streaming_grounder: Optional[StreamingGrounder] = None


def get_streaming_grounder() -> StreamingGrounder:
    global _streaming_grounder
    if _streaming_grounder is None:
        _streaming_grounder = StreamingGrounder()
    return _streaming_grounder
