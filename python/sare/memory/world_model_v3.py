"""
world_model_v3.py — SHIM

WorldModelV3 has been merged into WorldModel in world_model.py.
This file re-exports everything for any future imports.

The unified WorldModel now includes:
  - V2: Facts, CausalLinks, Schemas, predict_transform, enrich_from_proof, etc.
  - V3: Beliefs, Analogies, solve_history, CausalDiscovery, SchemaInduction,
         ContradictionDetector, discover_analogies, predict, check_consistency
"""
from sare.memory.world_model import (
    WorldModel,
    get_world_model,
    on_rule_promoted,
    CausalLink,
    Schema,
    Fact,
    Belief,
    Analogy,
    Prediction,
    CausalDiscovery,
    SchemaInduction,
    ContradictionDetector,
)

# Alias for any code that imported WorldModelV3
WorldModelV3 = WorldModel


def get_world_model_v3() -> WorldModel:
    """Alias — returns the same unified singleton as get_world_model()."""
    return get_world_model()
