# SARE-HX

**Graph-Native Cognitive Architecture with Developmental Growth**

A persistent, graph-native, energy-driven cognitive system for structured reasoning, code optimization, abstraction formation, and cross-domain learning.

## Architecture

| Subsystem | Location | Language | Phase |
|---|---|---|---|
| Structured Cognitive Graph | `core/graph/` | C++ | 1 |
| Energy Evaluation Engine | `core/energy/` | C++ | 1 |
| Transformation Module Engine | `core/transforms/` | C++ | 1 |
| Search & Simulation Engine | `core/search/` | C++ | 1 |
| Verification Layer | `core/verification/` | C++ | 1 |
| Memory & Strategy Store | `core/memory/` | C++ | 2 |
| Heuristic Learning | `python/sare/heuristics/` | Python | 2 |
| Abstraction Formation | `core/abstraction/` | C++ | 3 |
| Structural Plasticity | `core/plasticity/` | C++ | 4 |
| Causal Modeling | `core/causal/` | C++ | 5 |

## Build

```bash
# C++ core (recommended)
cmake -S . -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run C++ tests
ctest --test-dir build --output-on-failure

# Python package
pip install -e .
```

## Design Principles

1. Stability before growth
2. Verification before creativity
3. Determinism before heuristics
4. Measurement before expansion
5. Never remove safety constraints
