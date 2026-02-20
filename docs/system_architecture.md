# SARE-HX System Documentation

## Overview

**SARE-HX** (Structural Algebra of Realizable Expressions) is a self-optimizing, graph-native cognitive architecture designed to solve complex algebraic and symbolic problems while learning generalized strategies from its own experiences.

The system operates on a hybrid architecture combining:
1.  **C++ Core (Deterministic)**: High-performance graph manipulation, energy-based optimization, and Memory-Guided MCTS.
2.  **Python Layer (Probabilistic)**: Heuristic learning, curiosity-driven curriculum generation, and web interface.

## Architecture Phases

The development is structured into 8 phases, progressing from basic mechanics to advanced metacognition.

### Phase 1: Deterministic Core ✅
Establishes the fundamental data structures and operations.
-   **Graph Engine**: Handles heterogeneous nodes (Variables, Constants, Operators) and edges.
-   **Energy System**: Evaluates solution quality using a multi-component objective function (Syntactic, Conceptual, Complexity, etc.).
-   **Transforms**: Axiomatic rules (Algebra, Logic, AST) that modify the graph state.
-   **Search**: Beam Search for local optimization.

### Phase 2: Memory & Heuristics ✅
Introduces learning from past solves to guide future search.
-   **Episodic Memory**: Stores full solve traces (Graph Trajectory, Energy Profile).
-   **Strategy Memory**: Maps problem structure signatures to successful transform sequences.
-   **Heuristic Model**: GNN-based value function $V_\theta(G)$ trained to predict energy reduction potential.
-   **MCTS**: Monte Carlo Tree Search guided by both heuristics and memory.

### Phase 3: Abstraction Engine ✅
Enables the system to create its own complex operations.
-   **Trace Mining**: Detects frequent subsequences in successful solve traces.
-   **Macro Builder**: Promotes frequent patterns into new atomic `MacroTransform` units.
-   **Abstraction Registry**: Manages the lifecycle of learned macros.

### Phase 4: Structural Plasticity ✅
Allows the cognitive architecture to evolve its own topology.
-   **Module Generator**: Proposes new transform types based on persistent failure modes.
-   **Sandbox Runner**: Safely evaluates new modules in isolation before full integration.
-   **Pruning**: Removes underperforming or obsolete modules.

### Phase 5: Causal Modeling ✅
Reasoning about "Why" and "What If".
-   **Intervention**: $do(X=x)$ operator to structurally modify graphs.
-   **Counterfactuals**: Simulating alternative outcomes from past states.
-   **Hypothesis Ranking**: selecting causal explanations based on Occam's Razor.

### Phase 6: The Reflective Loop (Why?) 🚧
Self-supervised rule learning from success.
-   **Reflection Engine**: Computes the graph difference $\Delta(S_{initial}, S_{final})$ after a successful solve.
-   **Concept Registry**: Generalizes $\Delta$ into reusable abstract rules (Concepts).

### Phase 7: Active Curiosity (What if?) 🚧
Self-directed exploration.
-   **Curriculum Generator**: Mutates known problems to create a "frontier" of progressively harder tasks.
-   **Web UI**: Visualizes the learned concepts and the curiosity frontier.

## Installation & Build

### Prerequisites
-   **C++ Compiler**: GCC 9+ or Clang 10+ (C++17 support required).
-   **CMake**: 3.16+
-   **Python**: 3.8+ (3.13 tested)
-   **Dependencies**: `googletest`, `pybind11` (automatically fetched by CMake).

### Building from Source

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/akshitpatel/sare-hx.git
    cd sare-hx
    ```

2.  **Configure and Build**:
    ```bash
    mkdir build && cd build
    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build . -- -j$(nproc)
    ```

    *Note for macOS users:* If you encounter `<cstdint>` errors, try running:
    ```bash
    export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
    ```
    or rely on the `CMakeLists.txt` fix implemented in this repo.

3.  **Install Python Bindings**:
    Locate the generated `.so` file in `build/` (e.g., `sare_bindings.cpython-313-darwin.so`).
    Copy it to the `python/` directory and rename if necessary (though the `sare` package expects it in `PYTHONPATH`).
    ```bash
    cp build/sare_bindings*.so python/sare_bindings.so
    ```

### Running the Web Interface

1.  **Set PYTHONPATH**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/python
    ```
    Ensure `sare_bindings.so` is importable. You can test with:
    ```bash
    python3 -c "import sare_bindings; print('Bindings loaded!')"
    ```

2.  **Start the Server**:
    ```bash
    python3 -m sare.interface.web
    ```
    Access the UI at `http://localhost:8080`.

## API Usage

### Core Python API

```python
import sare_bindings as sb

# 1. Create a Graph
g = sb.Graph()
n1 = g.add_node_with_id(1, "x", "variable")
n2 = g.add_node_with_id(2, "2", "constant")
g.add_edge(n1, n2, "add")

# 2. Evaluate Energy
energy = sb.EnergyAggregator()
result = energy.evaluate(g)
print(f"Energy: {result.total}")

# 3. Solve (Beam Search)
searcher = sb.BeamSearch()
transforms = sb.TransformRegistry.default_registry()
result = searcher.search(g, energy, transforms, width=8, budget_seconds=5.0)

if result.success:
    print("Solved!", result.final_expression)
```

## Directory Structure

-   `core/`: C++ source code (the brain).
    -   `graph/`, `energy/`, `search/`, `transforms/`: Functional distinct modules.
    -   `bindings/`: PyBind11 definitions.
-   `python/`: Python glue code and high-level logic.
    -   `sare/interface/`: Web and CLI interfaces.
    -   `sare/curiosity/`: Phase 7 curiosity modules.
-   `tests/`: C++ GTest suites.
