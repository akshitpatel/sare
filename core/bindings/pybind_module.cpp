// PyBind11 bindings for SARE-HX C++ core.
// Exposes Graph, Energy, Search, Transform, and Verification APIs to Python.

// NOTE: Requires pybind11 to be installed.
// Build with: cmake -DBUILD_PYTHON_BINDINGS=ON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "graph/graph.hpp"
#include "graph/graph_diff.hpp"
#include "graph/subgraph_matcher.hpp"
#include "graph/graph_snapshot.hpp"
#include "energy/energy.hpp"
#include "energy/default_energy.hpp"
#include "transforms/transform_base.hpp"
#include "transforms/transform_registry.hpp"
#include "transforms/default_transforms.hpp"
#include "search/search_state.hpp"
#include "search/beam_search.hpp"
#include "search/mcts.hpp"
#include "search/search_controller.hpp"
#include "verification/verification.hpp"
#include "reflection/reflection_engine.hpp"
#include "reflection/concept_registry.hpp"
#include "reflection/causal_induction.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sare_bindings, m) {
    m.doc() = "SARE-HX C++ Core Bindings";

    // ── Node ──
    py::class_<sare::Node>(m, "Node")
        .def(py::init<>())
        .def(py::init<uint64_t, std::string>())
        .def_readwrite("id", &sare::Node::id)
        .def_readwrite("type", &sare::Node::type)
        .def_readwrite("activation", &sare::Node::activation)
        .def_readwrite("uncertainty", &sare::Node::uncertainty)
        .def("set_attribute", &sare::Node::setAttribute)
        .def("get_attribute", &sare::Node::getAttribute,
             py::arg("key"), py::arg("default_val") = "")
        .def("has_attribute", &sare::Node::hasAttribute);

    // ── Edge ──
    py::class_<sare::Edge>(m, "Edge")
        .def(py::init<>())
        .def(py::init<uint64_t, uint64_t, uint64_t, std::string, double>())
        .def_readwrite("id", &sare::Edge::id)
        .def_readwrite("source", &sare::Edge::source)
        .def_readwrite("target", &sare::Edge::target)
        .def_readwrite("relationship_type", &sare::Edge::relationship_type)
        .def_readwrite("weight", &sare::Edge::weight);

    // ── GraphDelta ──
    py::class_<sare::GraphDelta>(m, "GraphDelta")
        .def(py::init<>())
        .def("empty", &sare::GraphDelta::empty)
        .def_readwrite("added_nodes", &sare::GraphDelta::added_nodes)
        .def_readwrite("added_edges", &sare::GraphDelta::added_edges)
        .def_readwrite("removed_node_ids", &sare::GraphDelta::removed_node_ids)
        .def_readwrite("removed_edge_ids", &sare::GraphDelta::removed_edge_ids);

    // ── Graph ──
    py::class_<sare::Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_node", &sare::Graph::addNode)
        .def("add_node_with_id", &sare::Graph::addNodeWithId)
        .def("remove_node", &sare::Graph::removeNode)
        .def("get_node", py::overload_cast<uint64_t>(&sare::Graph::getNode),
             py::return_value_policy::reference)
        .def("get_node_ids", &sare::Graph::getNodeIds)
        .def("node_count", &sare::Graph::nodeCount)
        .def("add_edge", &sare::Graph::addEdge,
             py::arg("source"), py::arg("target"),
             py::arg("relationship_type"), py::arg("weight") = 1.0)
        .def("add_edge_with_id", &sare::Graph::addEdgeWithId)
        .def("remove_edge", &sare::Graph::removeEdge)
        .def("get_edge", py::overload_cast<uint64_t>(&sare::Graph::getEdge),
             py::return_value_policy::reference)
        .def("get_edge_ids", &sare::Graph::getEdgeIds)
        .def("edge_count", &sare::Graph::edgeCount)
        .def("get_outgoing", &sare::Graph::getOutgoing)
        .def("get_incoming", &sare::Graph::getIncoming)
        .def("get_neighbor_nodes", &sare::Graph::getNeighborNodes)
        .def("extract_subgraph", &sare::Graph::extractSubgraph)
        .def("apply_delta", &sare::Graph::applyDelta)
        .def("undo_delta", &sare::Graph::undoDelta)
        .def("clone", &sare::Graph::clone);

    // ── EnergyBreakdown ──
    py::class_<sare::EnergyBreakdown>(m, "EnergyBreakdown")
        .def(py::init<>())
        .def_readwrite("syntax", &sare::EnergyBreakdown::syntax)
        .def_readwrite("constraint", &sare::EnergyBreakdown::constraint)
        .def_readwrite("test_failure", &sare::EnergyBreakdown::test_failure)
        .def_readwrite("complexity", &sare::EnergyBreakdown::complexity)
        .def_readwrite("resource", &sare::EnergyBreakdown::resource)
        .def_readwrite("uncertainty", &sare::EnergyBreakdown::uncertainty)
        .def("total", &sare::EnergyBreakdown::total);

    // ── EnergyWeights ──
    py::class_<sare::EnergyWeights>(m, "EnergyWeights")
        .def(py::init<>())
        .def_readwrite("alpha", &sare::EnergyWeights::alpha)
        .def_readwrite("beta", &sare::EnergyWeights::beta)
        .def_readwrite("gamma", &sare::EnergyWeights::gamma)
        .def_readwrite("delta", &sare::EnergyWeights::delta)
        .def_readwrite("lambda_", &sare::EnergyWeights::lambda)
        .def_readwrite("mu", &sare::EnergyWeights::mu);

    // ── SearchConfig ──
    py::class_<sare::SearchConfig>(m, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("beam_width", &sare::SearchConfig::beam_width)
        .def_readwrite("max_depth", &sare::SearchConfig::max_depth)
        .def_readwrite("kappa", &sare::SearchConfig::kappa)
        .def_readwrite("budget_seconds", &sare::SearchConfig::budget_seconds)
        .def_readwrite("max_expansions", &sare::SearchConfig::max_expansions);

    // ── SearchState ──
    py::class_<sare::SearchState>(m, "SearchState")
        .def(py::init<>())
        .def_readwrite("id", &sare::SearchState::id)
        .def_readwrite("energy", &sare::SearchState::energy)
        .def_readwrite("score", &sare::SearchState::score)
        .def_readwrite("transform_trace", &sare::SearchState::transform_trace)
        .def_readwrite("depth", &sare::SearchState::depth);

    // ── SearchResult ──
    py::class_<sare::SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readwrite("best_state", &sare::SearchResult::best_state)
        .def_readwrite("best_graph", &sare::SearchResult::best_graph)
        .def_readwrite("total_expansions", &sare::SearchResult::total_expansions)
        .def_readwrite("max_depth_reached", &sare::SearchResult::max_depth_reached)
        .def_readwrite("elapsed_seconds", &sare::SearchResult::elapsed_seconds)
        .def_readwrite("budget_exhausted", &sare::SearchResult::budget_exhausted);

    // ── McsResult ──
    py::class_<sare::McsResult>(m, "McsResult")
        .def(py::init<>())
        .def_readwrite("mapping", &sare::McsResult::mapping)
        .def_readwrite("score",   &sare::McsResult::score);

    // ── SubgraphMatcher ──
    py::class_<sare::SubgraphMatcher>(m, "SubgraphMatcher")
        .def_static("find_mcs", &sare::SubgraphMatcher::find_mcs,
                    py::arg("g1"), py::arg("g2"), py::arg("max_depth") = 8);

    // ── VerificationResult ──
    py::class_<sare::VerificationResult>(m, "VerificationResult")
        .def(py::init<>())
        .def_readwrite("passed", &sare::VerificationResult::passed)
        .def_readwrite("check_name", &sare::VerificationResult::check_name)
        .def_readwrite("message", &sare::VerificationResult::message)
        .def_readwrite("node_id", &sare::VerificationResult::node_id);

    // ── SyntaxChecker ──
    py::class_<sare::SyntaxChecker>(m, "SyntaxChecker")
        .def(py::init<>())
        .def("check", &sare::SyntaxChecker::check);

    py::class_<sare::BeamSearch>(m, "BeamSearch")
        .def(py::init<>())
        .def("search", [](sare::BeamSearch& self, const sare::Graph& graph, const sare::SearchConfig& config) {
            auto energy = sare::makeDefaultEnergyAggregator();
            sare::TransformRegistry registry;
            sare::registerDefaultTransforms(registry);
            return self.search(graph, energy, registry, config);
        }, py::arg("graph"), py::arg("config"));

    py::class_<sare::MCTSSearch>(m, "MCTSSearch")
        .def(py::init<>())
        .def("search", [](sare::MCTSSearch& self, const sare::Graph& graph, const sare::SearchConfig& config) {
            auto energy = sare::makeDefaultEnergyAggregator();
            sare::TransformRegistry registry;
            sare::registerDefaultTransforms(registry);
            return self.search(graph, energy, registry, config);
        }, py::arg("graph"), py::arg("config"));

    // ─── TypeConstraint ──
    py::class_<sare::TypeConstraint>(m, "TypeConstraint")
        .def(py::init<>())
        .def_readwrite("node_id",       &sare::TypeConstraint::node_id)
        .def_readwrite("required_type", &sare::TypeConstraint::required_type)
        .def_readwrite("required_label",&sare::TypeConstraint::required_label);

    // ─── AbstractRule ──  (updated with domain, type_constraints, valid)
    py::class_<sare::AbstractRule>(m, "AbstractRule")
        .def(py::init<>())
        .def_readwrite("name",             &sare::AbstractRule::name)
        .def_readwrite("domain",           &sare::AbstractRule::domain)
        .def_readwrite("pattern",          &sare::AbstractRule::pattern)
        .def_readwrite("replacement",      &sare::AbstractRule::replacement)
        .def_readwrite("type_constraints", &sare::AbstractRule::type_constraints)
        .def_readwrite("confidence",       &sare::AbstractRule::confidence)
        .def_readwrite("observations",     &sare::AbstractRule::observations)
        .def("valid",                      &sare::AbstractRule::valid);
    
    // ─── ReflectionEngine ──
    py::class_<sare::ReflectionEngine>(m, "ReflectionEngine")
        .def(py::init<>())
        .def("reflect",      &sare::ReflectionEngine::reflect)
        .def("compute_diff", &sare::ReflectionEngine::computeDiff)
        .def("generalize",   &sare::ReflectionEngine::generalize);

    // ─── InductionResult ──
    py::class_<sare::InductionResult>(m, "InductionResult")
        .def(py::init<>())
        .def_readwrite("promoted",       &sare::InductionResult::promoted)
        .def_readwrite("evidence_score", &sare::InductionResult::evidence_score)
        .def_readwrite("tests_run",      &sare::InductionResult::tests_run)
        .def_readwrite("tests_passed",   &sare::InductionResult::tests_passed)
        .def_readwrite("reasoning",      &sare::InductionResult::reasoning);

    // ─── CausalInduction ──
    py::class_<sare::CausalInduction>(m, "CausalInduction")
        .def(py::init<>())
        .def("evaluate", &sare::CausalInduction::evaluate,
             py::arg("rule"), py::arg("energy"),
             py::arg("num_tests") = sare::CausalInduction::DEFAULT_TESTS);

    // ─── ConceptRegistry ──
    py::class_<sare::ConceptRegistry>(m, "ConceptRegistry")
        .def(py::init<>())
        .def("add_rule",              &sare::ConceptRegistry::addRule)
        .def("get_rules",             &sare::ConceptRegistry::getRules)
        .def("get_consolidated_rules",&sare::ConceptRegistry::getConsolidatedRules);

    m.def("default_search_config", []() {
        return sare::SearchConfig{};
    });

    m.def("default_transform_names", []() {
        sare::TransformRegistry registry;
        sare::registerDefaultTransforms(registry);
        std::vector<std::string> names;
        for (sare::Transform* transform : registry.getAll()) {
            if (transform) {
                names.push_back(transform->name());
            }
        }
        return names;
    });

    m.def("run_beam_search", [](const sare::Graph& graph, const sare::SearchConfig& config) {
        auto energy = sare::makeDefaultEnergyAggregator();
        sare::TransformRegistry registry;
        sare::registerDefaultTransforms(registry);
        sare::BeamSearch search;
        return search.search(graph, energy, registry, config);
    }, py::arg("graph"), py::arg("config"));

    m.def("run_mcts_search", [](const sare::Graph& graph, const sare::SearchConfig& config) {
        auto energy = sare::makeDefaultEnergyAggregator();
        sare::TransformRegistry registry;
        sare::registerDefaultTransforms(registry);
        sare::MCTSSearch search;
        return search.search(graph, energy, registry, config);
    }, py::arg("graph"), py::arg("config"));
}
