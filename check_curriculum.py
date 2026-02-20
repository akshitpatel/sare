import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

# Ensure we can import sare
sys.path.append(os.path.abspath("python"))

try:
    from sare import Graph, Node, Edge
    from sare.curiosity.curriculum_generator import CurriculumGenerator
except ImportError as e:
    logging.error(f"Failed to import SARE components: {e}")
    sys.exit(1)

def test_generation():
    if not Graph:
        logging.error("Graph binding is None")
        return

    logging.info("Initializing CurriculumGenerator...")
    gen = CurriculumGenerator()
    
    # Create seed: x + 1
    g = Graph()
    
    # x
    x = g.add_node("variable")
    g.get_node(x).set_attribute("label", "x")
    
    # +
    op = g.add_node("operator") 
    g.get_node(op).set_attribute("label", "+")
    
    # 1
    one = g.add_node("constant")
    g.get_node(one).set_attribute("value", "1.0")
    g.get_node(one).set_attribute("label", "1.0")
    
    # Edges
    g.add_edge(op, x, "left_operand")
    g.add_edge(op, one, "right_operand")
    
    gen.add_seed(g)
    logging.info("Seed problem added: x + 1.0")
    
    new_probs = gen.generate_batch(size=5)
    logging.info(f"Generated {len(new_probs)} variants")
    
    for i, p in enumerate(new_probs):
        nodes_str = []
        for nid in p.get_node_ids():
            n = p.get_node(nid)
            label = n.get_attribute("label")
            nodes_str.append(f"{n.type}({label})")
        logging.info(f"Variant {i}: {', '.join(nodes_str)}")

if __name__ == "__main__":
    test_generation()
