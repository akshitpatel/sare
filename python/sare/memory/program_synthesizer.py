import logging
import ast
import traceback
from typing import Optional, Dict, Any, Tuple
from sare.interface.llm_bridge import _call_llm as _call_llm_base

def _call_llm(prompt: str) -> str:
    """Use the synthesis model for code generation (higher quality)."""
    return _call_llm_base(prompt, use_synthesis_model=True)
from sare.engine import Graph

log = logging.getLogger(__name__)

class SandboxSimulator:
    """Safely executes dynamically generated Python transforms."""
    
    @staticmethod
    def is_safe(code_string: str) -> bool:
        """Basic static analysis to prevent gross security issues."""
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("os", "sys", "subprocess", "socket"):
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ("os", "sys", "subprocess", "socket"):
                        return False
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ("eval", "exec", "open", "__import__"):
                            return False
            return True
        except Exception:
            return False

    @staticmethod
    def execute_transform(code_string: str, graph: Graph) -> Tuple[bool, Optional[Graph], str]:
        """
        Executes a generated transform function safely.
        The LLM should generate a function called `apply_synthetic_transform(graph: dict) -> dict`
        We serialize the graph, pass it in, and deserialize the result.
        """
        if not SandboxSimulator.is_safe(code_string):
            return False, None, "Code failed safety checks."
            
        import copy
        local_scope = {}
        try:
            # We enforce that the LLM generates a function that manipulates the graph dictionary 
            # to avoid dealing with object references directly inside the exec environment.
            # Provide safe standard library modules that the LLM might use
            exec(code_string, {'copy': copy}, local_scope)
            if 'apply_synthetic_transform' not in local_scope:
                return False, None, "Missing apply_synthetic_transform function."
                
            transform_fn = local_scope['apply_synthetic_transform']
            
            # Serialize
            graph_dict = graph.to_dict()
            
            # Execute
            new_graph_dict = transform_fn(graph_dict)
            
            # Deserialize
            if not isinstance(new_graph_dict, dict) or 'nodes' not in new_graph_dict:
                return False, None, "Function did not return a valid graph dictionary."
                
            new_graph = Graph.from_dict(new_graph_dict)
            return True, new_graph, "Success"
            
        except Exception as e:
            err = traceback.format_exc()
            return False, None, f"Execution failed: {e}\n{err}"


class ProgramSynthesizer:
    """
    Tier 7 / Pillar 1 Engine.
    Observes local minima in graph search and hallucinates new Python transforms 
    to break out of them.
    """
    def __init__(self, registry: Any):
        self.registry = registry
        
    def generate_transform(self, graph: Graph, goal_description: str) -> Tuple[bool, str, Optional[Graph]]:
        """
        Asks the LLM to write a transform for the current graph to achieve the goal.
        Returns (success, generated_code, new_graph)
        """
        prompt = f"""
You are the autonomic nervous system of SARE-HX, a neuro-symbolic graph reasoning engine.
The reasoning engine is stuck in a local minimum and needs a new primitive operation to proceed.

CURRENT GRAPH (JSON):
{graph.to_dict()}

GOAL:
{goal_description}

TASK:
Write exactly one Python function named `apply_synthetic_transform(graph: dict) -> dict`.
This function receives a dictionary representing the current graph (with "nodes" and "edges" lists) and must return a heavily mutated copy of that dictionary representing the graph AFTER your transformation.

Rules:
1. Only return the raw Python code.
2. Do not use Markdown backticks (```python) in your response, just the raw text of the code.
3. You must manipulate the "nodes" and "edges" lists directly.
4. Nodes follow: {{"id": int, "type": str, "label": str, "attributes": dict}}
5. Edges follow: {{"id": int, "source": int, "target": int, "type": str}}
6. Ensure any new nodes have unique integer IDs (use max(node id) + 1).

Example Logic for Identity Elimination (x + 0 -> x):
def apply_synthetic_transform(graph):
    # find + and 0, rewire parent to point to x, delete + and 0
    return mutated_graph
"""
        log.info("Requesting synthetic transform from LLM...")
        code = _call_llm(prompt)
        log.debug(f"RAW LLM RESPONSE: {repr(code)}")
        
        # Clean up markdown if LLM disobeyed
        import re
        match = re.search(r'```(?:python)?\n(.*?)\n```', code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = code.strip()
            if code.startswith("python\n"):
                code = code[7:].strip()
                
        log.debug(f"Received synthetic transform code:\n{code}")
        
        # Sandbox Execution
        success, new_graph, msg = SandboxSimulator.execute_transform(code, graph)
        if success:
            log.info("Synthetic transform executed successfully.")
            return True, code, new_graph
        else:
            log.warning(f"Synthetic transform failed: {msg}")
            return False, code, None
