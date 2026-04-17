# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-15 16:36:25
# Name: syntharithmeticpattern_51124
from sare.engine import Transform, Graph

from sare.engine import Transform, Graph

class SynthArithmeticPattern_51124(Transform):
    def name(self):
        return "syntharithmeticpattern_51124"
    
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "/":
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    num_edge, den_edge = children[0], children[1]
                    num = graph.get_node(num_edge.target)
                    den = graph.get_node(den_edge.target)
                    if num and den and num.type == "constant" and den.type == "constant":
                        try:
                            num_val = int(num.label)
                            den_val = int(den.label)
                            if den_val != 0 and num_val % den_val == 0:
                                matches.append({"div": n.id, "num": num.id, "den": den.id})
                        except ValueError:
                            pass
            elif n.type == "operator" and n.label == "*":
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    left_edge, right_edge = children[0], children[1]
                    left = graph.get_node(left_edge.target)
                    right = graph.get_node(right_edge.target)
                    if left and right and left.type == "variable" and right.type == "variable":
                        if left.label == right.label:
                            matches.append({"mul": n.id, "var": left.id})
        return matches
    
    def apply(self, graph, context):
        g = graph.clone()
        if "div" in context:
            div_id = context["div"]
            num_id = context["num"]
            den_id = context["den"]
            num_node = g.get_node(num_id)
            den_node = g.get_node(den_id)
            num_val = int(num_node.label)
            den_val = int(den_node.label)
            result_val = num_val // den_val
            new_const = g.add_node("constant", str(result_val), {})
            for e in list(g.edges):
                if e.target == div_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_const, e.relationship_type)
            g.remove_node(div_id)
            g.remove_node(num_id)
            g.remove_node(den_id)
            return g, -5.0
        elif "mul" in context:
            mul_id = context["mul"]
            var_id = context["var"]
            var_node = g.get_node(var_id)
            new_pow = g.add_node("operator", "^", {})
            g.add_edge(new_pow, var_id, "arg")
            exp_const = g.add_node("constant", "2", {})
            g.add_edge(new_pow, exp_const, "arg")
            for e in list(g.edges):
                if e.target == mul_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_pow, e.relationship_type)
            g.remove_node(mul_id)
            return g, -2.0
        return g, 0.0