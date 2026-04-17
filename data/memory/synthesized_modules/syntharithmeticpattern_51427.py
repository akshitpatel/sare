# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-15 16:40:53
# Name: syntharithmeticpattern_51427
from sare.engine import Transform, Graph

class SynthArithmeticPattern_51427(Transform):
    def name(self):
        return "syntharithmeticpattern_51427"

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
                            if num_val % den_val == 0:
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
                            matches.append({"mul": n.id, "var1": left.id, "var2": right.id})
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        if "div" in context:
            div_id = context["div"]
            num_id = context["num"]
            den_id = context["den"]
            num_val = int(graph.get_node(num_id).label)
            den_val = int(graph.get_node(den_id).label)
            result_val = num_val // den_val
            result_node_id = g.add_node("constant", str(result_val), {})
            for e in list(g.edges):
                if e.target == div_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, result_node_id, e.relationship_type)
            g.remove_node(div_id)
            g.remove_node(num_id)
            g.remove_node(den_id)
            return g, -5.0
        elif "mul" in context:
            mul_id = context["mul"]
            var_id = context["var1"]
            var_node = graph.get_node(var_id)
            exp_node_id = g.add_node("operator", "^", {})
            const_node_id = g.add_node("constant", "2", {})
            g.add_edge(exp_node_id, var_id, "arg")
            g.add_edge(exp_node_id, const_node_id, "arg")
            for e in list(g.edges):
                if e.target == mul_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, exp_node_id, e.relationship_type)
            g.remove_node(mul_id)
            return g, -2.0
        return graph, 0.0