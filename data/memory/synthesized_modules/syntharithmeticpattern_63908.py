# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-15 20:08:55
# Name: syntharithmeticpattern_63908
from sare.engine import Transform, Graph

class SynthArithmeticPattern_63908(Transform):
    def name(self):
        return "syntharithmeticpattern_63908"
    
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "/":
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    num_edge, den_edge = children[0], children[1]
                    num_node = graph.get_node(num_edge.target)
                    den_node = graph.get_node(den_edge.target)
                    if num_node and den_node and num_node.type == "constant" and den_node.type == "constant":
                        try:
                            num_val = int(num_node.label)
                            den_val = int(den_node.label)
                            if den_val != 0 and num_val % den_val == 0:
                                matches.append({"div": n.id, "num": num_node.id, "den": den_node.id})
                        except ValueError:
                            pass
            elif n.type == "operator" and n.label == "*":
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    left_edge, right_edge = children[0], children[1]
                    left_node = graph.get_node(left_edge.target)
                    right_node = graph.get_node(right_edge.target)
                    if left_node and right_node and left_node.type == "variable" and right_node.type == "variable":
                        if left_node.label == right_node.label:
                            matches.append({"mul": n.id, "var1": left_node.id, "var2": right_node.id})
        return matches
    
    def apply(self, graph, context):
        g = graph.clone()
        if "div" in context:
            div_id = context["div"]
            num_node = g.get_node(context["num"])
            den_node = g.get_node(context["den"])
            num_val = int(num_node.label)
            den_val = int(den_node.label)
            result = num_val // den_val
            new_const = g.add_node("constant", str(result), {})
            for e in list(g.edges):
                if e.target == div_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_const, e.relationship_type)
            g.remove_node(div_id)
            g.remove_node(num_node.id)
            g.remove_node(den_node.id)
            return g, -5.0
        elif "mul" in context:
            mul_id = context["mul"]
            var_node = g.get_node(context["var1"])
            new_pow = g.add_node("operator", "^", {})
            g.add_edge(new_pow, var_node.id, "arg1")
            exp_const = g.add_node("constant", "2", {})
            g.add_edge(new_pow, exp_const, "arg2")
            for e in list(g.edges):
                if e.target == mul_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_pow, e.relationship_type)
            g.remove_node(mul_id)
            g.remove_node(var_node.id)
            return g, -4.0
        return graph, 0.0