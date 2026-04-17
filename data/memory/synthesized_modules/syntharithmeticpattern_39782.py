# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 17:14:01
# Name: syntharithmeticpattern_39782
from sare.engine import Transform, Graph

class SynthArithmeticPattern_39782(Transform):
    def name(self):
        return "syntharithmeticpattern_39782"

    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "-", "*", "/"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if not left or not right:
                    continue
                if left.type == "constant" and right.type == "constant":
                    try:
                        left_val = float(left.label)
                        right_val = float(right.label)
                        if n.label == "+":
                            result = left_val + right_val
                        elif n.label == "-":
                            result = left_val - right_val
                        elif n.label == "*":
                            result = left_val * right_val
                        elif n.label == "/":
                            if right_val == 0:
                                continue
                            result = left_val / right_val
                        else:
                            continue
                        if result.is_integer():
                            result = int(result)
                        matches.append({
                            "op": n.id,
                            "left": left.id,
                            "right": right.id,
                            "result": str(result)
                        })
                    except ValueError:
                        continue
                elif n.label == "*" and left.type == "variable" and right.type == "variable":
                    if left.label == right.label:
                        matches.append({
                            "op": n.id,
                            "var": left.id,
                            "other": right.id
                        })
        return matches

    def apply(self, graph, context):
        g = graph.clone()
        if "result" in context:
            op_id = context["op"]
            left_id = context["left"]
            right_id = context["right"]
            result = context["result"]
            new_const = g.add_node("constant", result, {})
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_const, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(left_id)
            g.remove_node(right_id)
            return g, -5.0
        else:
            op_id = context["op"]
            var_id = context["var"]
            other_id = context["other"]
            new_pow = g.add_node("operator", "^", {})
            g.add_edge(new_pow, var_id, "base")
            exp_node = g.add_node("constant", "2", {})
            g.add_edge(new_pow, exp_node, "exponent")
            for e in list(g.edges):
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, new_pow, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(other_id)
            return g, -2.0