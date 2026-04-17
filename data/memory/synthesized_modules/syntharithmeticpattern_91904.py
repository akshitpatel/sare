# Auto-synthesized Transform — domain: arithmetic
# Generated: 2026-04-16 05:21:30
# Name: syntharithmeticpattern_91904
from sare.engine import Transform, Graph

from sare.engine import Transform, Graph

class SynthArithmeticPattern_91904(Transform):
    def name(self):
        return "syntharithmeticpattern_91904"
    
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
                            num_val = float(num.label)
                            den_val = float(den.label)
                            if den_val != 0 and num_val % den_val == 0:
                                matches.append({
                                    "div_node": n.id,
                                    "num": num.id,
                                    "den": den.id,
                                    "value": num_val / den_val
                                })
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
                            matches.append({
                                "mul_node": n.id,
                                "var_left": left.id,
                                "var_right": right.id,
                                "var_label": left.label
                            })
        return matches
    
    def apply(self, graph, context):
        g = graph.clone()
        if "div_node" in context:
            div_id = context["div_node"]
            num_id = context["num"]
            den_id = context["den"]
            value = context["value"]
            
            const_id = g.add_node("constant", str(int(value) if value.is_integer() else value), {})
            
            for e in list(g.edges):
                if e.target == div_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, const_id, e.relationship_type)
            
            g.remove_node(div_id)
            g.remove_node(num_id)
            g.remove_node(den_id)
            return g, -2.0
        
        elif "mul_node" in context:
            mul_id = context["mul_node"]
            var_label = context["var_label"]
            
            pow_id = g.add_node("operator", "^", {})
            var_id = g.add_node("variable", var_label, {})
            exp_id = g.add_node("constant", "2", {})
            
            g.add_edge(pow_id, var_id, "arg0")
            g.add_edge(pow_id, exp_id, "arg1")
            
            for e in list(g.edges):
                if e.target == mul_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, pow_id, e.relationship_type)
            
            g.remove_node(mul_id)
            g.remove_node(context["var_left"])
            g.remove_node(context["var_right"])
            return g, -1.0
        
        return g, 0.0