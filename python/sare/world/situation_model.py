"""
SituationModel — physical commonsense via rule-based simulation.

Instead of looking up static triples, SituationModel represents objects
with properties and applies causal rules to answer "what happens if..."
questions dynamically.

This bridges the gap between the 1.9% commonsense lookup rate and
human-level physical intuition (~90%).
"""
from __future__ import annotations

import re
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any

log = logging.getLogger(__name__)


# ── Object property templates ──────────────────────────────────────────────────
# (object, property, value)
_OBJECT_PROPERTIES: List[Tuple[str, str, str]] = [
    # States of matter
    ("ice",      "state",    "solid"),
    ("water",    "state",    "liquid"),
    ("steam",    "state",    "gas"),
    ("butter",   "state",    "solid"),
    ("wax",      "state",    "solid"),
    ("glass",    "state",    "solid"),
    ("iron",     "state",    "solid"),
    ("gold",     "state",    "solid"),
    ("mercury",  "state",    "liquid"),
    ("alcohol",  "state",    "liquid"),
    ("oxygen",   "state",    "gas"),
    ("nitrogen", "state",    "gas"),
    ("lava",     "state",    "liquid"),
    ("snow",     "state",    "solid"),

    # Thermal properties (melting_point in qualitative terms)
    ("ice",      "melts_at",   "0C"),
    ("butter",   "melts_at",   "32C"),
    ("wax",      "melts_at",   "60C"),
    ("iron",     "melts_at",   "1538C"),
    ("gold",     "melts_at",   "1064C"),
    ("glass",    "melts_at",   "1400C"),

    # Conductivity
    ("metal",    "conducts",   "electricity"),
    ("iron",     "conducts",   "electricity"),
    ("copper",   "conducts",   "electricity"),
    ("gold",     "conducts",   "electricity"),
    ("rubber",   "insulates",  "electricity"),
    ("wood",     "insulates",  "electricity"),
    ("plastic",  "insulates",  "electricity"),
    ("glass",    "insulates",  "electricity"),

    # Density / sinking
    ("wood",     "density",    "low"),
    ("cork",     "density",    "low"),
    ("oil",      "density",    "low"),
    ("ice",      "density",    "low"),
    ("iron",     "density",    "high"),
    ("gold",     "density",    "high"),
    ("rock",     "density",    "high"),
    ("lead",     "density",    "high"),

    # Flammability
    ("wood",     "flammable",  "yes"),
    ("paper",    "flammable",  "yes"),
    ("alcohol",  "flammable",  "yes"),
    ("gasoline", "flammable",  "yes"),
    ("water",    "flammable",  "no"),
    ("iron",     "flammable",  "no"),
    ("glass",    "flammable",  "no"),

    # Transparency
    ("glass",    "transparent", "yes"),
    ("water",    "transparent", "yes"),
    ("air",      "transparent", "yes"),
    ("wood",     "transparent", "no"),
    ("metal",    "transparent", "no"),
    ("iron",     "transparent", "no"),

    # Magnetic
    ("iron",     "magnetic",   "yes"),
    ("steel",    "magnetic",   "yes"),
    ("nickel",   "magnetic",   "yes"),
    ("wood",     "magnetic",   "no"),
    ("gold",     "magnetic",   "no"),
    ("copper",   "magnetic",   "no"),
    ("plastic",  "magnetic",   "no"),

    # Life
    ("plant",    "needs",      "sunlight"),
    ("plant",    "needs",      "water"),
    ("animal",   "needs",      "food"),
    ("human",    "needs",      "oxygen"),
    ("fish",     "needs",      "water"),
    ("seed",     "grows_into", "plant"),
]

# ── Causal rules ───────────────────────────────────────────────────────────────
# (condition_fn, conclusion_template)
# condition_fn receives (obj: str, props: dict, context: dict) → bool
# conclusion_template is a string with {obj} placeholders

_RULES: List[Dict[str, Any]] = [
    # Heat + solid → melts
    {
        "id": "heat_melts_solid",
        "trigger_words": ["heat", "warm", "hot", "fire", "melt", "sun", "temperature"],
        "object_prop": ("state", "solid"),
        "conclusion": "{obj} melts",
        "answer": "melts",
    },
    # Cold + liquid → freezes
    {
        "id": "cold_freezes_liquid",
        "trigger_words": ["cold", "freeze", "freezing", "ice", "frost", "cool"],
        "object_prop": ("state", "liquid"),
        "conclusion": "{obj} freezes",
        "answer": "freezes",
    },
    # Heat + liquid → evaporates/boils
    {
        "id": "heat_boils_liquid",
        "trigger_words": ["boil", "heat", "hot", "steam", "evaporate"],
        "object_prop": ("state", "liquid"),
        "conclusion": "{obj} evaporates",
        "answer": "evaporates",
    },
    # Drop heavy object → falls / sinks
    {
        "id": "dense_sinks",
        "trigger_words": ["drop", "sink", "water", "float", "throw in water"],
        "object_prop": ("density", "high"),
        "conclusion": "{obj} sinks",
        "answer": "sinks",
    },
    # Low density → floats
    {
        "id": "light_floats",
        "trigger_words": ["drop", "float", "water", "throw in water"],
        "object_prop": ("density", "low"),
        "conclusion": "{obj} floats",
        "answer": "floats",
    },
    # Flammable + fire → burns
    {
        "id": "flammable_burns",
        "trigger_words": ["fire", "flame", "burn", "light", "ignite"],
        "object_prop": ("flammable", "yes"),
        "conclusion": "{obj} burns",
        "answer": "burns",
    },
    # Not flammable + fire → no burning
    {
        "id": "not_flammable",
        "trigger_words": ["fire", "flame", "burn", "light", "ignite"],
        "object_prop": ("flammable", "no"),
        "conclusion": "{obj} does not burn",
        "answer": "does not burn",
    },
    # Conductor + electricity → conducts
    {
        "id": "conducts_electricity",
        "trigger_words": ["electricity", "electric", "current", "circuit", "wire"],
        "object_prop": ("conducts", "electricity"),
        "conclusion": "{obj} conducts electricity",
        "answer": "conducts electricity",
    },
    # Insulator + electricity → insulates
    {
        "id": "insulates_electricity",
        "trigger_words": ["electricity", "electric", "current", "circuit"],
        "object_prop": ("insulates", "electricity"),
        "conclusion": "{obj} does not conduct electricity",
        "answer": "does not conduct electricity",
    },
    # Magnet + magnetic material → attracts
    {
        "id": "magnet_attracts",
        "trigger_words": ["magnet", "magnetic", "attract"],
        "object_prop": ("magnetic", "yes"),
        "conclusion": "magnet attracts {obj}",
        "answer": "attracted",
    },
    # Magnet + non-magnetic → no attraction
    {
        "id": "magnet_no_attract",
        "trigger_words": ["magnet", "magnetic", "attract"],
        "object_prop": ("magnetic", "no"),
        "conclusion": "magnet does not attract {obj}",
        "answer": "not attracted",
    },
]

# ── Yes/No question templates ──────────────────────────────────────────────────
_YES_NO_RULES: List[Dict[str, Any]] = [
    # "does X melt in/on/near Y" where Y is hot → yes if solid, no if liquid/gas
    {
        "pattern": r"(?:does|will|would|can)\s+(\w[\w\s]{1,20}?)\s+melt",
        "object_group": 1,
        "check_prop": ("state", "solid"),
        "yes_answer": "yes",
        "no_answer": "no",
    },
    # "does X float" → yes if low density
    {
        "pattern": r"(?:does|will|would|can)\s+(\w[\w\s]{1,20}?)\s+float",
        "object_group": 1,
        "check_prop": ("density", "low"),
        "yes_answer": "yes",
        "no_answer": "no",
    },
    # "does X sink" → yes if high density
    {
        "pattern": r"(?:does|will|would|can)\s+(\w[\w\s]{1,20}?)\s+sink",
        "object_group": 1,
        "check_prop": ("density", "high"),
        "yes_answer": "yes",
        "no_answer": "no",
    },
    # "does X burn / catch fire" → yes if flammable
    {
        "pattern": r"(?:does|will|would|can)\s+(\w[\w\s]{1,20}?)\s+(?:burn|catch\s+fire|ignite)",
        "object_group": 1,
        "check_prop": ("flammable", "yes"),
        "yes_answer": "yes",
        "no_answer": "no",
    },
    # "does X conduct electricity" → yes if conducts
    {
        "pattern": r"(?:does|will|would|can)\s+(\w[\w\s]{1,20}?)\s+conduct",
        "object_group": 1,
        "check_prop": ("conducts", "electricity"),
        "yes_answer": "yes",
        "no_answer": "no",
    },
    # "does X conduct electricity" → no if insulates
    {
        "pattern": r"(?:does|will|would|can)\s+(\w[\w\s]{1,20}?)\s+conduct",
        "object_group": 1,
        "check_prop": ("insulates", "electricity"),
        "yes_answer": "no",  # insulator → does NOT conduct
        "no_answer": "yes",  # non-insulator → may conduct
    },
    # "is X transparent / can you see through X"
    {
        "pattern": r"(?:is\s+(?:a\s+|an\s+|the\s+)?(\w[\w\s]{1,20}?)\s+transparent|can\s+you\s+see\s+through\s+(?:a\s+)?(\w[\w\s]{1,20}?))",
        "object_group": 1,
        "check_prop": ("transparent", "yes"),
        "yes_answer": "yes",
        "no_answer": "no",
    },
    # "is X magnetic / does magnet attract X"
    {
        "pattern": r"(?:is\s+(?:a\s+|an\s+|the\s+)?(\w[\w\s]{1,20}?)\s+magnetic|(?:does|will)\s+(?:a\s+)?magnet\s+attract\s+(\w[\w\s]{1,20}?))",
        "object_group": 1,
        "check_prop": ("magnetic", "yes"),
        "yes_answer": "yes",
        "no_answer": "no",
    },
]


class SituationModel:
    """
    Rule-based physical situation model.

    Instead of KB lookup, applies rules dynamically to answer
    "what happens to X when Y?" and "does X do Z?" questions.
    """

    def __init__(self):
        # Build property index: object → {property: value}
        self._props: Dict[str, Dict[str, str]] = {}
        for obj, prop, val in _OBJECT_PROPERTIES:
            self._props.setdefault(obj.lower(), {})[prop] = val

    def get_properties(self, obj: str) -> Dict[str, str]:
        return self._props.get(obj.lower(), {})

    def answer_what_happens(self, question: str) -> Optional[str]:
        """
        Answer "what happens to X when Y?" or "what happens if Y to X?" questions.

        Returns a short string answer or None if no rule applies.
        """
        q = question.lower().strip()

        # Extract object: "what happens to <OBJ> when/if ..."
        obj_m = re.search(
            r"what\s+(?:happens?|would\s+happen)\s+(?:to\s+)?(?:a\s+|an\s+|the\s+)?(\w[\w\s]{0,20}?)"
            r"\s+(?:when|if|after|in|on|near|with)",
            q, re.IGNORECASE,
        )
        if not obj_m:
            # "if you heat X, what happens?"
            obj_m = re.search(
                r"(?:heat|cool|freeze|drop|put)\s+(?:a\s+|an\s+|the\s+)?(\w[\w\s]{0,20}?)"
                r"(?:\s+(?:in|on|near|into))?\s*[,\.]",
                q, re.IGNORECASE,
            )
        if not obj_m:
            return None

        obj = obj_m.group(1).strip().lower().rstrip(",.")
        obj_props = self.get_properties(obj)
        if not obj_props:
            return None

        # Try each rule: check if question contains trigger words and object has required property
        for rule in _RULES:
            triggers = rule["trigger_words"]
            req_prop, req_val = rule["object_prop"]
            if any(tw in q for tw in triggers) and obj_props.get(req_prop) == req_val:
                return rule["answer"].replace("{obj}", obj)

        return None

    def answer_yes_no(self, question: str) -> Optional[str]:
        """Answer yes/no physical questions: 'does X melt?', 'will X float?'"""
        q = question.lower().strip()

        for rule in _YES_NO_RULES:
            m = re.search(rule["pattern"], q, re.IGNORECASE)
            if m:
                # Extract object from correct group
                obj = None
                for g in range(1, m.lastindex + 1 if m.lastindex else 2):
                    try:
                        obj = m.group(g)
                        if obj:
                            break
                    except IndexError:
                        pass
                if not obj:
                    continue
                obj = obj.strip().lower()
                obj_props = self.get_properties(obj)
                if not obj_props:
                    continue
                check_prop, check_val = rule["check_prop"]
                if obj_props.get(check_prop) == check_val:
                    return rule["yes_answer"]
                elif check_prop in obj_props:
                    return rule["no_answer"]

        return None

    def answer(self, question: str) -> Optional[str]:
        """Try all situation model strategies on the question."""
        ans = self.answer_yes_no(question)
        if ans:
            return ans
        ans = self.answer_what_happens(question)
        if ans:
            return ans
        return None

    def generate_training_problems(self, n: int = 500) -> List[Dict[str, Any]]:
        """
        Generate training QA pairs from the situation model rules.

        Returns list of {"question": ..., "answer": ..., "domain": "commonsense"}.
        These are provably correct (derived from rules) and diverse.
        """
        problems = []

        # What-happens problems
        what_templates = [
            ("what happens to {obj} when heated?", "heat_melts_solid", "melts"),
            ("what happens to {obj} when cooled?", "cold_freezes_liquid", "freezes"),
            ("what happens if you heat {obj}?", "heat_melts_solid", "melts"),
            ("if you heat {obj}, what happens?", "heat_melts_solid", "melts"),
            ("what happens to {obj} in fire?", "flammable_burns", "burns"),
            ("what happens when {obj} is dropped in water?", "dense_sinks", "sinks"),
            ("what happens when {obj} is dropped in water?", "light_floats", "floats"),
        ]

        # Yes/no problems
        yn_templates = [
            ("does {obj} melt?", ("state", "solid"), "yes", ("state", "liquid"), "no"),
            ("does {obj} float in water?", ("density", "low"), "yes", ("density", "high"), "no"),
            ("does {obj} sink in water?", ("density", "high"), "yes", ("density", "low"), "no"),
            ("can {obj} burn?", ("flammable", "yes"), "yes", ("flammable", "no"), "no"),
            ("does {obj} conduct electricity?", ("conducts", "electricity"), "yes", ("insulates", "electricity"), "no"),
            ("is {obj} magnetic?", ("magnetic", "yes"), "yes", ("magnetic", "no"), "no"),
            ("is {obj} transparent?", ("transparent", "yes"), "yes", ("transparent", "no"), "no"),
        ]

        import random

        for obj, obj_props in self._props.items():
            # What-happens
            for tmpl, rule_id, answer in what_templates:
                rule = next((r for r in _RULES if r["id"] == rule_id), None)
                if rule is None:
                    continue
                req_prop, req_val = rule["object_prop"]
                if obj_props.get(req_prop) == req_val:
                    problems.append({
                        "question": tmpl.replace("{obj}", obj),
                        "answer": answer,
                        "domain": "commonsense",
                    })

            # Yes/no
            for tmpl, yes_prop, yes_ans, no_prop, no_ans in yn_templates:
                if obj_props.get(yes_prop[0]) == yes_prop[1]:
                    problems.append({
                        "question": tmpl.replace("{obj}", obj),
                        "answer": yes_ans,
                        "domain": "commonsense",
                    })
                elif obj_props.get(no_prop[0]) == no_prop[1]:
                    problems.append({
                        "question": tmpl.replace("{obj}", obj),
                        "answer": no_ans,
                        "domain": "commonsense",
                    })

        random.shuffle(problems)
        return problems[:n]


# ── Singleton ──────────────────────────────────────────────────────────────────

_SM_INSTANCE: Optional[SituationModel] = None
_SM_LOCK = threading.Lock()


def get_situation_model() -> SituationModel:
    global _SM_INSTANCE
    if _SM_INSTANCE is None:
        with _SM_LOCK:
            if _SM_INSTANCE is None:
                _SM_INSTANCE = SituationModel()
    return _SM_INSTANCE
