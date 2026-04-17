"""
LiveWorld — persistent interactive world the SARE system lives in and explores.

The system autonomously experiments: it picks untried object combinations,
applies real physics/chemistry/biology rules, and feeds discoveries into the
world model and commonsense KB as genuine causal knowledge.

Discoveries are emergent — rules are predicate functions over object properties,
so the system finds interactions it was never explicitly told about.

Usage (daemon):
    from sare.world.live_world import get_live_world
    lw = get_live_world()
    fact = lw.explore_step()
    if fact:
        lw.feed_to_memory(fact, world_model=wm, commonsense=cs)
    lw.save_state()
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "live_world_state.json"


# ── Object schema ──────────────────────────────────────────────────────────────
@dataclass
class WorldObject:
    name: str
    formula: str
    state: str              # solid | liquid | gas | plasma | radiation
    temperature: float      # Celsius (ambient = 20)
    melting_point: float    # Celsius
    boiling_point: float    # Celsius
    ph: Optional[float]     # None = not applicable
    density: float          # g/cm³
    conductivity: str       # conductor | insulator | semiconductor
    magnetic: bool
    flammable: bool
    organic: bool
    solubility: float       # g/100mL in water (0 = insoluble)
    oxidizer: bool
    reducer: bool
    acid: bool
    base: bool
    reactive: bool
    properties: Dict[str, Any] = field(default_factory=dict)


# ── Fact schema ────────────────────────────────────────────────────────────────
@dataclass
class WorldFact:
    subject: str
    predicate: str
    obj: str
    domain: str
    confidence: float
    explanation: str
    rule_name: str
    timestamp: float = field(default_factory=time.time)
    novel: bool = True

    def to_dict(self) -> dict:
        return {
            "subject": self.subject, "predicate": self.predicate,
            "obj": self.obj, "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation, "rule_name": self.rule_name,
            "timestamp": self.timestamp, "novel": self.novel,
        }

    @staticmethod
    def from_dict(d: dict) -> "WorldFact":
        return WorldFact(**{k: d[k] for k in WorldFact.__dataclass_fields__ if k in d})


# ── Rule schema ────────────────────────────────────────────────────────────────
@dataclass
class InteractionRule:
    name: str
    domain: str
    predicate: Callable[[WorldObject, WorldObject], bool]
    outcome: Callable[[WorldObject, WorldObject], WorldFact]
    confidence: float
    reversible: bool = False


# ── World state ────────────────────────────────────────────────────────────────
@dataclass
class WorldState:
    discovered_facts: List[WorldFact] = field(default_factory=list)
    tried_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    cycle_count: int = 0
    last_saved: float = 0.0


# ── Main class ─────────────────────────────────────────────────────────────────
class LiveWorld:
    """Persistent real-world simulation that SARE autonomously explores."""

    def __init__(self):
        self._objects: Dict[str, WorldObject] = {}
        self._rules: List[InteractionRule] = []
        self._state = WorldState()
        self._lock = threading.Lock()
        self._name_index: Dict[str, str] = {}   # lowercase → canonical
        self._init_objects()
        self._init_rules()
        self._load_state()
        log.info("[LiveWorld] Ready: %d objects, %d rules, %d prior discoveries",
                 len(self._objects), len(self._rules),
                 len(self._state.discovered_facts))

    # ── Object catalogue (~50 real-world objects) ──────────────────────────────
    def _init_objects(self):
        defs = [
            # Chemistry — pure substances
            ("water",           "H2O",        "liquid",    20, 0,    100,  7.0,  1.0,   "insulator",   False, False, False, 999, False, False, False, False, False, {}),
            ("ice",             "H2O",        "solid",    -10, 0,    100,  7.0,  0.92,  "insulator",   False, False, False,   0, False, False, False, False, False, {}),
            ("steam",           "H2O",        "gas",      110, 0,    100,  7.0,  0.001, "insulator",   False, False, False, 999, False, False, False, False, False, {}),
            ("oxygen",          "O2",         "gas",       20,-219,  -183, None, 0.001, "insulator",   False, False, False,   0, True,  False, False, False, True,  {"supports_combustion": True}),
            ("hydrogen",        "H2",         "gas",       20,-259,  -253, None, 0.0001,"insulator",   False, True,  False,   0, False, True,  False, False, True,  {"lightest_element": True}),
            ("carbon_dioxide",  "CO2",        "gas",       20,-78.5,  -78.5,None,0.002, "insulator",   False, False, False,   0, False, False, False, False, False, {"greenhouse_gas": True}),
            ("nitrogen",        "N2",         "gas",       20,-210,  -196, None, 0.001, "insulator",   False, False, False,   0, False, False, False, False, False, {"inert": True}),
            ("methane",         "CH4",        "gas",       20,-182,  -161, None, 0.0007,"insulator",   False, True,  True,    0, False, True,  False, False, True,  {}),
            ("ethanol",         "C2H5OH",     "liquid",    20,-114,   78,  None, 0.789, "insulator",   False, True,  True,    999, False, False, False, False, True, {"fuel": True}),
            ("salt",            "NaCl",       "solid",     20, 801,  1413, None, 2.16,  "insulator",   False, False, False, 360, False, False, False, False, False, {"electrolyte": True}),
            ("sugar",           "C12H22O11",  "solid",     20, 186,   186, None, 1.59,  "insulator",   False, False, True,  200, False, False, False, False, False, {}),
            ("salt_water",      "NaCl+H2O",   "liquid",    20,  -1,   101, 7.0,  1.04,  "conductor",   False, False, False, 999, False, False, False, False, False, {"electrolyte": True}),
            ("hydrochloric_acid","HCl",       "liquid",    20,-27,    110, 1.0,  1.18,  "conductor",   False, False, False, 999, False, False, True,  False, True,  {"corrosive": True}),
            ("sulfuric_acid",   "H2SO4",      "liquid",    20, 10,    337, 0.3,  1.84,  "conductor",   False, False, False, 999, True,  False, True,  False, True,  {"corrosive": True, "dehydrating": True}),
            ("sodium_hydroxide","NaOH",       "solid",     20, 318,  1388, 14.0, 2.13,  "insulator",   False, False, False, 111, False, False, False, True,  True,  {"corrosive": True}),
            ("calcium_carbonate","CaCO3",     "solid",     20,1339,  1339, None, 2.71,  "insulator",   False, False, False,  0.0013, False, False, False, False, False, {"in": "limestone,marble,chalk"}),
            ("rust",            "Fe2O3",      "solid",     20,1565,  2623, None, 5.25,  "insulator",   False, False, False,   0, False, False, False, False, False, {"formed_from": "iron+oxygen"}),
            ("ammonia",         "NH3",        "gas",       20,-78,   -33,  11.0, 0.0006,"insulator",   False, False, False, 900, False, True,  False, True,  True,  {"pungent": True}),
            ("chlorine",        "Cl2",        "gas",       20,-101,  -34,  None, 0.003, "insulator",   False, False, False,   0.7, True, False, False, False, True,  {"toxic": True}),
            ("ozone",           "O3",         "gas",       20,-193,  -112, None, 0.002, "insulator",   False, False, False,   0, True,  False, False, False, True,  {"uv_shield": True}),
            ("carbon",          "C",          "solid",     20,3550,  4027, None, 2.26,  "semiconductor",False, True, True,    0, False, True,  False, False, False, {"allotropes": "diamond,graphite,graphene"}),
            ("ash",             "C+minerals", "solid",     20,1000,  1000, None, 0.7,   "insulator",   False, False, False,   0, False, False, False, True,  False, {"from": "combustion"}),
            # Metals
            ("iron",            "Fe",         "solid",     20,1538,  2862, None, 7.87,  "conductor",   True,  False, False,   0, False, False, False, False, True,  {"ferromagnetic": True}),
            ("copper",          "Cu",         "solid",     20,1085,  2562, None, 8.96,  "conductor",   False, False, False,   0, False, False, False, False, True,  {"best_conductor": True}),
            ("aluminum",        "Al",         "solid",     20, 660,  2519, None, 2.7,   "conductor",   False, False, False,   0, False, True,  False, False, True,  {"lightweight": True}),
            ("gold",            "Au",         "solid",     20,1064,  2856, None,19.3,   "conductor",   False, False, False,   0, False, False, False, False, False, {"noble_metal": True, "non_reactive": True}),
            ("silver",          "Ag",         "solid",     20, 962,  2162, None,10.49,  "conductor",   False, False, False,   0, False, False, False, False, True,  {}),
            ("zinc",            "Zn",         "solid",     20, 420,   907, None, 7.13,  "conductor",   False, False, False,   0, False, True,  False, False, True,  {}),
            ("mercury",         "Hg",         "liquid",    20,-39,    357, None,13.53,  "conductor",   False, False, False,   0, False, False, False, False, True,  {"liquid_metal": True, "toxic": True}),
            # Physics / materials
            ("glass",           "SiO2",       "solid",     20,1723,  2230, None, 2.5,   "insulator",   False, False, False,   0, False, False, False, False, False, {"transparent": True, "brittle": True}),
            ("magnet",          "Fe3O4",      "solid",     20,1597,  2623, None, 5.17,  "conductor",   True,  False, False,   0, False, False, False, False, False, {"permanent_magnet": True}),
            ("rubber",          "C5H8",       "solid",     20,-73,    180, None, 1.2,   "insulator",   False, True,  True,    0, False, False, False, False, False, {"elastic": True}),
            ("plastic",         "C2H4",       "solid",     20, 130,   400, None, 0.95,  "insulator",   False, True,  True,    0, False, False, False, False, False, {}),
            ("wood",            "C6H10O5",    "solid",     20, 300,   300, None, 0.6,   "insulator",   False, True,  True,    0, False, False, False, False, False, {"cellulose": True}),
            ("coal",            "C",          "solid",     20,3550,  4027, None, 1.3,   "insulator",   False, True,  False,   0, False, True,  False, False, False, {"fossil_fuel": True}),
            ("diamond",         "C",          "solid",     20,3550,  4027, None, 3.51,  "insulator",   False, False, False,   0, False, False, False, False, False, {"hardest_natural": True, "transparent": True}),
            ("sand",            "SiO2",       "solid",     20,1723,  2230, None, 1.5,   "insulator",   False, False, False,   0, False, False, False, False, False, {}),
            ("limestone",       "CaCO3",      "solid",     20,1339,  1339, None, 2.71,  "insulator",   False, False, False, 0.0013, False, False, False, False, False, {}),
            # Biology
            ("seed",            "organic",    "solid",     20, 300,   300, None, 1.0,   "insulator",   False, False, True,    0, False, False, False, False, False, {"alive": True, "dormant": True}),
            ("soil",            "minerals",   "solid",     20, 200,   200, None, 1.3,   "insulator",   False, False, True,    0, False, False, False, False, False, {"nutrients": True}),
            ("plant",           "organic",    "solid",     20, 250,   250, None, 0.8,   "insulator",   False, False, True,    0, False, False, False, False, False, {"alive": True, "photosynthesis": True}),
            ("yeast",           "organic",    "solid",     20, 250,   250, None, 1.1,   "insulator",   False, False, True,    0, False, False, False, False, False, {"alive": True, "ferments": True}),
            ("enzyme",          "protein",    "solid",     20, 200,   200, None, 1.3,   "insulator",   False, False, True,    0, False, False, False, False, False, {"catalyst": True, "biological": True}),
            # Energy / misc
            ("fire",            "plasma",     "plasma",  900, -273,  -273, None, 0.001, "plasma",      False, False, False,   0, True,  False, False, False, True,  {"light": True, "heat": True}),
            ("sunlight",        "photons",    "radiation",20, -273,  -273, None, 0.0,   "radiation",   False, False, False,   0, False, False, False, False, True,  {"uv": True, "energy_source": True}),
            ("battery",         "electrochemical","solid", 20, 200,   200, None, 2.0,   "conductor",   False, False, False,   0, False, False, False, False, False, {"stores_energy": True, "voltage": True}),
            ("wire",            "Cu",         "solid",     20,1085,  2562, None, 8.96,  "conductor",   False, False, False,   0, False, False, False, False, False, {"conducts_electricity": True}),
            ("acid_base_reaction","H2O+salt", "liquid",    20,   0,   100, 7.0,  1.05,  "conductor",   False, False, False, 999, False, False, False, False, False, {}),
        ]
        for row in defs:
            (name, formula, state, temp, mp, bp, ph, dens, cond,
             mag, flam, org, sol, oxid, red, acid, base, react, props) = row
            obj = WorldObject(
                name=name, formula=formula, state=state, temperature=temp,
                melting_point=mp, boiling_point=bp, ph=ph, density=dens,
                conductivity=cond, magnetic=mag, flammable=flam, organic=org,
                solubility=sol, oxidizer=oxid, reducer=red, acid=acid,
                base=base, reactive=react, properties=props,
            )
            self._objects[name] = obj
            self._name_index[name.lower()] = name
            self._name_index[name.lower().replace("_", " ")] = name
            if formula:
                self._name_index[formula.lower()] = name

    # ── Rule catalogue (~60 real physics/chemistry/biology rules) ──────────────
    def _init_rules(self):
        O = self._objects

        def rule(name, domain, pred, out, conf=0.92, rev=False):
            self._rules.append(InteractionRule(name, domain, pred, out, conf, rev))

        def fact(s, p, o, dom, conf, expl, rname):
            return WorldFact(s, p, o, dom, conf, expl, rname)

        # ── Chemistry ─────────────────────────────────────────────────────────

        # Combustion: any flammable + fire
        rule("combustion", "chemistry",
             lambda a, b: (a.flammable and b.name == "fire") or (b.flammable and a.name == "fire"),
             lambda a, b: fact(
                 f"{(a if a.name != 'fire' else b).name}+fire", "produces",
                 "carbon_dioxide+water+heat",
                 "chemistry", 0.97,
                 f"{(a if a.name != 'fire' else b).name} burns in fire, producing CO₂, water, and heat",
                 "combustion"),
             0.97)

        # Dissolution: soluble solid + water
        rule("dissolution", "chemistry",
             lambda a, b: (a.solubility > 5 and b.formula == "H2O" and b.state == "liquid") or
                          (b.solubility > 5 and a.formula == "H2O" and a.state == "liquid"),
             lambda a, b: (
                 (lambda s=(a if b.formula == "H2O" else b):
                  fact(s.name, "dissolves_in", "water",
                       "chemistry", 0.93,
                       f"{s.name} (solubility={s.solubility}g/100mL) dissolves in water",
                       "dissolution"))()
             ), 0.93, True)

        # Acid + base neutralization
        rule("neutralization", "chemistry",
             lambda a, b: (a.acid and b.base) or (b.acid and a.base),
             lambda a, b: fact(
                 f"{(a if a.acid else b).name}+{(a if a.base else b).name}",
                 "reacts_to_form", "salt+water",
                 "chemistry", 0.96,
                 "Acid and base neutralize to form salt and water (neutralization reaction)",
                 "neutralization"),
             0.96)

        # Rusting: iron + oxygen or water
        rule("rusting", "chemistry",
             lambda a, b: (a.name == "iron" and b.name in ("oxygen", "water")) or
                          (b.name == "iron" and a.name in ("oxygen", "water")),
             lambda a, b: fact(
                 "iron", "oxidizes_to", "rust (Fe₂O₃)",
                 "chemistry", 0.94,
                 "Iron reacts with oxygen/water to form iron oxide (rust) over time",
                 "rusting"),
             0.94)

        # Water electrolysis: water + electricity (battery/wire)
        rule("electrolysis", "chemistry",
             lambda a, b: (a.formula == "H2O" and b.properties.get("voltage")) or
                          (b.formula == "H2O" and a.properties.get("voltage")),
             lambda a, b: fact(
                 "water+electricity", "splits_into", "hydrogen+oxygen",
                 "chemistry", 0.95,
                 "Electrolysis splits water (H₂O) into hydrogen gas (H₂) and oxygen gas (O₂)",
                 "electrolysis"),
             0.95)

        # Hydrogen combustion
        rule("hydrogen_combustion", "chemistry",
             lambda a, b: (a.name == "hydrogen" and b.name == "fire") or
                          (b.name == "hydrogen" and a.name == "fire"),
             lambda a, b: fact(
                 "hydrogen+fire", "produces", "water+explosion",
                 "chemistry", 0.97,
                 "Hydrogen is highly flammable — ignition produces water and energy release",
                 "hydrogen_combustion"),
             0.97)

        # Acid on metal (corrosion): acid + any metal
        rule("acid_corrosion", "chemistry",
             lambda a, b: (a.acid and b.conductivity == "conductor" and b.reactive) or
                          (b.acid and a.conductivity == "conductor" and a.reactive),
             lambda a, b: (
                 (lambda acid=(a if a.acid else b), metal=(a if not a.acid else b):
                  fact(f"{acid.name}+{metal.name}", "produces",
                       f"{metal.name}_salt+hydrogen_gas",
                       "chemistry", 0.91,
                       f"{acid.name} corrodes {metal.name}, releasing hydrogen gas",
                       "acid_corrosion"))()
             ), 0.91)

        # CO2 + water → carbonic acid (weak)
        rule("carbonic_acid_formation", "chemistry",
             lambda a, b: (a.name == "carbon_dioxide" and b.formula == "H2O") or
                          (b.name == "carbon_dioxide" and a.formula == "H2O"),
             lambda a, b: fact(
                 "CO2+water", "forms", "carbonic_acid (H₂CO₃)",
                 "chemistry", 0.90,
                 "Carbon dioxide dissolves in water to form carbonic acid — this acidifies oceans",
                 "carbonic_acid_formation"),
             0.90)

        # Salt + water → electrolyte solution (conductivity increase)
        rule("electrolyte", "chemistry",
             lambda a, b: (a.properties.get("electrolyte") and b.formula == "H2O") or
                          (b.properties.get("electrolyte") and a.formula == "H2O"),
             lambda a, b: fact(
                 (a if a.properties.get("electrolyte") else b).name + "+water",
                 "creates", "electrically_conductive_solution",
                 "chemistry", 0.92,
                 "Electrolytes dissolved in water create ions that conduct electricity",
                 "electrolyte"),
             0.92)

        # Calcium carbonate + acid → CO2 (limestone dissolves in acid)
        rule("limestone_acid", "chemistry",
             lambda a, b: (a.name in ("calcium_carbonate", "limestone") and b.acid) or
                          (b.name in ("calcium_carbonate", "limestone") and a.acid),
             lambda a, b: fact(
                 "limestone+acid", "produces", "CO2+water+calcium_salt",
                 "chemistry", 0.93,
                 "Limestone (CaCO₃) reacts with acid to produce CO₂ gas, water, and a calcium salt",
                 "limestone_acid"),
             0.93)

        # Chlorine + water (bleaching/disinfection)
        rule("chlorination", "chemistry",
             lambda a, b: (a.name == "chlorine" and b.formula == "H2O") or
                          (b.name == "chlorine" and a.formula == "H2O"),
             lambda a, b: fact(
                 "chlorine+water", "forms", "hypochlorous_acid+HCl",
                 "chemistry", 0.88,
                 "Chlorine dissolves in water to form hypochlorous acid (used in disinfection)",
                 "chlorination"),
             0.88)

        # Ammonia + acid → ammonium salt
        rule("ammonium_formation", "chemistry",
             lambda a, b: (a.name == "ammonia" and b.acid) or (b.name == "ammonia" and a.acid),
             lambda a, b: fact(
                 "ammonia+acid", "forms", "ammonium_salt",
                 "chemistry", 0.91,
                 "Ammonia (a base) reacts with acids to form ammonium salts",
                 "ammonium_formation"),
             0.91)

        # Sand + extreme heat → glass formation
        rule("glass_formation", "chemistry",
             lambda a, b: (a.name == "sand" and b.temperature > 1700) or
                          (b.name == "sand" and a.temperature > 1700),
             lambda a, b: fact(
                 "sand+extreme_heat", "forms", "glass (SiO₂)",
                 "chemistry", 0.87,
                 "Sand (silicon dioxide, SiO₂) melts at ~1700°C and forms glass when cooled",
                 "glass_formation"),
             0.87)

        # Methane + oxygen combustion
        rule("methane_combustion", "chemistry",
             lambda a, b: (a.name == "methane" and b.name in ("fire", "oxygen")) or
                          (b.name == "methane" and a.name in ("fire", "oxygen")),
             lambda a, b: fact(
                 "methane+oxygen", "combustion_produces", "CO2+H2O+energy",
                 "chemistry", 0.96,
                 "Methane (natural gas) burns in oxygen: CH₄ + 2O₂ → CO₂ + 2H₂O + energy",
                 "methane_combustion"),
             0.96)

        # Saponification: oil/fat + base → soap
        rule("saponification", "chemistry",
             lambda a, b: (a.organic and a.properties.get("fat") and b.base) or
                          (b.organic and b.properties.get("fat") and a.base),
             lambda a, b: fact(
                 "fat+base", "saponification_produces", "soap+glycerol",
                 "chemistry", 0.90,
                 "Saponification: fat + strong base (NaOH) → soap + glycerol",
                 "saponification"),
             0.90)

        # Fermentation: yeast + sugar → alcohol + CO2
        rule("fermentation", "biology",
             lambda a, b: (a.name == "yeast" and b.name in ("sugar", "glucose")) or
                          (b.name == "yeast" and a.name in ("sugar", "glucose")),
             lambda a, b: fact(
                 "yeast+sugar", "ferments_to", "ethanol+CO2",
                 "biology", 0.95,
                 "Yeast ferments sugar (C₆H₁₂O₆) to produce ethanol and carbon dioxide",
                 "fermentation"),
             0.95)

        # Enzyme catalysis: enzyme + any organic substrate
        rule("enzyme_catalysis", "biology",
             lambda a, b: (a.name == "enzyme" and b.organic and b.name not in ("plant", "seed", "yeast")) or
                          (b.name == "enzyme" and a.organic and a.name not in ("plant", "seed", "yeast")),
             lambda a, b: (
                 (lambda substrate=(a if a.name != "enzyme" else b):
                  fact("enzyme", "catalyzes_breakdown_of", substrate.name,
                       "biology", 0.88,
                       f"Enzymes are biological catalysts that speed up reactions — breaks down {substrate.name}",
                       "enzyme_catalysis"))()
             ), 0.88)

        # ── Physics ────────────────────────────────────────────────────────────

        # Phase transition: heating a solid past melting point
        rule("melting", "physics",
             lambda a, b: (a.state == "solid" and b.temperature > a.melting_point and b.state != "solid") or
                          (b.state == "solid" and a.temperature > b.melting_point and a.state != "solid"),
             lambda a, b: (
                 (lambda s=(a if a.state == "solid" else b):
                  fact(s.name, "melts_above", f"{s.melting_point}°C",
                       "physics", 0.98,
                       f"{s.name} melts above {s.melting_point}°C (melting point)",
                       "melting"))()
             ), 0.98)

        # Phase transition: cooling liquid below freezing point
        rule("freezing", "physics",
             lambda a, b: (a.state == "liquid" and b.temperature < a.melting_point and b.temperature < 10) or
                          (b.state == "liquid" and a.temperature < b.melting_point and a.temperature < 10),
             lambda a, b: (
                 (lambda liq=(a if a.state == "liquid" else b):
                  fact(liq.name, "freezes_below", f"{liq.melting_point}°C",
                       "physics", 0.98,
                       f"{liq.name} freezes below {liq.melting_point}°C (freezing point)",
                       "freezing"))()
             ), 0.98, True)

        # Phase transition: heating liquid past boiling point
        rule("boiling", "physics",
             lambda a, b: (a.state == "liquid" and b.temperature > a.boiling_point) or
                          (b.state == "liquid" and a.temperature > b.boiling_point),
             lambda a, b: (
                 (lambda liq=(a if a.state == "liquid" else b):
                  fact(liq.name, "boils_at", f"{liq.boiling_point}°C",
                       "physics", 0.98,
                       f"{liq.name} boils at {liq.boiling_point}°C — becomes gas",
                       "boiling"))()
             ), 0.98)

        # Ice + fire → melting
        rule("ice_fire", "physics",
             lambda a, b: (a.name == "ice" and b.name == "fire") or
                          (b.name == "ice" and a.name == "fire"),
             lambda a, b: fact(
                 "fire", "melts", "ice",
                 "physics", 0.99,
                 "Fire heats ice past 0°C, melting it to liquid water",
                 "ice_fire"),
             0.99)

        # Magnetism: magnet attracts iron/nickel/cobalt
        rule("magnetic_attraction", "physics",
             lambda a, b: (a.magnetic and b.properties.get("ferromagnetic")) or
                          (b.magnetic and a.properties.get("ferromagnetic")),
             lambda a, b: (
                 (lambda mag=(a if a.magnetic else b), fm=(a if a.properties.get("ferromagnetic") else b):
                  fact(mag.name, "attracts", fm.name,
                       "physics", 0.97,
                       f"{mag.name} attracts {fm.name} due to ferromagnetism",
                       "magnetic_attraction"))()
             ), 0.97)

        # Magnet near non-magnetic conductor: no attraction, but induces current (Faraday)
        rule("electromagnetic_induction", "physics",
             lambda a, b: (a.magnetic and b.conductivity == "conductor" and not b.magnetic) or
                          (b.magnetic and a.conductivity == "conductor" and not a.magnetic),
             lambda a, b: (
                 (lambda mag=(a if a.magnetic else b), cond=(a if not a.magnetic else b):
                  fact(f"moving_{mag.name}", "induces_current_in", cond.name,
                       "physics", 0.93,
                       f"A moving magnet near {cond.name} (conductor) induces electric current (Faraday's law)",
                       "electromagnetic_induction"))()
             ), 0.93)

        # Electrical conduction: battery + conductor
        rule("electrical_circuit", "physics",
             lambda a, b: (a.properties.get("voltage") and b.conductivity == "conductor") or
                          (b.properties.get("voltage") and a.conductivity == "conductor"),
             lambda a, b: fact(
                 "battery+conductor", "creates", "electric_current",
                 "physics", 0.95,
                 "A battery connected to a conductor (wire) drives electric current flow",
                 "electrical_circuit"),
             0.95)

        # Heat conduction: fire + metal conductor
        rule("heat_conduction", "physics",
             lambda a, b: (a.name == "fire" and b.conductivity == "conductor") or
                          (b.name == "fire" and a.conductivity == "conductor"),
             lambda a, b: (
                 (lambda metal=(a if a.conductivity == "conductor" else b):
                  fact(metal.name, "conducts_heat_from", "fire",
                       "physics", 0.94,
                       f"{metal.name} is a conductor — heat from fire travels through it rapidly",
                       "heat_conduction"))()
             ), 0.94)

        # Insulation: fire + insulator
        rule("thermal_insulation", "physics",
             lambda a, b: (a.name == "fire" and b.conductivity == "insulator" and b.state == "solid") or
                          (b.name == "fire" and a.conductivity == "insulator" and a.state == "solid"),
             lambda a, b: (
                 (lambda ins=(a if a.conductivity == "insulator" and a.state == "solid" else b):
                  fact(ins.name, "insulates_against", "heat",
                       "physics", 0.89,
                       f"{ins.name} is an insulator — resists heat transfer",
                       "thermal_insulation"))()
             ), 0.89)

        # Light through glass (transparency)
        rule("transparency", "physics",
             lambda a, b: (a.name == "sunlight" and b.properties.get("transparent")) or
                          (b.name == "sunlight" and a.properties.get("transparent")),
             lambda a, b: (
                 (lambda mat=(a if a.properties.get("transparent") else b):
                  fact(mat.name, "is_transparent_to", "light",
                       "physics", 0.96,
                       f"{mat.name} allows light to pass through — it is transparent",
                       "transparency"))()
             ), 0.96)

        # Diamond hardness
        rule("diamond_properties", "physics",
             lambda a, b: a.name == "diamond" or b.name == "diamond",
             lambda a, b: fact(
                 "diamond", "is", "hardest_natural_material (10 on Mohs scale)",
                 "physics", 0.99,
                 "Diamond is the hardest natural substance — 10 on Mohs hardness scale; pure carbon",
                 "diamond_properties"),
             0.99)

        # Water as universal solvent
        rule("universal_solvent", "physics",
             lambda a, b: (a.formula == "H2O" and b.solubility > 0.5 and b.state == "solid") or
                          (b.formula == "H2O" and a.solubility > 0.5 and a.state == "solid"),
             lambda a, b: (
                 (lambda solute=(a if a.formula != "H2O" else b):
                  fact("water", "is_universal_solvent_for", solute.name,
                       "chemistry", 0.88,
                       f"Water dissolves {solute.name} — it is often called the universal solvent",
                       "universal_solvent"))()
             ), 0.88)

        # ── Biology ────────────────────────────────────────────────────────────

        # Photosynthesis: plant + sunlight + CO2
        rule("photosynthesis_light", "biology",
             lambda a, b: (a.name == "plant" and b.name == "sunlight") or
                          (b.name == "plant" and a.name == "sunlight"),
             lambda a, b: fact(
                 "plant+sunlight+CO2", "produces", "glucose+oxygen",
                 "biology", 0.97,
                 "Photosynthesis: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂ (glucose + oxygen)",
                 "photosynthesis_light"),
             0.97)

        rule("photosynthesis_co2", "biology",
             lambda a, b: (a.name == "plant" and b.name == "carbon_dioxide") or
                          (b.name == "plant" and a.name == "carbon_dioxide"),
             lambda a, b: fact(
                 "plant", "absorbs", "carbon_dioxide (used in photosynthesis)",
                 "biology", 0.96,
                 "Plants absorb CO₂ from the air and convert it to glucose during photosynthesis",
                 "photosynthesis_co2"),
             0.96)

        # Seed + water + soil → germination
        rule("germination", "biology",
             lambda a, b: (a.name == "seed" and b.name in ("water", "soil")) or
                          (b.name == "seed" and a.name in ("water", "soil")),
             lambda a, b: fact(
                 "seed+water+soil", "germinates_into", "plant",
                 "biology", 0.93,
                 "Seeds germinate when given water and soil — they grow into plants",
                 "germination"),
             0.93)

        # Oxygen + organic matter → respiration (reverse of photosynthesis)
        rule("respiration", "biology",
             lambda a, b: (a.name == "oxygen" and b.organic and b.properties.get("alive")) or
                          (b.name == "oxygen" and a.organic and a.properties.get("alive")),
             lambda a, b: (
                 (lambda org=(a if a.organic else b):
                  fact(f"{org.name}+oxygen", "cellular_respiration_produces", "CO2+water+energy",
                       "biology", 0.95,
                       f"Cellular respiration: {org.name} uses O₂ to produce CO₂, water, and ATP energy",
                       "respiration"))()
             ), 0.95)

        # Decomposition: organic matter + soil (microbes/fungi)
        rule("decomposition", "biology",
             lambda a, b: (a.organic and not a.properties.get("alive") and b.name == "soil") or
                          (b.organic and not b.properties.get("alive") and a.name == "soil"),
             lambda a, b: (
                 (lambda org=(a if a.organic and not a.properties.get("alive") else b):
                  fact(org.name + "+soil_microbes", "decomposes_to", "compost+minerals",
                       "biology", 0.90,
                       f"Dead organic matter ({org.name}) decomposes via soil microbes into compost",
                       "decomposition"))()
             ), 0.90)

        # UV/sunlight damages organic material
        rule("uv_damage", "biology",
             lambda a, b: (a.name == "sunlight" and b.organic) or
                          (b.name == "sunlight" and a.organic),
             lambda a, b: (
                 (lambda org=(a if a.organic else b):
                  fact("sunlight_UV", "can_damage", org.name + " DNA/proteins",
                       "biology", 0.85,
                       f"UV radiation in sunlight can damage DNA and proteins in {org.name}",
                       "uv_damage"))()
             ), 0.85)

        # ── Emergent cross-domain rules ────────────────────────────────────────

        # Ozone shields UV
        rule("ozone_uv_shield", "physics",
             lambda a, b: (a.name == "ozone" and b.name == "sunlight") or
                          (b.name == "ozone" and a.name == "sunlight"),
             lambda a, b: fact(
                 "ozone_layer", "blocks", "UV_radiation_from_sunlight",
                 "physics", 0.97,
                 "Ozone (O₃) in the stratosphere absorbs harmful UV radiation — it shields life on Earth",
                 "ozone_uv_shield"),
             0.97)

        # Mercury toxicity + water (bioaccumulation)
        rule("mercury_contamination", "chemistry",
             lambda a, b: (a.name == "mercury" and b.formula == "H2O") or
                          (b.name == "mercury" and a.formula == "H2O"),
             lambda a, b: fact(
                 "mercury", "contaminates", "water (toxic, bioaccumulates in food chain)",
                 "chemistry", 0.94,
                 "Mercury is a toxic heavy metal that contaminates water and bioaccumulates in organisms",
                 "mercury_contamination"),
             0.94)

        # Noble metal: gold + acid (no reaction)
        rule("gold_inertness", "chemistry",
             lambda a, b: (a.name == "gold" and b.acid) or (b.name == "gold" and a.acid),
             lambda a, b: fact(
                 "gold", "does_not_react_with", "most_acids",
                 "chemistry", 0.96,
                 "Gold is a noble metal — it does not react with most acids (resistant to corrosion)",
                 "gold_inertness"),
             0.96)

        # Carbon allotropes: carbon + pressure → diamond
        rule("diamond_from_carbon", "physics",
             lambda a, b: (a.name == "carbon" and b.properties.get("extreme_pressure")) or
                          (b.name == "carbon" and a.properties.get("extreme_pressure")),
             lambda a, b: fact(
                 "carbon+extreme_pressure", "forms", "diamond",
                 "physics", 0.93,
                 "Under extreme pressure and heat, carbon atoms rearrange into diamond crystal structure",
                 "diamond_from_carbon"),
             0.93)

        # Rubber + vulcanization (sulfur) → stronger rubber
        rule("vulcanization", "chemistry",
             lambda a, b: (a.name == "rubber" and b.name == "sulfuric_acid") or
                          (b.name == "rubber" and a.name == "sulfuric_acid"),
             lambda a, b: fact(
                 "rubber+sulfur_treatment", "vulcanization_produces", "stronger_durable_rubber",
                 "chemistry", 0.86,
                 "Vulcanization treats rubber with sulfur to create cross-links, making it stronger and more elastic",
                 "vulcanization"),
             0.86)

        # Acid rain: CO2/SO2 + water in atmosphere
        rule("acid_rain", "chemistry",
             lambda a, b: (a.name == "carbon_dioxide" and b.name == "rain") or
                          (b.name == "carbon_dioxide" and a.name == "rain") or
                          (a.name == "carbon_dioxide" and b.formula == "H2O" and "cloud" in str(b.properties)) or
                          (b.name == "carbon_dioxide" and a.formula == "H2O" and "cloud" in str(a.properties)),
             lambda a, b: fact(
                 "CO2+atmospheric_water", "forms", "carbonic_acid (acid_rain)",
                 "chemistry", 0.88,
                 "Carbon dioxide dissolved in atmospheric water forms carbonic acid — the basis of acid rain",
                 "acid_rain"),
             0.88)

        # Ozone depletion: chlorine + ozone
        rule("ozone_depletion", "chemistry",
             lambda a, b: (a.name == "chlorine" and b.name == "ozone") or
                          (b.name == "chlorine" and a.name == "ozone"),
             lambda a, b: fact(
                 "chlorine", "destroys", "ozone (O₃ → O₂)",
                 "chemistry", 0.93,
                 "Chlorine atoms (from CFCs) catalytically destroy ozone molecules in the stratosphere",
                 "ozone_depletion"),
             0.93)

        log.debug("[LiveWorld] Initialized %d rules", len(self._rules))

    # ── State persistence ──────────────────────────────────────────────────────
    def _load_state(self):
        try:
            if _STATE_PATH.exists():
                raw = _STATE_PATH.read_text("utf-8")
                data = json.loads(raw)
                facts = [WorldFact.from_dict(f) for f in data.get("discovered_facts", [])]
                pairs = {frozenset(p) for p in data.get("tried_pairs", [])}
                self._state = WorldState(
                    discovered_facts=facts,
                    tried_pairs=pairs,
                    cycle_count=data.get("cycle_count", 0),
                    last_saved=data.get("last_saved", 0.0),
                )
                log.debug("[LiveWorld] Loaded state: %d facts, %d pairs tried",
                          len(facts), len(pairs))
        except Exception as e:
            log.debug("[LiveWorld] State load error (starting fresh): %s", e)
            self._state = WorldState()

    def save_state(self):
        with self._lock:
            try:
                _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "discovered_facts": [f.to_dict() for f in self._state.discovered_facts[-500:]],
                    "tried_pairs": [list(p) for p in self._state.tried_pairs],
                    "cycle_count": self._state.cycle_count,
                    "last_saved": time.time(),
                }
                tmp = _STATE_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2), "utf-8")
                os.replace(tmp, _STATE_PATH)
            except Exception as e:
                log.debug("[LiveWorld] Save error: %s", e)

    # ── Exploration ────────────────────────────────────────────────────────────
    def explore_step(self) -> List["WorldFact"]:
        """Pick one untried pair, apply all rules, return new facts found."""
        with self._lock:
            pair = self._pick_untried_pair()
            if pair is None:
                # All pairs exhausted — reset to re-explore (discoveries preserved)
                self._state.tried_pairs = set()
                pair = self._pick_untried_pair()
                if pair is None:
                    return []

            a_name, b_name = pair
            self._state.tried_pairs.add(frozenset({a_name, b_name}))
            self._state.cycle_count += 1

        a = self._objects[a_name]
        b = self._objects[b_name]
        new_facts = self._apply_rules(a, b)

        if new_facts:
            with self._lock:
                # Mark as not novel if already discovered
                known = {(f.subject, f.predicate, f.obj) for f in self._state.discovered_facts}
                for f in new_facts:
                    f.novel = (f.subject, f.predicate, f.obj) not in known
                self._state.discovered_facts.extend(new_facts)

        return new_facts

    def _pick_untried_pair(self) -> Optional[Tuple[str, str]]:
        names = list(self._objects.keys())
        all_pairs = [
            (a, b) for i, a in enumerate(names)
            for b in names[i+1:]
            if frozenset({a, b}) not in self._state.tried_pairs
        ]
        if not all_pairs:
            return None

        # Bias: objects with fewer known facts explored first
        fact_counts: Dict[str, int] = {}
        for f in self._state.discovered_facts:
            for token in (f.subject, f.obj):
                for name in names:
                    if name in token:
                        fact_counts[name] = fact_counts.get(name, 0) + 1

        # Weight: objects with fewer facts get 3× selection chance
        weights = []
        for a, b in all_pairs:
            ca = fact_counts.get(a, 0)
            cb = fact_counts.get(b, 0)
            w = (3 if ca < 2 else 1) * (3 if cb < 2 else 1)
            weights.append(w)

        total = sum(weights)
        r = random.random() * total
        running = 0.0
        for (a, b), w in zip(all_pairs, weights):
            running += w
            if r <= running:
                return a, b
        return all_pairs[-1]

    def _apply_rules(self, a: WorldObject, b: WorldObject) -> List[WorldFact]:
        facts = []
        for rule in self._rules:
            try:
                if rule.predicate(a, b):
                    f = rule.outcome(a, b)
                    if f:
                        f.timestamp = time.time()
                        facts.append(f)
            except Exception as e:
                log.debug("[LiveWorld] Rule %s error on (%s,%s): %s",
                          rule.name, a.name, b.name, e)
        return facts

    # ── Hypothesis testing ─────────────────────────────────────────────────────
    def hypothesize_and_test(self, world_model=None) -> List[WorldFact]:
        """
        Use existing world model beliefs to form and test novel hypotheses.
        E.g., if world model says 'iron is magnetic', try iron with all known objects.
        """
        novel_facts = []
        if world_model is None:
            return novel_facts

        try:
            # Get some existing beliefs to form hypotheses
            beliefs = []
            if hasattr(world_model, "search_beliefs"):
                beliefs = world_model.search_beliefs(domain="chemistry") or []
                beliefs += world_model.search_beliefs(domain="physics") or []
            elif hasattr(world_model, "_beliefs"):
                beliefs = list(world_model._beliefs.values())[:30]

            known = {(f.subject, f.predicate, f.obj) for f in self._state.discovered_facts}
            tested = set()

            for belief in beliefs[:20]:
                # Try to find object names in the belief
                subj = str(getattr(belief, "subject", "") or "").lower()
                obj_val = str(getattr(belief, "value", "") or "").lower()

                # Map belief subjects to world objects
                for s_name, s_obj in self._objects.items():
                    if s_name not in subj and s_name not in obj_val:
                        continue
                    for t_name, t_obj in self._objects.items():
                        if t_name == s_name:
                            continue
                        pair_key = frozenset({s_name, t_name})
                        if pair_key in tested:
                            continue
                        tested.add(pair_key)

                        new = self._apply_rules(s_obj, t_obj)
                        for f in new:
                            if (f.subject, f.predicate, f.obj) not in known:
                                f.novel = True
                                novel_facts.append(f)
                                known.add((f.subject, f.predicate, f.obj))

        except Exception as e:
            log.debug("[LiveWorld] hypothesize_and_test error: %s", e)

        return novel_facts[:10]  # cap to 10 per call

    # ── Memory integration ─────────────────────────────────────────────────────
    def feed_to_memory(self, fact: WorldFact, world_model=None, commonsense=None) -> None:
        """Inject a discovery into world model and commonsense KB."""
        try:
            if world_model and hasattr(world_model, "update_belief"):
                world_model.update_belief(
                    fact.subject, fact.predicate, fact.obj,
                    confidence=fact.confidence, domain=fact.domain,
                )
        except Exception:
            pass
        try:
            if commonsense and hasattr(commonsense, "add_fact"):
                commonsense.add_fact(fact.subject, fact.predicate, fact.obj)
                # Also add as a human-readable Q→A
                q = f"What happens when {fact.subject.replace('+', ' and ')}?"
                commonsense.add_fact(q, "AnswerTo", fact.explanation[:200])
        except Exception:
            pass

    # ── Status ─────────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        n_objects = len(self._objects)
        n_pairs = n_objects * (n_objects - 1) // 2
        tried = len(self._state.tried_pairs)
        return {
            "objects": n_objects,
            "rules": len(self._rules),
            "possible_pairs": n_pairs,
            "tried_pairs": tried,
            "untried_remaining": max(0, n_pairs - tried),
            "discoveries": len(self._state.discovered_facts),
            "novel_discoveries": sum(1 for f in self._state.discovered_facts if f.novel),
            "cycle_count": self._state.cycle_count,
            "domain_breakdown": self._domain_breakdown(),
        }

    def _domain_breakdown(self) -> dict:
        counts: Dict[str, int] = {}
        for f in self._state.discovered_facts:
            counts[f.domain] = counts.get(f.domain, 0) + 1
        return counts

    def get_recent(self, n: int = 20) -> List[dict]:
        return [f.to_dict() for f in self._state.discovered_facts[-n:]]


# ── Singleton ──────────────────────────────────────────────────────────────────
_lw_instance: Optional[LiveWorld] = None
_lw_lock = threading.Lock()


def get_live_world() -> LiveWorld:
    global _lw_instance
    if _lw_instance is None:
        with _lw_lock:
            if _lw_instance is None:
                _lw_instance = LiveWorld()
    return _lw_instance
