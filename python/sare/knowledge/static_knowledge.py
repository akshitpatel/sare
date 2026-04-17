"""
StaticKnowledgeLoader — Hardcoded structured facts for SARE-HX.

Loads 400+ curated facts across math, physics, chemistry, logic/CS, and
commonsense domains into the WorldModel and CommonSenseBase WITHOUT any LLM
calls.  Safe to call multiple times; uses a sentinel fact to skip re-loading.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any

log = logging.getLogger(__name__)

# Sentinel used to detect whether load_all() already ran in this WorldModel.
_SENTINEL_FACT = "__static_knowledge_loaded__"
_SENTINEL_DOMAIN = "meta"


# ── Fact collections ──────────────────────────────────────────────────────────
# Each entry: (domain: str, fact_str: str, confidence: float)

def _math_facts() -> List[Tuple[str, str, float]]:
    """150+ math facts covering algebra, trig, calculus, number theory, sets,
    combinatorics, and probability."""
    facts: List[Tuple[str, str, float]] = [
        # ── Algebra ──────────────────────────────────────────────────────────
        ("algebra", "(a+b)^2 = a^2 + 2*a*b + b^2", 1.0),
        ("algebra", "(a-b)^2 = a^2 - 2*a*b + b^2", 1.0),
        ("algebra", "(a+b)*(a-b) = a^2 - b^2", 1.0),
        ("algebra", "(a+b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3", 1.0),
        ("algebra", "(a-b)^3 = a^3 - 3*a^2*b + 3*a*b^2 - b^3", 1.0),
        ("algebra", "a^3 + b^3 = (a+b)*(a^2 - a*b + b^2)", 1.0),
        ("algebra", "a^3 - b^3 = (a-b)*(a^2 + a*b + b^2)", 1.0),
        ("algebra", "quadratic formula: x = (-b ± sqrt(b^2 - 4*a*c)) / (2*a)", 1.0),
        ("algebra", "discriminant: b^2 - 4*a*c; >0 two real roots, =0 one root, <0 complex roots", 1.0),
        ("algebra", "factor theorem: (x - r) divides p(x) iff p(r) = 0", 1.0),
        ("algebra", "remainder theorem: p(a) is the remainder when p(x) is divided by (x-a)", 1.0),
        ("algebra", "FOIL: (a+b)*(c+d) = a*c + a*d + b*c + b*d", 1.0),
        ("algebra", "sum of arithmetic series: S_n = n*(a1 + an)/2", 1.0),
        ("algebra", "nth term of arithmetic sequence: a_n = a1 + (n-1)*d", 1.0),
        ("algebra", "sum of geometric series: S_n = a1*(1 - r^n)/(1 - r) for r != 1", 1.0),
        ("algebra", "infinite geometric series sum: a/(1-r) for |r| < 1", 1.0),
        ("algebra", "nth term of geometric sequence: a_n = a1 * r^(n-1)", 1.0),
        ("algebra", "x^0 = 1 for any nonzero x", 1.0),
        ("algebra", "x^1 = x", 1.0),
        ("algebra", "x^(-n) = 1/x^n", 1.0),
        ("algebra", "x^(m/n) = nth root of x^m", 1.0),
        ("algebra", "x^a * x^b = x^(a+b)", 1.0),
        ("algebra", "x^a / x^b = x^(a-b)", 1.0),
        ("algebra", "(x^a)^b = x^(a*b)", 1.0),
        ("algebra", "log(a*b) = log(a) + log(b)", 1.0),
        ("algebra", "log(a/b) = log(a) - log(b)", 1.0),
        ("algebra", "log(a^n) = n*log(a)", 1.0),
        ("algebra", "log_b(b) = 1", 1.0),
        ("algebra", "log_b(1) = 0", 1.0),
        ("algebra", "change of base: log_b(x) = ln(x)/ln(b)", 1.0),
        ("algebra", "ln(e) = 1", 1.0),
        ("algebra", "e^0 = 1", 1.0),
        ("algebra", "e^(ln x) = x for x > 0", 1.0),
        ("algebra", "ln(e^x) = x", 1.0),
        ("algebra", "additive identity: a + 0 = a", 1.0),
        ("algebra", "multiplicative identity: a * 1 = a", 1.0),
        ("algebra", "additive inverse: a + (-a) = 0", 1.0),
        ("algebra", "multiplicative inverse: a * (1/a) = 1 for a != 0", 1.0),
        ("algebra", "distributive law: a*(b+c) = a*b + a*c", 1.0),
        ("algebra", "commutative addition: a + b = b + a", 1.0),
        ("algebra", "commutative multiplication: a*b = b*a", 1.0),
        ("algebra", "associative addition: (a+b)+c = a+(b+c)", 1.0),
        ("algebra", "associative multiplication: (a*b)*c = a*(b*c)", 1.0),
        ("algebra", "a*0 = 0", 1.0),
        ("algebra", "(-1)*a = -a", 1.0),
        ("algebra", "(-a)*(-b) = a*b", 1.0),
        ("algebra", "AM-GM: (a+b)/2 >= sqrt(a*b) for a,b >= 0", 1.0),
        ("algebra", "Cauchy-Schwarz: (a1^2+a2^2)*(b1^2+b2^2) >= (a1*b1+a2*b2)^2", 0.95),
        # ── Trigonometry ─────────────────────────────────────────────────────
        ("trigonometry", "sin^2(x) + cos^2(x) = 1", 1.0),
        ("trigonometry", "1 + tan^2(x) = sec^2(x)", 1.0),
        ("trigonometry", "1 + cot^2(x) = csc^2(x)", 1.0),
        ("trigonometry", "sin(0) = 0", 1.0),
        ("trigonometry", "cos(0) = 1", 1.0),
        ("trigonometry", "tan(0) = 0", 1.0),
        ("trigonometry", "sin(pi/6) = 1/2", 1.0),
        ("trigonometry", "cos(pi/6) = sqrt(3)/2", 1.0),
        ("trigonometry", "sin(pi/4) = sqrt(2)/2", 1.0),
        ("trigonometry", "cos(pi/4) = sqrt(2)/2", 1.0),
        ("trigonometry", "sin(pi/3) = sqrt(3)/2", 1.0),
        ("trigonometry", "cos(pi/3) = 1/2", 1.0),
        ("trigonometry", "sin(pi/2) = 1", 1.0),
        ("trigonometry", "cos(pi/2) = 0", 1.0),
        ("trigonometry", "sin(pi) = 0", 1.0),
        ("trigonometry", "cos(pi) = -1", 1.0),
        ("trigonometry", "sin(3*pi/2) = -1", 1.0),
        ("trigonometry", "cos(3*pi/2) = 0", 1.0),
        ("trigonometry", "sin(2*pi) = 0", 1.0),
        ("trigonometry", "cos(2*pi) = 1", 1.0),
        ("trigonometry", "tan(x) = sin(x)/cos(x)", 1.0),
        ("trigonometry", "double angle: sin(2x) = 2*sin(x)*cos(x)", 1.0),
        ("trigonometry", "double angle: cos(2x) = cos^2(x) - sin^2(x)", 1.0),
        ("trigonometry", "double angle: cos(2x) = 2*cos^2(x) - 1", 1.0),
        ("trigonometry", "double angle: cos(2x) = 1 - 2*sin^2(x)", 1.0),
        ("trigonometry", "double angle: tan(2x) = 2*tan(x)/(1 - tan^2(x))", 1.0),
        ("trigonometry", "sum: sin(A+B) = sin(A)*cos(B) + cos(A)*sin(B)", 1.0),
        ("trigonometry", "sum: sin(A-B) = sin(A)*cos(B) - cos(A)*sin(B)", 1.0),
        ("trigonometry", "sum: cos(A+B) = cos(A)*cos(B) - sin(A)*sin(B)", 1.0),
        ("trigonometry", "sum: cos(A-B) = cos(A)*cos(B) + sin(A)*sin(B)", 1.0),
        ("trigonometry", "sum: tan(A+B) = (tan(A)+tan(B))/(1-tan(A)*tan(B))", 1.0),
        ("trigonometry", "product-to-sum: sin(A)*sin(B) = (cos(A-B) - cos(A+B))/2", 1.0),
        ("trigonometry", "product-to-sum: cos(A)*cos(B) = (cos(A-B) + cos(A+B))/2", 1.0),
        ("trigonometry", "sin(-x) = -sin(x)  (odd function)", 1.0),
        ("trigonometry", "cos(-x) = cos(x)   (even function)", 1.0),
        ("trigonometry", "law of sines: a/sin(A) = b/sin(B) = c/sin(C)", 1.0),
        ("trigonometry", "law of cosines: c^2 = a^2 + b^2 - 2*a*b*cos(C)", 1.0),
        ("trigonometry", "Euler's formula: e^(i*x) = cos(x) + i*sin(x)", 1.0),
        # ── Calculus ─────────────────────────────────────────────────────────
        ("calculus", "d/dx(x^n) = n*x^(n-1)", 1.0),
        ("calculus", "d/dx(e^x) = e^x", 1.0),
        ("calculus", "d/dx(ln x) = 1/x  for x > 0", 1.0),
        ("calculus", "d/dx(sin x) = cos x", 1.0),
        ("calculus", "d/dx(cos x) = -sin x", 1.0),
        ("calculus", "d/dx(tan x) = sec^2(x)", 1.0),
        ("calculus", "d/dx(a^x) = a^x * ln(a)", 1.0),
        ("calculus", "d/dx(constant) = 0", 1.0),
        ("calculus", "chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)", 1.0),
        ("calculus", "product rule: d/dx[u*v] = u'*v + u*v'", 1.0),
        ("calculus", "quotient rule: d/dx[u/v] = (u'*v - u*v') / v^2", 1.0),
        ("calculus", "integral: ∫x^n dx = x^(n+1)/(n+1) + C  for n != -1", 1.0),
        ("calculus", "integral: ∫e^x dx = e^x + C", 1.0),
        ("calculus", "integral: ∫1/x dx = ln|x| + C", 1.0),
        ("calculus", "integral: ∫sin(x) dx = -cos(x) + C", 1.0),
        ("calculus", "integral: ∫cos(x) dx = sin(x) + C", 1.0),
        ("calculus", "integral: ∫a dx = a*x + C", 1.0),
        ("calculus", "fundamental theorem: d/dx[∫_a^x f(t)dt] = f(x)", 1.0),
        ("calculus", "linearity: ∫[f(x)+g(x)]dx = ∫f(x)dx + ∫g(x)dx", 1.0),
        ("calculus", "L'Hopital: lim f(x)/g(x) = lim f'(x)/g'(x) when 0/0 or ∞/∞", 1.0),
        ("calculus", "limit: lim_(x→0) sin(x)/x = 1", 1.0),
        # ── Number theory ─────────────────────────────────────────────────────
        ("number_theory", "gcd(a,b) * lcm(a,b) = a * b", 1.0),
        ("number_theory", "Fermat's little theorem: a^p ≡ a (mod p) for prime p", 1.0),
        ("number_theory", "Euler's theorem: a^φ(n) ≡ 1 (mod n) if gcd(a,n)=1", 1.0),
        ("number_theory", "every integer n>1 has a unique prime factorization", 1.0),
        ("number_theory", "there are infinitely many primes", 1.0),
        ("number_theory", "a | b and b | c implies a | c", 1.0),
        ("number_theory", "if p is prime and p | a*b, then p | a or p | b", 1.0),
        ("number_theory", "Wilson's theorem: (p-1)! ≡ -1 (mod p) for prime p", 0.95),
        # ── Sets ─────────────────────────────────────────────────────────────
        ("sets", "A ∪ ∅ = A", 1.0),
        ("sets", "A ∩ A = A  (idempotent)", 1.0),
        ("sets", "A ∩ ∅ = ∅", 1.0),
        ("sets", "A ∪ A = A  (idempotent)", 1.0),
        ("sets", "|A ∪ B| = |A| + |B| - |A ∩ B|  (inclusion-exclusion)", 1.0),
        ("sets", "De Morgan: ¬(A ∪ B) = ¬A ∩ ¬B", 1.0),
        ("sets", "De Morgan: ¬(A ∩ B) = ¬A ∪ ¬B", 1.0),
        ("sets", "A ⊆ B iff A ∩ B = A", 1.0),
        ("sets", "power set of n-element set has 2^n elements", 1.0),
        ("sets", "A × B has |A| * |B| elements", 1.0),
        # ── Combinatorics ────────────────────────────────────────────────────
        ("combinatorics", "C(n,k) = n! / (k! * (n-k)!)  (combinations)", 1.0),
        ("combinatorics", "P(n,k) = n! / (n-k)!  (permutations)", 1.0),
        ("combinatorics", "C(n,0) = 1", 1.0),
        ("combinatorics", "C(n,n) = 1", 1.0),
        ("combinatorics", "C(n,1) = n", 1.0),
        ("combinatorics", "C(n,k) = C(n, n-k)  (symmetry)", 1.0),
        ("combinatorics", "Pascal's identity: C(n,k) = C(n-1,k-1) + C(n-1,k)", 1.0),
        ("combinatorics", "binomial theorem: (x+y)^n = sum_{k=0}^{n} C(n,k)*x^(n-k)*y^k", 1.0),
        ("combinatorics", "number of subsets of an n-element set is 2^n", 1.0),
        ("combinatorics", "derangements: D_n = n! * sum_{k=0}^{n} (-1)^k / k!", 0.9),
        # ── Probability ──────────────────────────────────────────────────────
        ("probability", "P(A ∪ B) = P(A) + P(B) - P(A ∩ B)", 1.0),
        ("probability", "P(not A) = 1 - P(A)", 1.0),
        ("probability", "P(A | B) = P(A ∩ B) / P(B)  for P(B) > 0", 1.0),
        ("probability", "Bayes theorem: P(A|B) = P(B|A)*P(A) / P(B)", 1.0),
        ("probability", "independence: P(A ∩ B) = P(A)*P(B)", 1.0),
        ("probability", "total probability: P(B) = sum_i P(B|A_i)*P(A_i)", 1.0),
        ("probability", "0 <= P(A) <= 1 for any event A", 1.0),
        ("probability", "P(sample space) = 1", 1.0),
        ("probability", "expected value: E[X] = sum x_i * P(X=x_i)", 1.0),
        ("probability", "variance: Var(X) = E[X^2] - (E[X])^2", 1.0),
    ]
    return facts


def _science_facts() -> List[Tuple[str, str, float]]:
    """130+ physics and chemistry facts."""
    facts: List[Tuple[str, str, float]] = [
        # ── Mechanics ────────────────────────────────────────────────────────
        ("physics", "Newton's second law: F = m*a", 1.0),
        ("physics", "Newton's first law: an object at rest stays at rest unless acted on by net force", 1.0),
        ("physics", "Newton's third law: every action has an equal and opposite reaction", 1.0),
        ("physics", "kinematic: v = u + a*t", 1.0),
        ("physics", "kinematic: s = u*t + 0.5*a*t^2", 1.0),
        ("physics", "kinematic: v^2 = u^2 + 2*a*s", 1.0),
        ("physics", "kinematic: s = (u + v)*t / 2", 1.0),
        ("physics", "kinetic energy: KE = 0.5*m*v^2", 1.0),
        ("physics", "potential energy (gravitational): PE = m*g*h", 1.0),
        ("physics", "work: W = F*d*cos(theta)", 1.0),
        ("physics", "power: P = W/t = F*v", 1.0),
        ("physics", "momentum: p = m*v", 1.0),
        ("physics", "impulse-momentum theorem: F*t = Δ(m*v)", 1.0),
        ("physics", "conservation of momentum: total momentum is constant in isolated system", 1.0),
        ("physics", "conservation of energy: total energy is constant in isolated system", 1.0),
        ("physics", "centripetal acceleration: a_c = v^2/r", 1.0),
        ("physics", "centripetal force: F_c = m*v^2/r", 1.0),
        ("physics", "gravitational force: F = G*m1*m2/r^2", 1.0),
        ("physics", "g on Earth surface ≈ 9.8 m/s^2", 1.0),
        ("physics", "torque: τ = r × F = r*F*sin(theta)", 1.0),
        ("physics", "moment of inertia (solid sphere): I = 2/5 * m*r^2", 0.95),
        ("physics", "moment of inertia (thin rod center): I = 1/12 * m*L^2", 0.95),
        # ── Waves and optics ─────────────────────────────────────────────────
        ("physics", "wave speed: v = f*λ", 1.0),
        ("physics", "frequency-period: f = 1/T", 1.0),
        ("physics", "photon energy: E = h*f", 1.0),
        ("physics", "speed of light in vacuum: c ≈ 3×10^8 m/s", 1.0),
        ("physics", "photon momentum: p = h/λ", 1.0),
        ("physics", "Snell's law: n1*sin(θ1) = n2*sin(θ2)", 1.0),
        ("physics", "Doppler effect: f_observed = f_source*(v±v_observer)/(v∓v_source)", 0.95),
        # ── Thermodynamics ───────────────────────────────────────────────────
        ("physics", "ideal gas law: P*V = n*R*T", 1.0),
        ("physics", "first law of thermodynamics: ΔU = Q - W", 1.0),
        ("physics", "second law: entropy of an isolated system never decreases", 1.0),
        ("physics", "heat transfer: Q = m*c*ΔT  (specific heat)", 1.0),
        ("physics", "thermal efficiency: η = 1 - T_cold/T_hot  (Carnot)", 1.0),
        ("physics", "entropy change: ΔS = Q_rev/T", 1.0),
        ("physics", "Boltzmann constant k_B ≈ 1.38×10^-23 J/K", 1.0),
        ("physics", "Avogadro's number: N_A ≈ 6.022×10^23", 1.0),
        ("physics", "universal gas constant: R ≈ 8.314 J/(mol·K)", 1.0),
        # ── Electricity and magnetism ─────────────────────────────────────────
        ("physics", "Ohm's law: V = I*R", 1.0),
        ("physics", "electric power: P = I*V", 1.0),
        ("physics", "electric power: P = I^2*R", 1.0),
        ("physics", "electric power: P = V^2/R", 1.0),
        ("physics", "capacitance: C = Q/V", 1.0),
        ("physics", "Coulomb's law: F = k*q1*q2/r^2", 1.0),
        ("physics", "electric field: E = F/q", 1.0),
        ("physics", "electric potential energy: U = k*q1*q2/r", 1.0),
        ("physics", "magnetic force on charge: F = q*v×B", 1.0),
        ("physics", "Faraday's law: EMF = -dΦ/dt", 1.0),
        ("physics", "resistors in series: R_total = R1 + R2 + ...", 1.0),
        ("physics", "resistors in parallel: 1/R_total = 1/R1 + 1/R2 + ...", 1.0),
        # ── Modern physics ────────────────────────────────────────────────────
        ("physics", "mass-energy equivalence: E = m*c^2", 1.0),
        ("physics", "de Broglie wavelength: λ = h/p = h/(m*v)", 1.0),
        ("physics", "Heisenberg uncertainty: Δx*Δp >= h/(4π)", 1.0),
        ("physics", "photoelectric effect: KE_max = h*f - φ  (work function φ)", 1.0),
        # ── Chemistry — Elements ─────────────────────────────────────────────
        ("chemistry", "hydrogen (H): atomic number 1, atomic mass ≈ 1", 1.0),
        ("chemistry", "helium (He): atomic number 2, atomic mass ≈ 4, noble gas", 1.0),
        ("chemistry", "lithium (Li): atomic number 3, alkali metal", 1.0),
        ("chemistry", "carbon (C): atomic number 6, atomic mass ≈ 12", 1.0),
        ("chemistry", "nitrogen (N): atomic number 7, diatomic gas N2", 1.0),
        ("chemistry", "oxygen (O): atomic number 8, diatomic gas O2", 1.0),
        ("chemistry", "sodium (Na): atomic number 11, alkali metal", 1.0),
        ("chemistry", "magnesium (Mg): atomic number 12, alkaline earth metal", 1.0),
        ("chemistry", "aluminum (Al): atomic number 13", 1.0),
        ("chemistry", "silicon (Si): atomic number 14, semiconductor", 1.0),
        ("chemistry", "phosphorus (P): atomic number 15", 1.0),
        ("chemistry", "sulfur (S): atomic number 16", 1.0),
        ("chemistry", "chlorine (Cl): atomic number 17, halogen, diatomic Cl2", 1.0),
        ("chemistry", "potassium (K): atomic number 19, alkali metal", 1.0),
        ("chemistry", "calcium (Ca): atomic number 20, alkaline earth metal", 1.0),
        ("chemistry", "iron (Fe): atomic number 26, transition metal", 1.0),
        ("chemistry", "copper (Cu): atomic number 29, transition metal", 1.0),
        ("chemistry", "zinc (Zn): atomic number 30, transition metal", 1.0),
        ("chemistry", "silver (Ag): atomic number 47, noble metal", 1.0),
        ("chemistry", "gold (Au): atomic number 79, noble metal", 1.0),
        ("chemistry", "mercury (Hg): atomic number 80, liquid at room temperature", 1.0),
        ("chemistry", "lead (Pb): atomic number 82", 1.0),
        # ── Chemistry — Reactions and laws ───────────────────────────────────
        ("chemistry", "acid + base → salt + water  (neutralization)", 1.0),
        ("chemistry", "oxidation: substance loses electrons (OIL — Oxidation Is Loss)", 1.0),
        ("chemistry", "reduction: substance gains electrons (RIG — Reduction Is Gain)", 1.0),
        ("chemistry", "law of conservation of mass: mass of reactants = mass of products", 1.0),
        ("chemistry", "Boyle's law: P1*V1 = P2*V2  (constant T)", 1.0),
        ("chemistry", "Charles's law: V1/T1 = V2/T2  (constant P)", 1.0),
        ("chemistry", "Gay-Lussac's law: P1/T1 = P2/T2  (constant V)", 1.0),
        ("chemistry", "Avogadro's law: equal volumes of gases at same T,P contain equal moles", 1.0),
        ("chemistry", "Avogadro's number: 6.022×10^23 particles per mole", 1.0),
        ("chemistry", "molar mass of water (H2O) ≈ 18 g/mol", 1.0),
        ("chemistry", "molar mass of CO2 ≈ 44 g/mol", 1.0),
        ("chemistry", "pH = -log[H+]; pH 7 is neutral", 1.0),
        ("chemistry", "strong acid: fully dissociates in water (HCl, H2SO4, HNO3)", 1.0),
        ("chemistry", "electronegativity increases across a period and up a group", 1.0),
        ("chemistry", "atomic radius increases down a group and decreases across a period", 1.0),
        ("chemistry", "ionic bond: metal + nonmetal electron transfer", 1.0),
        ("chemistry", "covalent bond: nonmetal + nonmetal electron sharing", 1.0),
        ("chemistry", "enthalpy: ΔH < 0 exothermic, ΔH > 0 endothermic", 1.0),
        ("chemistry", "Hess's law: ΔH_total = sum of ΔH of individual steps", 1.0),
        ("chemistry", "Le Chatelier's principle: system shifts to oppose a stress", 1.0),
    ]
    return facts


def _logic_facts() -> List[Tuple[str, str, float]]:
    """60+ logic and computer science facts."""
    facts: List[Tuple[str, str, float]] = [
        # ── Propositional logic ───────────────────────────────────────────────
        ("logic", "modus ponens: if P then Q; P is true; therefore Q is true", 1.0),
        ("logic", "modus tollens: if P then Q; not Q; therefore not P", 1.0),
        ("logic", "double negation: not(not P) = P", 1.0),
        ("logic", "De Morgan: not(A and B) = not A or not B", 1.0),
        ("logic", "De Morgan: not(A or B) = not A and not B", 1.0),
        ("logic", "implication equivalence: P → Q ≡ not P or Q", 1.0),
        ("logic", "contrapositive: P → Q ≡ not Q → not P", 1.0),
        ("logic", "tautology: P or not P = true", 1.0),
        ("logic", "contradiction: P and not P = false", 1.0),
        ("logic", "P and true = P", 1.0),
        ("logic", "P and false = false", 1.0),
        ("logic", "P or true = true", 1.0),
        ("logic", "P or false = P", 1.0),
        ("logic", "P and P = P  (idempotent)", 1.0),
        ("logic", "P or P = P  (idempotent)", 1.0),
        ("logic", "absorption: P and (P or Q) = P", 1.0),
        ("logic", "absorption: P or (P and Q) = P", 1.0),
        ("logic", "disjunctive syllogism: P or Q; not P; therefore Q", 1.0),
        ("logic", "hypothetical syllogism: P→Q; Q→R; therefore P→R", 1.0),
        ("logic", "biconditional: P ↔ Q ≡ (P→Q) and (Q→P)", 1.0),
        # ── Predicate logic ───────────────────────────────────────────────────
        ("logic", "universal instantiation: ∀x P(x) implies P(a) for any a", 1.0),
        ("logic", "existential generalization: P(a) implies ∃x P(x)", 1.0),
        ("logic", "negation of universal: not(∀x P(x)) ≡ ∃x not P(x)", 1.0),
        ("logic", "negation of existential: not(∃x P(x)) ≡ ∀x not P(x)", 1.0),
        # ── Big-O complexity ──────────────────────────────────────────────────
        ("computer_science", "complexity order: O(1) < O(log n) < O(sqrt n) < O(n) < O(n log n) < O(n^2) < O(n^3) < O(2^n) < O(n!)", 1.0),
        ("computer_science", "bubble sort: worst/average O(n^2), best O(n) when sorted", 1.0),
        ("computer_science", "insertion sort: worst O(n^2), best O(n), stable, in-place", 1.0),
        ("computer_science", "selection sort: always O(n^2), not stable, in-place", 1.0),
        ("computer_science", "merge sort: O(n log n) all cases, stable, O(n) extra space", 1.0),
        ("computer_science", "quicksort: average O(n log n), worst O(n^2), in-place", 1.0),
        ("computer_science", "heapsort: O(n log n) all cases, in-place, not stable", 1.0),
        ("computer_science", "counting sort: O(n + k) for k distinct values", 1.0),
        ("computer_science", "radix sort: O(d*(n+k)) for d digits, k digit range", 1.0),
        ("computer_science", "binary search: O(log n) on sorted array", 1.0),
        ("computer_science", "linear search: O(n)", 1.0),
        # ── Data structures ───────────────────────────────────────────────────
        ("computer_science", "array: O(1) random access, O(n) insert/delete middle", 1.0),
        ("computer_science", "linked list: O(n) access, O(1) insert/delete at head", 1.0),
        ("computer_science", "stack: LIFO, O(1) push/pop", 1.0),
        ("computer_science", "queue: FIFO, O(1) enqueue/dequeue", 1.0),
        ("computer_science", "binary search tree (BST): avg O(log n) search/insert, worst O(n)", 1.0),
        ("computer_science", "AVL/red-black tree: O(log n) guaranteed search/insert/delete", 1.0),
        ("computer_science", "hash table: avg O(1) insert/lookup, worst O(n) with collisions", 1.0),
        ("computer_science", "heap (binary): O(log n) insert/extract-min, O(1) peek-min", 1.0),
        ("computer_science", "BFS: O(V+E), finds shortest path in unweighted graph", 1.0),
        ("computer_science", "DFS: O(V+E), used for cycle detection and topological sort", 1.0),
        ("computer_science", "Dijkstra's algorithm: O((V+E) log V) shortest path, non-negative weights", 1.0),
        ("computer_science", "dynamic programming: overlapping subproblems + optimal substructure", 1.0),
        ("computer_science", "greedy: locally optimal choices, works when exchange argument holds", 1.0),
        ("computer_science", "P: problems solvable in polynomial time", 1.0),
        ("computer_science", "NP: problems verifiable in polynomial time", 1.0),
        ("computer_science", "P ⊆ NP; whether P = NP is an open problem", 1.0),
        ("computer_science", "NP-complete: hardest problems in NP (SAT, TSP, knapsack, ...)", 1.0),
        ("computer_science", "recursion: base case + self-similar smaller subproblem", 1.0),
        ("computer_science", "halting problem: no algorithm can decide if arbitrary program halts", 1.0),
        ("computer_science", "Shannon entropy: H = -sum p_i * log2(p_i)", 1.0),
        ("computer_science", "TCP/IP: reliable, connection-oriented; UDP: unreliable, connectionless", 0.95),
        ("computer_science", "HTTP is stateless; HTTPS adds TLS encryption", 0.95),
        ("computer_science", "bit: smallest unit of information (0 or 1)", 1.0),
        ("computer_science", "byte = 8 bits; kilobyte ≈ 1024 bytes", 1.0),
        ("computer_science", "two's complement: standard integer representation in binary", 1.0),
        ("computer_science", "IEEE 754: standard for floating-point arithmetic", 1.0),
    ]
    return facts


def _commonsense_facts() -> List[Tuple[str, str, float]]:
    """60+ commonsense facts covering spatial, causal, social, and biological."""
    facts: List[Tuple[str, str, float]] = [
        # ── Spatial / physical ────────────────────────────────────────────────
        ("commonsense", "things fall down due to gravity", 1.0),
        ("commonsense", "water flows downhill", 1.0),
        ("commonsense", "fire needs oxygen, fuel, and heat to burn", 1.0),
        ("commonsense", "ice melts when temperature rises above 0°C at standard pressure", 1.0),
        ("commonsense", "water boils at 100°C at standard atmospheric pressure", 1.0),
        ("commonsense", "objects expand when heated and contract when cooled", 1.0),
        ("commonsense", "light travels faster than sound", 1.0),
        ("commonsense", "the Sun rises in the east and sets in the west", 1.0),
        ("commonsense", "day and night are caused by Earth's rotation", 1.0),
        ("commonsense", "seasons are caused by Earth's axial tilt, not distance to Sun", 1.0),
        ("commonsense", "the Moon causes tides through gravitational pull", 1.0),
        ("commonsense", "magnets attract iron and have north/south poles; like poles repel", 1.0),
        ("commonsense", "electricity flows from high potential to low potential", 1.0),
        ("commonsense", "heat flows from hotter objects to cooler objects", 1.0),
        # ── Causal ───────────────────────────────────────────────────────────
        ("commonsense", "exercise improves cardiovascular fitness and muscle strength", 1.0),
        ("commonsense", "practice improves skill through muscle memory and learning", 1.0),
        ("commonsense", "sleep restores energy and consolidates memories", 1.0),
        ("commonsense", "stress can impair decision-making and memory", 1.0),
        ("commonsense", "eating too much sugar can lead to weight gain and health issues", 1.0),
        ("commonsense", "smoking increases the risk of lung cancer and heart disease", 1.0),
        ("commonsense", "washing hands reduces the spread of germs", 1.0),
        ("commonsense", "vaccines stimulate the immune system to prevent disease", 1.0),
        ("commonsense", "dehydration impairs cognitive and physical performance", 1.0),
        ("commonsense", "pollution damages ecosystems and human health", 1.0),
        ("commonsense", "sunlight enables photosynthesis in plants", 1.0),
        ("commonsense", "deforestation leads to loss of biodiversity and increased CO2", 1.0),
        # ── Social ────────────────────────────────────────────────────────────
        ("commonsense", "effective communication requires a shared language or code", 1.0),
        ("commonsense", "trust is built through consistent honest behavior over time", 1.0),
        ("commonsense", "conflict often arises from misunderstanding or resource scarcity", 1.0),
        ("commonsense", "cooperation generally produces better outcomes than competition for shared goals", 1.0),
        ("commonsense", "empathy involves understanding others' feelings and perspectives", 1.0),
        ("commonsense", "laws and social norms regulate behavior within a community", 1.0),
        ("commonsense", "money is a medium of exchange representing stored value", 1.0),
        ("commonsense", "supply and demand determine prices in a market economy", 1.0),
        ("commonsense", "education increases lifetime earning potential and opportunity", 1.0),
        ("commonsense", "democratic governments derive authority from the consent of the governed", 1.0),
        # ── Biological ────────────────────────────────────────────────────────
        ("commonsense", "plants need sunlight, water, and CO2 for photosynthesis", 1.0),
        ("commonsense", "animals need food (energy), water, and oxygen to survive", 1.0),
        ("commonsense", "cells are the basic structural and functional unit of life", 1.0),
        ("commonsense", "DNA carries genetic information and is passed from parent to offspring", 1.0),
        ("commonsense", "evolution occurs through natural selection acting on heritable variation", 1.0),
        ("commonsense", "viruses require a host cell to replicate", 1.0),
        ("commonsense", "bacteria are single-celled prokaryotes; some cause disease, many are beneficial", 1.0),
        ("commonsense", "the human body has about 37 trillion cells", 0.9),
        ("commonsense", "the brain controls the nervous system and higher cognitive functions", 1.0),
        ("commonsense", "the heart pumps blood through the circulatory system", 1.0),
        ("commonsense", "lungs exchange O2 and CO2 in respiration", 1.0),
        ("commonsense", "the liver detoxifies blood and produces bile for digestion", 1.0),
        ("commonsense", "muscles contract when stimulated by nerve signals", 1.0),
        ("commonsense", "bones provide structural support and protect organs", 1.0),
        # ── Practical / everyday ──────────────────────────────────────────────
        ("commonsense", "knives are used for cutting; hammers are used for driving nails", 1.0),
        ("commonsense", "cars need fuel; electric vehicles need charged batteries", 1.0),
        ("commonsense", "computers process information using binary arithmetic", 1.0),
        ("commonsense", "the internet connects computers globally via TCP/IP", 1.0),
        ("commonsense", "photographs capture light and preserve visual information", 1.0),
        ("commonsense", "cooking uses heat to alter food's texture, flavor, and safety", 1.0),
        ("commonsense", "maps represent geographical space at reduced scale", 1.0),
        ("commonsense", "clocks measure time; calendars track days and seasons", 1.0),
    ]
    return facts


# ── Commonsense KB triples ─────────────────────────────────────────────────────
# These are (subject, relation, object) triples loaded into CommonSenseBase.

_CS_TRIPLES: List[Tuple[str, str, str]] = [
    # Math / logic
    ("zero",          "HasProperty", "additive_identity"),
    ("one",           "HasProperty", "multiplicative_identity"),
    ("prime_number",  "HasProperty", "only_divisible_by_1_and_itself"),
    ("logarithm",     "IsA",         "inverse_of_exponentiation"),
    ("derivative",    "Represents",  "instantaneous_rate_of_change"),
    ("integral",      "Represents",  "area_under_curve"),
    ("probability",   "HasProperty", "value_between_0_and_1"),
    ("entropy",       "Measures",    "information_content"),
    # Physics
    ("force",         "Causes",      "acceleration"),
    ("gravity",       "Causes",      "falling"),
    ("heat",          "Causes",      "expansion"),
    ("friction",      "Causes",      "energy_loss_as_heat"),
    ("wave",          "HasProperty", "frequency_and_wavelength"),
    ("photon",        "IsA",         "quantum_of_light"),
    ("electron",      "HasProperty", "negative_charge"),
    ("proton",        "HasProperty", "positive_charge"),
    ("neutron",       "HasProperty", "no_charge"),
    ("nucleus",       "PartOf",      "atom"),
    ("atom",          "PartOf",      "molecule"),
    ("molecule",      "PartOf",      "substance"),
    # Chemistry
    ("acid",          "HasProperty", "donates_protons"),
    ("base",          "HasProperty", "accepts_protons"),
    ("catalyst",      "Causes",      "faster_reaction"),
    ("oxidation",     "Causes",      "electron_loss"),
    ("reduction",     "Causes",      "electron_gain"),
    ("water",         "IsA",         "polar_solvent"),
    ("carbon",        "CapableOf",   "forming_four_bonds"),
    ("oxygen",        "RequiredFor", "combustion"),
    # Biology
    ("cell",          "IsA",         "basic_unit_of_life"),
    ("dna",           "HasProperty", "genetic_information"),
    ("photosynthesis","RequiredFor", "sunlight"),
    ("plant",         "CapableOf",   "photosynthesis"),
    ("animal",        "RequiredFor", "oxygen"),
    ("brain",         "CapableOf",   "reasoning"),
    ("virus",         "RequiredFor", "host_cell"),
    ("evolution",     "Causes",      "speciation"),
    # Computer science
    ("algorithm",     "UsedFor",     "problem_solving"),
    ("recursion",     "RequiredFor", "base_case"),
    ("hash_table",    "CapableOf",   "O(1)_average_lookup"),
    ("binary_tree",   "CapableOf",   "O(log_n)_search"),
    ("compiler",      "UsedFor",     "translating_source_to_machine_code"),
    ("cache",         "UsedFor",     "speeding_up_memory_access"),
    ("encryption",    "UsedFor",     "securing_data"),
    # Commonsense
    ("sleep",         "Causes",      "memory_consolidation"),
    ("exercise",      "Causes",      "improved_fitness"),
    ("fire",          "RequiredFor", "oxygen"),
    ("water",         "HasProperty", "flows_downhill"),
    ("gravity",       "HasProperty", "attracts_masses"),
    ("trust",         "RequiredFor", "consistency"),
    ("communication", "RequiredFor", "shared_language"),
    ("education",     "Enables",     "higher_earning_potential"),
    ("sun",           "Causes",      "photosynthesis"),
    ("deforestation", "Causes",      "biodiversity_loss"),
    ("vaccine",       "Causes",      "immune_response"),
    ("pollution",     "Causes",      "ecosystem_damage"),
]


# ── Singleton and class ────────────────────────────────────────────────────────

_LOADER_SINGLETON: "StaticKnowledgeLoader | None" = None


class StaticKnowledgeLoader:
    """
    Loads 400+ curated facts into WorldModel and CommonSenseBase WITHOUT any
    LLM calls.  Call ``load_all()`` once at startup (idempotent; uses a
    sentinel fact to detect if already loaded).
    """

    def __init__(self) -> None:
        self._loaded: bool = False

    # ── Category loaders ──────────────────────────────────────────────────────

    def load_math(self) -> List[Tuple[str, str, float]]:
        """Return (domain, fact_str, confidence) tuples for math facts."""
        return _math_facts()

    def load_science(self) -> List[Tuple[str, str, float]]:
        """Return (domain, fact_str, confidence) tuples for science facts."""
        return _science_facts()

    def load_logic(self) -> List[Tuple[str, str, float]]:
        """Return (domain, fact_str, confidence) tuples for logic/CS facts."""
        return _logic_facts()

    def load_commonsense(self) -> List[Tuple[str, str, float]]:
        """Return (domain, fact_str, confidence) tuples for commonsense facts."""
        return _commonsense_facts()

    # ── Main loader ───────────────────────────────────────────────────────────

    def load_all(self) -> int:
        """
        Load all knowledge categories into WorldModel and CommonSenseBase.

        Returns the total number of facts loaded.  Safe to call multiple times;
        subsequent calls return 0 if facts were already loaded.
        """
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
        except Exception as e:
            log.error("StaticKnowledgeLoader: cannot import WorldModel: %s", e)
            return 0

        # Check sentinel to avoid re-loading.
        existing_meta = wm.get_facts(_SENTINEL_DOMAIN)
        for f in existing_meta:
            if f.get("fact") == _SENTINEL_FACT:
                self._loaded = True
                log.info("StaticKnowledgeLoader: already loaded, skipping.")
                return 0

        total = 0

        # ── Load WorldModel facts ─────────────────────────────────────────────
        all_tuples: List[Tuple[str, str, float]] = (
            self.load_math()
            + self.load_science()
            + self.load_logic()
            + self.load_commonsense()
        )
        for domain, fact_str, conf in all_tuples:
            try:
                wm.add_fact(domain, fact_str, confidence=conf, source="static_knowledge")
                total += 1
            except Exception as e:
                log.debug("StaticKnowledgeLoader: add_fact error for %r: %s", fact_str[:40], e)

        # ── Load CommonSenseBase triples ──────────────────────────────────────
        try:
            from sare.knowledge.commonsense import CommonSenseBase
            kb = CommonSenseBase()
            kb.load()  # load any previously persisted facts first
            for subj, rel, obj in _CS_TRIPLES:
                try:
                    kb._add(subj, rel, obj)
                    total += 1
                except Exception as e:
                    log.debug("StaticKnowledgeLoader: _add triple error: %s", e)
            kb.save()
        except Exception as e:
            log.warning("StaticKnowledgeLoader: CommonSenseBase unavailable: %s", e)

        # ── Write sentinel ────────────────────────────────────────────────────
        try:
            wm.add_fact(_SENTINEL_DOMAIN, _SENTINEL_FACT, confidence=1.0,
                        source="static_knowledge")
        except Exception as e:
            log.debug("StaticKnowledgeLoader: could not write sentinel: %s", e)

        # ── Persist WorldModel ────────────────────────────────────────────────
        try:
            wm.save()
        except Exception as e:
            log.warning("StaticKnowledgeLoader: WorldModel.save() failed: %s", e)

        self._loaded = True
        log.info("StaticKnowledgeLoader: loaded %d facts.", total)
        return total

    # ── Status ────────────────────────────────────────────────────────────────

    def get_load_status(self) -> Dict[str, Any]:
        """Return {already_loaded: bool, fact_count: int}."""
        count = (
            len(_math_facts())
            + len(_science_facts())
            + len(_logic_facts())
            + len(_commonsense_facts())
            + len(_CS_TRIPLES)
        )
        return {
            "already_loaded": self._loaded,
            "fact_count": count,
        }


def get_static_loader() -> StaticKnowledgeLoader:
    """Return the module-level singleton StaticKnowledgeLoader."""
    global _LOADER_SINGLETON
    if _LOADER_SINGLETON is None:
        _LOADER_SINGLETON = StaticKnowledgeLoader()
    return _LOADER_SINGLETON
