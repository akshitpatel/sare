"""Transform family routing — maps domain labels to relevant transform subsets."""

TRANSFORM_FAMILIES = {
    "algebra":    ["AddZeroElimination", "MulOneElimination", "ConstantFolding",
                   "MulZeroElimination", "DistributiveExpansion", "AlgebraicFactoring",
                   "CombineLikeTerms", "SubtractSelfElimination", "AdditiveCancellation",
                   "DivisionSelfElimination", "PowerZeroElimination", "PowerOneElimination",
                   "PerfectSquareTrinomial", "CommutativityCanonicalize", "MacroTransform"],
    "equations":  ["EquationSolver", "LinearEquationSolver", "MultiplyEquationSolver",
                   "EquationSubtractConst", "QuadraticSolver", "AddZeroElimination",
                   "MulOneElimination", "CombineLikeTerms"],
    "logic":      ["DoubleNegation", "BooleanAndTrue", "BooleanAndFalse", "BooleanOrFalse",
                   "BooleanOrTrue", "BooleanIdempotent", "DeMorganAnd", "DeMorganOr",
                   "ImplicationExpansion", "ContrapositiveLaw",
                   "ModusPonensTransform", "DoubleNegRemoveTransform", "ImpliesElimTransform",
                   "FillUnknownTransform", "ChainInferenceTransform", "TaxonomyChainTransform"],
    "calculus":   ["DerivativeConstant", "DerivativePower", "DerivativeLinear",
                   "SumRuleDerivative", "ProductRuleDerivative", "SinDerivative",
                   "CosDerivative", "ChainRuleSin", "ChainRuleCos", "ExpDerivative",
                   "LnDerivative", "ChainRuleExp", "QuotientRule",
                   "IntegralConstant", "IntegralLinear", "IntegralPower", "IntegralSumRule"],
    "trig":       ["TrigZero", "CosZero", "SqrtSquare", "PythagoreanSimplify",
                   "AngleSumTriangle", "CombineLikeTerms"],
    "physics":    ["NewtonsSecondLaw", "OhmsLaw", "KinematicVelocity",
                   "KinematicDisplacement", "EnergyKinetic", "EnergyPotential", "PVnRT",
                   "IdealGasLaw", "ConstantFolding", "AddZeroElimination"],
    "chemistry":  ["IdealGasLaw", "StoichiometryCoefficients", "AvogadroConversion",
                   "ConservationOfMass", "ConstantFolding"],
    "code":       ["IfTrueElimTransform", "IfFalseElimTransform", "NotTrueTransform",
                   "NotFalseTransform", "AndSelfTransform", "OrSelfTransform",
                   "SelfAssignElimTransform", "ReturnConstantFoldTransform"],
    "probability":["ProbabilityEmptySet", "ProbabilityComplement", "ProbabilityOne",
                   "ConstantFolding"],
    "general":    None,  # None = use all transforms
}

# Fallback expansion: if family match finds <5 transforms, also include algebra core
_ALGEBRA_CORE = TRANSFORM_FAMILIES["algebra"]

def get_family_transforms(domain: str, all_transforms: list) -> list:
    """Return the subset of transforms relevant for the given domain.
    Falls back to all transforms if domain unknown or family too small."""
    family_names = TRANSFORM_FAMILIES.get(domain.lower())
    if family_names is None:
        return all_transforms

    name_set = set(family_names)
    # Always include any ConceptRule (learned rules) regardless of domain
    filtered = [t for t in all_transforms
                if type(t).__name__ in name_set or type(t).__name__ == "ConceptRule"]

    # Fallback: if fewer than 5 matched, include algebra core too
    if len(filtered) < 5:
        core_set = set(_ALGEBRA_CORE)
        filtered = [t for t in all_transforms
                    if type(t).__name__ in name_set or
                       type(t).__name__ in core_set or
                       type(t).__name__ == "ConceptRule"]

    return filtered if filtered else all_transforms
