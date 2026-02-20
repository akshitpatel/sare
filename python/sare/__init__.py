"""SARE-HX: Graph-Native Cognitive Architecture"""
__version__ = "0.1.0"

try:
    from .sare_bindings import *
except ImportError:
    pass
