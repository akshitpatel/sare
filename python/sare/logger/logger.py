"""
SARE-HX Structured Logger — re-export shim.

The canonical implementation lives in sare.sare_logging.logger.
This module re-exports everything from there so that any code
importing from sare.logging.logger continues to work.
"""
from sare.sare_logging.logger import SolveLog, SareLogger  # noqa: F401

__all__ = ["SolveLog", "SareLogger"]
