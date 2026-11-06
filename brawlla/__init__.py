"""BrawlIA: observação heurística local, somente leitura."""

from .config import VERSION as __version__
from .engine import analyze, SessionAnalyzer

__all__ = ["__version__", "analyze", "SessionAnalyzer"]
