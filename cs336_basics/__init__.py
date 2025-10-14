import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

# Public re-exports
from .linear import Linear  # noqa: F401
from .embedding import Embedding  # noqa: F401
