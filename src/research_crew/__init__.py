"""research-crew: distributed multi-agent research service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("research-crew")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
