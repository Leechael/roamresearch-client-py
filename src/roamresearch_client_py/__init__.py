from importlib.metadata import version, PackageNotFoundError

from .client import RoamClient
from .config import configure_logging

__all__ = ["RoamClient", "configure_logging"]

try:
    __version__ = version("roamresearch-client-py")
except PackageNotFoundError:
    # Package not installed (e.g., running from source)
    __version__ = "0.0.0-dev"