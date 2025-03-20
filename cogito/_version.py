from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cogito")
except PackageNotFoundError:
    __version__ = "unknown"
