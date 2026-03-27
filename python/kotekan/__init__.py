from ._version import get_versions

# Should probably set this from the overall kotekan version
__version__ = get_versions()["version"]
del get_versions
