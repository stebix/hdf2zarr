"""Exception hierarchy for hdf2zarr."""


class HDF2ZarrError(Exception):
    """Base exception for all hdf2zarr errors."""


class UnsupportedDtypeError(HDF2ZarrError):
    """Raised when an HDF5 dtype cannot be mapped to a Zarr-compatible dtype."""

    def __init__(self, dtype: object, path: str) -> None:
        self.dtype = dtype
        self.path = path
        super().__init__(f"Unsupported dtype {dtype!r} at {path!r}")


class ConversionError(HDF2ZarrError):
    """Raised when a file-level conversion fails."""

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"Conversion failed for {source!r}: {reason}")


class DiscoveryError(HDF2ZarrError):
    """Raised when file discovery encounters an error."""
