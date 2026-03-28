"""Configuration and result dataclasses for hdf2zarr."""

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConversionConfig:
    """Immutable configuration for an HDF5-to-Zarr conversion run.

    Parameters
    ----------
    compression : str
        Compression codec name. One of ``"zstd"``, ``"lz4"``, ``"gzip"``,
        ``"none"``, or ``"preserve"``.
    compression_level : int
        Compression level (codec-dependent).
    chunking : str
        Chunking strategy. ``"preserve"`` keeps HDF5 chunks (falls back to
        auto for contiguous datasets), ``"auto"`` lets zarr decide.
    overwrite : bool
        Whether to overwrite existing Zarr stores.
    zarr_format : int
        Zarr format version (2 or 3).
    skip_on_error : bool
        If ``True``, skip files that fail and continue with the rest.
    output_dir : Path | None
        Output directory. ``None`` means write alongside the source file.
    """

    compression: str = "zstd"
    compression_level: int = 3
    chunking: str = "preserve"
    overwrite: bool = False
    zarr_format: int = 3
    skip_on_error: bool = True
    output_dir: Path | None = None


@dataclass
class ConversionResult:
    """Outcome of converting a single HDF5 file.

    Parameters
    ----------
    source : Path
        Path to the source HDF5 file.
    destination : Path
        Path to the output Zarr store.
    success : bool
        Whether the conversion completed without errors.
    groups_copied : int
        Number of HDF5 groups copied.
    datasets_copied : int
        Number of HDF5 datasets copied.
    datasets_skipped : int
        Number of datasets skipped due to unsupported dtypes or errors.
    bytes_copied : int
        Total bytes of array data copied.
    errors : list[str]
        Error messages for any issues encountered.
    elapsed : float
        Wall-clock time in seconds for this file's conversion.
    """

    source: Path
    destination: Path
    success: bool = True
    groups_copied: int = 0
    datasets_copied: int = 0
    datasets_skipped: int = 0
    bytes_copied: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class ConversionSummary:
    """Aggregate results for a batch conversion run.

    Parameters
    ----------
    results : list[ConversionResult]
        Per-file results.
    elapsed : float
        Total wall-clock time for the entire run.
    """

    results: list[ConversionResult] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def total_files(self) -> int:
        """Total number of files processed."""
        return len(self.results)

    @property
    def successful(self) -> int:
        """Number of fully successful conversions."""
        return sum(1 for r in self.results if r.success and not r.errors)

    @property
    def partial(self) -> int:
        """Number of conversions that completed with some skipped datasets."""
        return sum(1 for r in self.results if r.success and r.errors)

    @property
    def failed(self) -> int:
        """Number of completely failed conversions."""
        return sum(1 for r in self.results if not r.success)

    @property
    def total_bytes(self) -> int:
        """Total bytes copied across all files."""
        return sum(r.bytes_copied for r in self.results)


class Timer:
    """Simple context-manager timer.

    Examples
    --------
    >>> with Timer() as t:
    ...     pass
    >>> t.elapsed >= 0.0
    True
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start
