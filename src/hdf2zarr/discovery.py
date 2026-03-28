"""File discovery for HDF5 files in directories."""

from pathlib import Path

import structlog

from hdf2zarr.exceptions import DiscoveryError

log = structlog.get_logger()

#: Default file extensions recognized as HDF5 files.
DEFAULT_EXTENSIONS = frozenset({".h5", ".hdf5", ".hdf", ".he5", ".nxs"})

#: Default glob patterns derived from the default extensions.
DEFAULT_PATTERNS = tuple(f"*{ext}" for ext in sorted(DEFAULT_EXTENSIONS))


def parse_patterns(pattern_str: str) -> list[str]:
    """Parse a comma-separated pattern string into a list of glob patterns.

    Parameters
    ----------
    pattern_str : str
        Comma-separated glob patterns, e.g. ``"*.h5,*.hdf5"``.

    Returns
    -------
    list[str]
        Individual glob patterns with whitespace stripped.
    """
    return [p.strip() for p in pattern_str.split(",") if p.strip()]


def discover_files(
    sources: list[Path],
    *,
    recursive: bool = False,
    patterns: list[str] | None = None,
) -> list[Path]:
    """Discover HDF5 files from a list of file/directory sources.

    Files are returned sorted by name for deterministic ordering.
    Duplicates are removed.

    Parameters
    ----------
    sources : list[Path]
        Paths to individual HDF5 files or directories to scan.
    recursive : bool
        If ``True``, scan directories recursively.
    patterns : list[str] or None
        Glob patterns for matching files in directories. If ``None``,
        uses the default HDF5 extensions.

    Returns
    -------
    list[Path]
        Sorted list of discovered HDF5 file paths.

    Raises
    ------
    DiscoveryError
        If a source path does not exist.
    """
    if patterns is None:
        patterns = list(DEFAULT_PATTERNS)

    found: set[Path] = set()

    for source in sources:
        source = source.resolve()

        if not source.exists():
            raise DiscoveryError(f"Source path does not exist: {source}")

        if source.is_file():
            found.add(source)
            log.debug("discovered_file", path=str(source))

        elif source.is_dir():
            dir_count = 0
            for pattern in patterns:
                glob_method = source.rglob if recursive else source.glob
                for match in glob_method(pattern):
                    if match.is_file():
                        found.add(match)
                        dir_count += 1
            log.debug(
                "discovered_directory",
                path=str(source),
                files=dir_count,
                recursive=recursive,
            )
        else:
            log.warning(
                "skipping_source", path=str(source), reason="not a file or directory"
            )

    result = sorted(found)
    log.info("discovery_complete", total_files=len(result))
    return result


def resolve_output_path(
    source: Path,
    output_dir: Path | None,
) -> Path:
    """Determine the output Zarr store path for a given source file.

    Parameters
    ----------
    source : Path
        The source HDF5 file path.
    output_dir : Path or None
        Output directory. If ``None``, the Zarr store is placed alongside
        the source file.

    Returns
    -------
    Path
        The output Zarr store path (e.g. ``/data/file.zarr``).
    """
    zarr_name = source.stem + ".zarr"
    if output_dir is not None:
        return output_dir / zarr_name
    return source.parent / zarr_name
