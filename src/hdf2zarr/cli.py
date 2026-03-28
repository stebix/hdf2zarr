"""Command-line interface for hdf2zarr."""

from pathlib import Path
from typing import TYPE_CHECKING

import click
import structlog

if TYPE_CHECKING:
    import h5py
    from rich.console import Console
    from rich.tree import Tree

from hdf2zarr.converter import HDF5ToZarrConverter
from hdf2zarr.discovery import (
    DEFAULT_PATTERNS,
    discover_files,
    parse_patterns,
    resolve_output_path,
)
from hdf2zarr.exceptions import ConversionError, DiscoveryError
from hdf2zarr.logging import configure_logging
from hdf2zarr.models import ConversionConfig, ConversionResult, ConversionSummary, Timer

log = structlog.get_logger()


@click.group()
@click.version_option(package_name="hdf2zarr")
def app() -> None:
    """hdf2zarr — Convert HDF5 files to Zarr format."""


@app.command()
@click.argument(
    "sources", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for Zarr stores. Default: alongside source files.",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    default=False,
    help="Recursively scan directories for HDF5 files.",
)
@click.option(
    "--pattern",
    default=",".join(DEFAULT_PATTERNS),
    show_default=True,
    help="Comma-separated glob patterns for directory scanning.",
)
@click.option(
    "--compression",
    type=click.Choice(
        ["zstd", "lz4", "gzip", "none", "preserve"], case_sensitive=False
    ),
    default="zstd",
    show_default=True,
    help="Compression codec for output.",
)
@click.option(
    "--compression-level",
    type=int,
    default=3,
    show_default=True,
    help="Compression level (codec-dependent).",
)
@click.option(
    "--chunking",
    type=click.Choice(["preserve", "auto"], case_sensitive=False),
    default="preserve",
    show_default=True,
    help="Chunking strategy.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing Zarr stores.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be converted without converting.",
)
@click.option(
    "--zarr-format",
    type=click.Choice(["2", "3"]),
    default="3",
    show_default=True,
    help="Zarr format version.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v for INFO, -vv for DEBUG).",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress all output except errors.",
)
@click.option(
    "--skip-on-error/--fail-on-error",
    default=True,
    show_default=True,
    help="Continue on error or halt immediately.",
)
@click.option(
    "--json-log",
    is_flag=True,
    default=False,
    help="Emit structured JSON logs to stderr.",
)
def convert(
    sources: tuple[Path, ...],
    output_dir: Path | None,
    recursive: bool,
    pattern: str,
    compression: str,
    compression_level: int,
    chunking: str,
    overwrite: bool,
    dry_run: bool,
    zarr_format: str,
    verbose: int,
    quiet: bool,
    skip_on_error: bool,
    json_log: bool,
) -> None:
    """Convert HDF5 files to Zarr format.

    SOURCES can be individual HDF5 files or directories to scan.
    """
    configure_logging(verbosity=verbose, quiet=quiet, json_output=json_log)

    from rich.console import Console

    console = Console()

    # Discover files
    patterns = parse_patterns(pattern)
    try:
        files = discover_files(list(sources), recursive=recursive, patterns=patterns)
    except DiscoveryError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc

    if not files:
        console.print("[yellow]No HDF5 files found.[/]")
        return

    console.print(f"Found [bold]{len(files)}[/] HDF5 file(s)")

    # Build config
    config = ConversionConfig(
        compression=compression.lower(),
        compression_level=compression_level,
        chunking=chunking.lower(),
        overwrite=overwrite,
        zarr_format=int(zarr_format),
        skip_on_error=skip_on_error,
        output_dir=output_dir,
    )

    # Resolve output paths
    file_pairs = []
    for f in files:
        dest = resolve_output_path(f, output_dir)
        file_pairs.append((f, dest))

    # Dry run
    if dry_run:
        _print_dry_run(console, file_pairs)
        return

    # Check for existing outputs
    if not overwrite:
        conflicts = [(s, d) for s, d in file_pairs if d.exists()]
        if conflicts:
            console.print(
                f"[bold red]Error:[/] {len(conflicts)} output(s) already exist. "
                "Use --overwrite to replace."
            )
            for _, d in conflicts:
                console.print(f"  {d}")
            raise SystemExit(1)

    # Ensure output directory exists
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Convert with progress tracking
    from hdf2zarr.progress import QuietProgressTracker, RichProgressTracker

    if quiet:
        tracker: RichProgressTracker | QuietProgressTracker = QuietProgressTracker()
    else:
        tracker = RichProgressTracker(console, total_files=len(file_pairs))

    summary = ConversionSummary()

    with Timer() as total_timer, tracker:
        converter = HDF5ToZarrConverter(config, progress=tracker)
        for source, dest in file_pairs:
            try:
                result = converter.convert(source, dest)
                summary.results.append(result)
            except ConversionError as exc:
                if skip_on_error:
                    console.print(f"  [red]Failed:[/] {exc}")
                    summary.results.append(
                        ConversionResult(
                            source=source,
                            destination=dest,
                            success=False,
                            errors=[str(exc)],
                        )
                    )
                else:
                    console.print(f"[bold red]Fatal:[/] {exc}")
                    raise SystemExit(1) from exc

    summary.elapsed = total_timer.elapsed
    _print_summary(console, summary)

    if summary.failed > 0:
        raise SystemExit(1)


@app.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--show-attrs/--hide-attrs",
    default=True,
    show_default=True,
    help="Show or hide attributes.",
)
@click.option(
    "--depth",
    type=int,
    default=None,
    help="Max depth to display. Default: unlimited.",
)
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet", is_flag=True, default=False)
def info(
    file: Path,
    show_attrs: bool,
    depth: int | None,
    verbose: int,
    quiet: bool,
) -> None:
    """Display structure and metadata of an HDF5 file."""
    configure_logging(verbosity=verbose, quiet=quiet)

    import h5py
    from rich.console import Console
    from rich.tree import Tree

    console = Console()

    try:
        with h5py.File(file, "r") as f:
            tree = Tree(f"[bold]{file.name}[/]")
            _build_tree(f, tree, show_attrs=show_attrs, depth=depth, current_depth=0)
            console.print(tree)
    except OSError as exc:
        console.print(f"[bold red]Error:[/] Cannot open {file}: {exc}")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _print_dry_run(
    console: "Console",
    file_pairs: list[tuple[Path, Path]],
) -> None:
    """Print a dry-run table of planned conversions.

    Parameters
    ----------
    console : rich.console.Console
        Rich console instance.
    file_pairs : list[tuple[Path, Path]]
        Source-destination path pairs.
    """
    from rich.table import Table

    table = Table(title="Dry Run: Planned Conversions")
    table.add_column("Source", style="cyan")
    table.add_column("Destination", style="green")
    table.add_column("Exists", style="yellow")

    for source, dest in file_pairs:
        exists = "yes" if dest.exists() else "no"
        table.add_row(str(source), str(dest), exists)

    console.print(table)
    console.print(f"\nTotal: [bold]{len(file_pairs)}[/] file(s)")


def _print_summary(
    console: "Console",
    summary: ConversionSummary,
) -> None:
    """Print a conversion summary table.

    Parameters
    ----------
    console : rich.console.Console
        Rich console instance.
    summary : ConversionSummary
        Aggregated conversion results.
    """
    from rich.table import Table

    table = Table(title="Conversion Summary")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Datasets", justify="right")
    table.add_column("Groups", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Time", justify="right")

    for r in summary.results:
        if r.success and not r.errors:
            status = "[green]OK[/]"
        elif r.success and r.errors:
            status = "[yellow]PARTIAL[/]"
        else:
            status = "[red]FAILED[/]"

        datasets = f"{r.datasets_copied}"
        if r.datasets_skipped:
            datasets += f" ({r.datasets_skipped} skipped)"

        size = _format_bytes(r.bytes_copied)
        time_str = f"{r.elapsed:.1f}s"

        table.add_row(
            r.source.name, status, datasets, str(r.groups_copied), size, time_str
        )

    console.print(table)
    console.print(
        f"\nTotal: [bold]{summary.total_files}[/] files | "
        f"[green]{summary.successful} OK[/]"
        + (f" | [yellow]{summary.partial} partial[/]" if summary.partial else "")
        + (f" | [red]{summary.failed} failed[/]" if summary.failed else "")
        + f" | {_format_bytes(summary.total_bytes)}"
        + f" | {summary.elapsed:.1f}s"
    )


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string.

    Parameters
    ----------
    n : int
        Number of bytes.

    Returns
    -------
    str
        Human-readable string (e.g. ``"1.2 MB"``).
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


def _build_tree(
    group: "h5py.Group",
    tree: "Tree",
    *,
    show_attrs: bool,
    depth: int | None,
    current_depth: int,
) -> None:
    """Recursively build a Rich tree from an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        Current HDF5 group.
    tree : rich.tree.Tree
        Current tree node.
    show_attrs : bool
        Whether to display attributes.
    depth : int or None
        Max depth. ``None`` means unlimited.
    current_depth : int
        Current recursion depth.
    """
    import h5py

    if show_attrs and group.attrs:
        for key, value in group.attrs.items():
            tree.add(f"[dim]@{key} = {value!r}[/]")

    if depth is not None and current_depth >= depth:
        if len(group) > 0:
            tree.add("[dim]...[/]")
        return

    for name in group:
        try:
            item = group[name]
        except Exception:
            tree.add(f"[red]{name} (inaccessible)[/]")
            continue

        if isinstance(item, h5py.Group):
            branch = tree.add(f"[bold blue]{name}/[/]")
            _build_tree(
                item,
                branch,
                show_attrs=show_attrs,
                depth=depth,
                current_depth=current_depth + 1,
            )
        elif isinstance(item, h5py.Dataset):
            shape_str = "x".join(str(s) for s in item.shape) if item.shape else "scalar"
            dtype_str = str(item.dtype)
            chunks_str = f"  chunks={item.chunks}" if item.chunks else ""
            compression_str = f"  {item.compression}" if item.compression else ""
            label = f"[green]{name}[/]  {dtype_str} ({shape_str}){chunks_str}{compression_str}"
            node = tree.add(label)

            if show_attrs and item.attrs:
                for key, value in item.attrs.items():
                    node.add(f"[dim]@{key} = {value!r}[/]")
        else:
            tree.add(f"[yellow]{name} ({type(item).__name__})[/]")
