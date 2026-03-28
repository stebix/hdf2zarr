"""Rich progress tracking for hdf2zarr conversions."""

from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)


class ProgressTracker(Protocol):
    """Protocol for progress tracking during conversion.

    Implementations of this protocol are passed to ``HDF5ToZarrConverter``
    to report progress.
    """

    def start_file(self, source: Path, total_datasets: int) -> None:
        """Signal that conversion of a new file has started.

        Parameters
        ----------
        source : Path
            The source HDF5 file.
        total_datasets : int
            Total number of datasets in the file.
        """
        ...

    def advance_dataset(self) -> None:
        """Signal that one dataset has been converted."""
        ...

    def finish_file(self) -> None:
        """Signal that conversion of the current file has completed."""
        ...


class RichProgressTracker:
    """Rich-based progress tracker with file-level and dataset-level bars.

    Parameters
    ----------
    console : Console
        Rich console instance for output.
    total_files : int
        Total number of files to convert.
    """

    def __init__(self, console: Console, total_files: int) -> None:
        self.console = console
        self._total_files = total_files

        self._file_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self._dataset_progress = Progress(
            TextColumn("  "),
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        )

        self._file_task: TaskID | None = None
        self._dataset_task: TaskID | None = None

    def __enter__(self) -> "RichProgressTracker":
        self._file_progress.start()
        self._file_task = self._file_progress.add_task(
            "Converting files", total=self._total_files
        )
        return self

    def __exit__(self, *args: object) -> None:
        self._file_progress.stop()

    def start_file(self, source: Path, total_datasets: int) -> None:
        """Signal start of a new file conversion.

        Parameters
        ----------
        source : Path
            Source HDF5 file.
        total_datasets : int
            Number of datasets in the file.
        """
        if self._dataset_task is not None:
            self._dataset_progress.stop()

        self._dataset_progress.start()
        self._dataset_task = self._dataset_progress.add_task(
            source.name, total=total_datasets
        )

    def advance_dataset(self) -> None:
        """Advance the dataset progress bar by one."""
        if self._dataset_task is not None:
            self._dataset_progress.advance(self._dataset_task)

    def finish_file(self) -> None:
        """Complete the current file and advance the file progress bar."""
        if self._dataset_task is not None:
            self._dataset_progress.stop()
            self._dataset_task = None

        if self._file_task is not None:
            self._file_progress.advance(self._file_task)


class QuietProgressTracker:
    """No-op progress tracker for quiet mode.

    Implements the same interface as ``RichProgressTracker`` but does nothing.
    """

    def start_file(self, source: Path, total_datasets: int) -> None:
        """No-op."""

    def advance_dataset(self) -> None:
        """No-op."""

    def finish_file(self) -> None:
        """No-op."""

    def __enter__(self) -> "QuietProgressTracker":
        return self

    def __exit__(self, *args: object) -> None:
        pass
