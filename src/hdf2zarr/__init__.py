"""hdf2zarr — Convert HDF5 files to Zarr format."""

__version__ = "0.1.0"


def main() -> None:
    """Entry point for the ``hdf2zarr`` console script."""
    from hdf2zarr.cli import app

    app()
