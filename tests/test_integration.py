"""End-to-end integration tests for hdf2zarr conversion."""

from pathlib import Path

import h5py
import numpy as np
import zarr
import zarr.storage
from click.testing import CliRunner

from hdf2zarr.cli import app
from hdf2zarr.converter import HDF5ToZarrConverter
from hdf2zarr.models import ConversionConfig


def _walk_and_compare(h5group: h5py.Group, zarr_group: zarr.Group) -> None:
    """Recursively compare HDF5 and Zarr hierarchies for data equality.

    Parameters
    ----------
    h5group : h5py.Group
        The source HDF5 group.
    zarr_group : zarr.Group
        The converted Zarr group.
    """
    for name in h5group:
        item = h5group[name]
        assert name in zarr_group, f"Missing key {name!r} in zarr group"

        if isinstance(item, h5py.Group):
            zarr_item = zarr_group[name]
            assert isinstance(zarr_item, zarr.Group), (
                f"Expected zarr.Group for {name!r}, got {type(zarr_item)}"
            )
            _walk_and_compare(item, zarr_item)

        elif isinstance(item, h5py.Dataset):
            zarr_arr = zarr_group[name]

            # Scalar datasets are stored as 1-element arrays
            if item.shape == ():
                assert zarr_arr.shape == (1,)
                continue

            h5_data = item[:]
            zarr_data = zarr_arr[:]

            # String data: compare as lists
            if item.dtype.kind in ("S", "O") or h5py.check_vlen_dtype(item.dtype):
                h5_list = _to_string_list(h5_data)
                zarr_list = list(zarr_data.flat)
                assert h5_list == zarr_list, (
                    f"String data mismatch at {item.name}: {h5_list} != {zarr_list}"
                )
            else:
                # Numeric data
                np.testing.assert_array_equal(
                    zarr_data,
                    h5_data,
                    err_msg=f"Data mismatch at {item.name}",
                )


def _to_string_list(data: np.ndarray) -> list[str]:
    """Convert HDF5 string data to a flat list of str.

    Parameters
    ----------
    data : np.ndarray
        HDF5 string array (byte strings or object strings).

    Returns
    -------
    list[str]
        Flat list of decoded strings.
    """
    result = []
    for val in data.flat:
        if isinstance(val, bytes):
            result.append(val.decode("utf-8", errors="replace"))
        else:
            result.append(str(val))
    return result


class TestEndToEndConversion:
    """End-to-end tests using the converter directly."""

    def test_simple_roundtrip(self, simple_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        result = converter.convert(simple_hdf5, dest)

        assert result.success
        assert result.datasets_copied == 1
        assert result.groups_copied == 1

        with h5py.File(simple_hdf5, "r") as h5f:
            store = zarr.storage.LocalStore(dest)
            root = zarr.open_group(store, mode="r")
            _walk_and_compare(h5f, root)

    def test_nested_roundtrip(self, nested_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        result = converter.convert(nested_hdf5, dest)

        assert result.success
        assert result.groups_copied == 3

        with h5py.File(nested_hdf5, "r") as h5f:
            store = zarr.storage.LocalStore(dest)
            root = zarr.open_group(store, mode="r")
            _walk_and_compare(h5f, root)

    def test_chunked_compressed_roundtrip(
        self, chunked_compressed_hdf5: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(compression="preserve", chunking="preserve")
        converter = HDF5ToZarrConverter(config)
        result = converter.convert(chunked_compressed_hdf5, dest)

        assert result.success
        assert result.datasets_copied == 2

        with h5py.File(chunked_compressed_hdf5, "r") as h5f:
            store = zarr.storage.LocalStore(dest)
            root = zarr.open_group(store, mode="r")
            _walk_and_compare(h5f, root)

    def test_edge_cases_roundtrip(self, edge_case_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        result = converter.convert(edge_case_hdf5, dest)

        assert result.success
        assert result.datasets_copied >= 8  # all supported datasets

        with h5py.File(edge_case_hdf5, "r") as h5f:
            store = zarr.storage.LocalStore(dest)
            root = zarr.open_group(store, mode="r")
            _walk_and_compare(h5f, root)

    def test_compound_roundtrip(self, compound_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        result = converter.convert(compound_hdf5, dest)

        assert result.success

        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")
        zarr_data = root["points"][:]
        with h5py.File(compound_hdf5, "r") as h5f:
            h5_data = h5f["points"][:]
        np.testing.assert_array_equal(zarr_data, h5_data)

    def test_zarr_v2_roundtrip(self, simple_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(zarr_format=2)
        converter = HDF5ToZarrConverter(config)
        result = converter.convert(simple_hdf5, dest)

        assert result.success

        with h5py.File(simple_hdf5, "r") as h5f:
            store = zarr.storage.LocalStore(dest)
            root = zarr.open_group(store, mode="r")
            _walk_and_compare(h5f, root)

    def test_compression_none(self, simple_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(compression="none")
        converter = HDF5ToZarrConverter(config)
        result = converter.convert(simple_hdf5, dest)

        assert result.success

    def test_chunking_auto(self, chunked_compressed_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(chunking="auto")
        converter = HDF5ToZarrConverter(config)
        result = converter.convert(chunked_compressed_hdf5, dest)

        assert result.success

        with h5py.File(chunked_compressed_hdf5, "r") as h5f:
            store = zarr.storage.LocalStore(dest)
            root = zarr.open_group(store, mode="r")
            _walk_and_compare(h5f, root)

    def test_chunk_preserve_matches_original(
        self, chunked_compressed_hdf5: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(chunking="preserve")
        converter = HDF5ToZarrConverter(config)
        converter.convert(chunked_compressed_hdf5, dest)

        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")

        with h5py.File(chunked_compressed_hdf5, "r") as h5f:
            h5_chunks = h5f["data"].chunks
            zarr_chunks = root["data"].chunks
            assert zarr_chunks == h5_chunks


class TestChunkedCopy:
    """Tests for the chunk-by-chunk copy path for large datasets."""

    def test_large_dataset_chunked_copy(self, tmp_path: Path) -> None:
        """Create a dataset that exceeds MEMORY_THRESHOLD to exercise chunked copy."""
        source = tmp_path / "large.h5"
        # Create a dataset just over the threshold
        # Using float64 (8 bytes): need > 256 MB
        # 256 MB / 8 = 33554432 elements
        # Let's use a smaller threshold for testing by temporarily patching
        import hdf2zarr.converter as conv_module

        original_threshold = conv_module.MEMORY_THRESHOLD
        conv_module.MEMORY_THRESHOLD = 1024  # 1 KB for testing

        try:
            # Create a dataset that's > 1 KB: 200 float64 = 1600 bytes
            data = np.random.default_rng(42).standard_normal((20, 10))
            with h5py.File(source, "w") as f:
                f.create_dataset("big", data=data, chunks=(5, 5))

            dest = tmp_path / "output.zarr"
            config = ConversionConfig(chunking="preserve")
            converter = HDF5ToZarrConverter(config)
            result = converter.convert(source, dest)

            assert result.success
            assert result.datasets_copied == 1

            store = zarr.storage.LocalStore(dest)
            root = zarr.open_group(store, mode="r")
            np.testing.assert_array_equal(root["big"][:], data)
        finally:
            conv_module.MEMORY_THRESHOLD = original_threshold


class TestCLIIntegration:
    """End-to-end tests through the CLI."""

    def test_full_pipeline_via_cli(
        self, multi_file_directory: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "zarr_output"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "convert",
                str(multi_file_directory),
                "-r",
                "-o",
                str(output_dir),
                "--compression",
                "gzip",
                "--compression-level",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output

        # Verify all HDF5 files were converted
        zarr_files = list(output_dir.glob("*.zarr"))
        assert len(zarr_files) >= 3  # file_a, file_b, file_c, nested

        # Verify data integrity for one file
        store = zarr.storage.LocalStore(output_dir / "file_a.zarr")
        root = zarr.open_group(store, mode="r")
        np.testing.assert_array_equal(root["data"][:], [1.0, 2.0])

    def test_skip_on_error_continues(self, tmp_path: Path) -> None:
        # Create a valid file and an invalid file (not actually HDF5)
        valid = tmp_path / "valid.h5"
        with h5py.File(valid, "w") as f:
            f.create_dataset("data", data=np.array([1, 2, 3]))

        invalid = tmp_path / "invalid.h5"
        invalid.write_text("not an hdf5 file")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "convert",
                str(valid),
                str(invalid),
                "-o",
                str(output_dir),
                "--skip-on-error",
            ],
        )
        # Should succeed overall (skip-on-error) but report the failure
        assert "Failed" in result.output or result.exit_code == 1

        # Valid file should still be converted
        assert (output_dir / "valid.zarr").exists()

    def test_fail_on_error_halts(self, tmp_path: Path) -> None:
        invalid = tmp_path / "invalid.h5"
        invalid.write_text("not an hdf5 file")

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "convert",
                str(invalid),
                "-o",
                str(output_dir),
                "--fail-on-error",
            ],
        )
        assert result.exit_code == 1
