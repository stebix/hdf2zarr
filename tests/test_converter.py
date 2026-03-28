"""Tests for the HDF5-to-Zarr converter."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
import zarr.storage

from hdf2zarr.converter import (
    HDF5ToZarrConverter,
    _convert_attr_value,
    _iter_chunk_slices,
    _resolve_chunks,
    build_compressor,
)
from hdf2zarr.exceptions import ConversionError
from hdf2zarr.models import ConversionConfig


class TestBuildCompressor:
    """Tests for compressor construction."""

    def test_zstd_default(self) -> None:
        config = ConversionConfig(compression="zstd", compression_level=3)
        codecs = build_compressor(config)
        assert len(codecs) == 1
        assert type(codecs[0]).__name__ == "ZstdCodec"

    def test_gzip(self) -> None:
        config = ConversionConfig(compression="gzip", compression_level=6)
        codecs = build_compressor(config)
        assert len(codecs) == 1
        assert type(codecs[0]).__name__ == "GzipCodec"

    def test_lz4(self) -> None:
        config = ConversionConfig(compression="lz4", compression_level=5)
        codecs = build_compressor(config)
        assert len(codecs) == 1
        assert type(codecs[0]).__name__ == "BloscCodec"

    def test_none_returns_empty(self) -> None:
        config = ConversionConfig(compression="none")
        codecs = build_compressor(config)
        assert codecs == []


class TestResolveChunks:
    """Tests for chunk resolution logic."""

    def test_auto_returns_none(self, simple_hdf5: Path) -> None:
        config = ConversionConfig(chunking="auto")
        with h5py.File(simple_hdf5, "r") as f:
            ds = f["measurements/temperature"]
            assert _resolve_chunks(ds, config) is None

    def test_preserve_chunked(self, chunked_compressed_hdf5: Path) -> None:
        config = ConversionConfig(chunking="preserve")
        with h5py.File(chunked_compressed_hdf5, "r") as f:
            ds = f["data"]
            chunks = _resolve_chunks(ds, config)
            assert chunks == (16, 16)

    def test_preserve_contiguous_returns_none(self, simple_hdf5: Path) -> None:
        config = ConversionConfig(chunking="preserve")
        with h5py.File(simple_hdf5, "r") as f:
            ds = f["measurements/temperature"]
            # Contiguous dataset: should fall back to auto (None)
            assert _resolve_chunks(ds, config) is None


class TestConvertAttrValue:
    """Tests for attribute value conversion."""

    def test_int(self) -> None:
        assert _convert_attr_value(42) == 42

    def test_str(self) -> None:
        assert _convert_attr_value("hello") == "hello"

    def test_float(self) -> None:
        assert _convert_attr_value(3.14) == 3.14

    def test_numpy_scalar(self) -> None:
        result = _convert_attr_value(np.float64(2.5))
        assert result == 2.5
        assert isinstance(result, float)

    def test_numpy_array(self) -> None:
        result = _convert_attr_value(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_bytes_to_str(self) -> None:
        result = _convert_attr_value(b"hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_byte_string_array(self) -> None:
        result = _convert_attr_value(np.array([b"a", b"b"]))
        assert result == ["a", "b"]

    def test_0d_array(self) -> None:
        result = _convert_attr_value(np.array(5))
        assert result == 5
        assert isinstance(result, int)


class TestIterChunkSlices:
    """Tests for chunk slice generation."""

    def test_1d(self) -> None:
        slices = _iter_chunk_slices((10,), (3,))
        assert len(slices) == 4  # [0:3], [3:6], [6:9], [9:10]
        assert slices[0] == (slice(0, 3),)
        assert slices[-1] == (slice(9, 10),)

    def test_2d(self) -> None:
        slices = _iter_chunk_slices((6, 4), (3, 2))
        assert len(slices) == 4  # 2 * 2

    def test_exact_fit(self) -> None:
        slices = _iter_chunk_slices((9,), (3,))
        assert len(slices) == 3


class TestHDF5ToZarrConverter:
    """Tests for the full converter class."""

    def test_simple_conversion(self, simple_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig()
        converter = HDF5ToZarrConverter(config)
        result = converter.convert(simple_hdf5, dest)

        assert result.success is True
        assert result.groups_copied == 1
        assert result.datasets_copied == 1
        assert result.datasets_skipped == 0
        assert result.bytes_copied > 0
        assert result.elapsed > 0.0

        # Verify zarr output
        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")
        assert "measurements" in root
        assert "temperature" in root["measurements"]
        arr = root["measurements/temperature"]
        np.testing.assert_array_equal(arr[:], np.arange(100.0).reshape(10, 10))

    def test_attributes_preserved(self, simple_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        converter.convert(simple_hdf5, dest)

        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")
        assert root.attrs["file_version"] == "1.0"
        assert root["measurements"].attrs["instrument"] == "sensor_A"
        assert root["measurements/temperature"].attrs["unit"] == "kelvin"
        assert root["measurements/temperature"].attrs["scale_factor"] == 0.01

    def test_chunked_compressed(
        self, chunked_compressed_hdf5: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(compression="gzip", compression_level=4)
        converter = HDF5ToZarrConverter(config)
        result = converter.convert(chunked_compressed_hdf5, dest)

        assert result.success is True
        assert result.datasets_copied == 2

        # Verify data integrity
        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")
        with h5py.File(chunked_compressed_hdf5, "r") as f:
            np.testing.assert_array_almost_equal(root["data"][:], f["data"][:])
            np.testing.assert_array_equal(root["labels"][:], f["labels"][:])

    def test_nested_groups(self, nested_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        result = converter.convert(nested_hdf5, dest)

        assert result.groups_copied == 3
        assert result.datasets_copied == 1

        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")
        assert root.attrs["root_attr"] == "root_value"
        assert root["level1"].attrs["depth"] == 1
        assert root["level1/level2"].attrs["depth"] == 2
        assert root["level1/level2/level3"].attrs["depth"] == 3
        np.testing.assert_array_equal(
            root["level1/level2/level3/deep_data"][:], [1, 2, 3]
        )

    def test_edge_cases(self, edge_case_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        converter.convert(edge_case_hdf5, dest)

        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")

        # Boolean
        np.testing.assert_array_equal(root["flags"][:], [True, False, True])

        # Complex
        np.testing.assert_array_equal(root["complex_data"][:], [1 + 2j, 3 + 4j])

        # Integer types
        np.testing.assert_array_equal(root["int8_data"][:], [1, 2, 3])
        np.testing.assert_array_equal(root["uint32_data"][:], [100, 200])

        # Empty dataset
        assert root["empty"].shape == (0, 10)

        # Fixed-length strings -> str
        assert list(root["fixed_strings"][:]) == ["hello", "world"]

        # Variable-length strings -> str
        assert list(root["vlen_strings"][:]) == ["foo", "bar", "baz"]

        # Scalar -> 1-element array
        assert root["scalar_int"][0] == 42

        # Byte attributes decoded
        assert root.attrs["byte_attr"] == "bytes_value"
        assert root.attrs["int_attr"] == 99

    def test_compound_dtype(self, compound_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        result = converter.convert(compound_hdf5, dest)

        assert result.success is True
        assert result.datasets_copied == 1

        store = zarr.storage.LocalStore(dest)
        root = zarr.open_group(store, mode="r")
        arr = root["points"][:]
        assert arr.dtype.names == ("x", "y", "id")
        np.testing.assert_array_almost_equal(arr["x"], [1.0, 3.0])

    def test_zarr_format_v2(self, simple_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(zarr_format=2)
        converter = HDF5ToZarrConverter(config)
        result = converter.convert(simple_hdf5, dest)
        assert result.success is True

    def test_nonexistent_source_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "nonexistent.h5"
        dest = tmp_path / "output.zarr"
        converter = HDF5ToZarrConverter(ConversionConfig())
        with pytest.raises(ConversionError, match="nonexistent.h5"):
            converter.convert(source, dest)

    def test_overwrite_existing(self, simple_hdf5: Path, tmp_path: Path) -> None:
        dest = tmp_path / "output.zarr"
        config = ConversionConfig(overwrite=True)
        converter = HDF5ToZarrConverter(config)

        # Convert twice — second should succeed due to overwrite
        converter.convert(simple_hdf5, dest)
        result = converter.convert(simple_hdf5, dest)
        assert result.success is True
