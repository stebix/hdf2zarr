"""Shared pytest fixtures for hdf2zarr tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def simple_hdf5(tmp_path: Path) -> Path:
    """Create a minimal HDF5 file with one group and one dataset."""
    path = tmp_path / "simple.h5"
    with h5py.File(path, "w") as f:
        f.attrs["file_version"] = "1.0"
        g = f.create_group("measurements")
        g.attrs["instrument"] = "sensor_A"
        ds = g.create_dataset("temperature", data=np.arange(100.0).reshape(10, 10))
        ds.attrs["unit"] = "kelvin"
        ds.attrs["scale_factor"] = 0.01
    return path


@pytest.fixture
def chunked_compressed_hdf5(tmp_path: Path) -> Path:
    """Create an HDF5 file with chunked, gzip-compressed datasets."""
    path = tmp_path / "chunked.h5"
    with h5py.File(path, "w") as f:
        data = np.random.default_rng(42).standard_normal((64, 64))
        f.create_dataset(
            "data",
            data=data,
            chunks=(16, 16),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "labels",
            data=np.arange(64, dtype=np.uint16),
            chunks=(32,),
            compression="gzip",
            compression_opts=6,
        )
    return path


@pytest.fixture
def nested_hdf5(tmp_path: Path) -> Path:
    """Create an HDF5 file with a deeply nested group hierarchy."""
    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as f:
        f.attrs["root_attr"] = "root_value"
        g1 = f.create_group("level1")
        g1.attrs["depth"] = 1
        g2 = g1.create_group("level2")
        g2.attrs["depth"] = 2
        g3 = g2.create_group("level3")
        g3.attrs["depth"] = 3
        g3.create_dataset("deep_data", data=np.array([1, 2, 3]))
    return path


@pytest.fixture
def edge_case_hdf5(tmp_path: Path) -> Path:
    """Create an HDF5 file with various edge-case dtypes and shapes.

    Includes: fixed-length strings, variable-length strings,
    scalar dataset, empty dataset, boolean, complex, integer dtypes.
    """
    path = tmp_path / "edge_cases.h5"
    with h5py.File(path, "w") as f:
        # Fixed-length byte strings
        f.create_dataset("fixed_strings", data=np.array([b"hello", b"world"]))

        # Variable-length strings
        dt = h5py.string_dtype()
        f.create_dataset(
            "vlen_strings", data=np.array(["foo", "bar", "baz"], dtype=object), dtype=dt
        )

        # Scalar dataset
        f.create_dataset("scalar_int", data=42)

        # Scalar string
        f.create_dataset("scalar_string", data=b"scalar_value")

        # Empty dataset
        f.create_dataset("empty", shape=(0, 10), dtype=np.float32)

        # Boolean
        f.create_dataset("flags", data=np.array([True, False, True]))

        # Complex
        f.create_dataset("complex_data", data=np.array([1 + 2j, 3 + 4j]))

        # Multiple integer types
        f.create_dataset("int8_data", data=np.array([1, 2, 3], dtype=np.int8))
        f.create_dataset("uint32_data", data=np.array([100, 200], dtype=np.uint32))

        # Byte attribute
        f.attrs["byte_attr"] = b"bytes_value"
        f.attrs["int_attr"] = 99
        f.attrs["float_array_attr"] = np.array([1.0, 2.0, 3.0])
    return path


@pytest.fixture
def compound_hdf5(tmp_path: Path) -> Path:
    """Create an HDF5 file with a compound (structured) dtype dataset."""
    path = tmp_path / "compound.h5"
    dt = np.dtype([("x", np.float32), ("y", np.float32), ("id", np.int32)])
    data = np.array([(1.0, 2.0, 0), (3.0, 4.0, 1)], dtype=dt)
    with h5py.File(path, "w") as f:
        f.create_dataset("points", data=data)
    return path


@pytest.fixture
def multi_file_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple HDF5 files and non-HDF5 files."""
    for name, ext in [("a", ".h5"), ("b", ".hdf5"), ("c", ".hdf"), ("d", ".txt")]:
        path = tmp_path / f"file_{name}{ext}"
        if ext in (".h5", ".hdf5", ".hdf"):
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=np.array([1.0, 2.0]))
        else:
            path.write_text("not an hdf5 file")

    # Nested subdirectory with another HDF5 file
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    with h5py.File(subdir / "nested.h5", "w") as f:
        f.create_dataset("data", data=np.array([3.0, 4.0]))

    return tmp_path
