"""Tests for HDF5-to-Zarr dtype mapping."""

import h5py
import numpy as np
import pytest

from hdf2zarr.dtypes import (
    is_fixed_length_string,
    is_numeric,
    is_supported,
    is_vlen_string,
    map_dtype,
)
from hdf2zarr.exceptions import UnsupportedDtypeError


class TestIsNumeric:
    """Tests for the is_numeric predicate."""

    @pytest.mark.parametrize(
        "dtype",
        [
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("uint64"),
            np.dtype("float16"),
            np.dtype("float32"),
            np.dtype("float64"),
            np.dtype("complex64"),
            np.dtype("complex128"),
            np.dtype("bool"),
        ],
    )
    def test_numeric_dtypes_are_numeric(self, dtype: np.dtype) -> None:
        assert is_numeric(dtype) is True

    @pytest.mark.parametrize(
        "dtype",
        [np.dtype("S10"), np.dtype("U10"), np.dtype("V8")],
    )
    def test_non_numeric_dtypes(self, dtype: np.dtype) -> None:
        assert is_numeric(dtype) is False


class TestIsFixedLengthString:
    """Tests for fixed-length byte string detection."""

    def test_fixed_string(self) -> None:
        assert is_fixed_length_string(np.dtype("S10")) is True

    def test_not_fixed_string(self) -> None:
        assert is_fixed_length_string(np.dtype("float32")) is False


class TestIsVlenString:
    """Tests for variable-length string detection."""

    def test_vlen_string_dtype(self) -> None:
        dt = h5py.string_dtype()
        assert is_vlen_string(dt) is True

    def test_numeric_not_vlen(self) -> None:
        assert is_vlen_string(np.dtype("float64")) is False

    def test_fixed_string_not_vlen(self) -> None:
        assert is_vlen_string(np.dtype("S10")) is False


class TestIsSupported:
    """Tests for the is_supported predicate."""

    def test_numeric_supported(self) -> None:
        assert is_supported(np.dtype("float32")) is True

    def test_fixed_string_supported(self) -> None:
        assert is_supported(np.dtype("S10")) is True

    def test_vlen_string_supported(self) -> None:
        assert is_supported(h5py.string_dtype()) is True

    def test_all_numeric_compound_supported(self) -> None:
        dt = np.dtype([("x", np.float32), ("y", np.float32)])
        assert is_supported(dt) is True

    def test_void_unsupported(self) -> None:
        assert is_supported(np.dtype("V8")) is False


class TestMapDtype:
    """Tests for the map_dtype function."""

    @pytest.mark.parametrize(
        "dtype",
        [
            np.dtype("int32"),
            np.dtype("float64"),
            np.dtype("complex128"),
            np.dtype("bool"),
        ],
    )
    def test_numeric_passthrough(self, dtype: np.dtype) -> None:
        result = map_dtype(dtype, "/test")
        assert result is dtype

    def test_fixed_string_maps_to_str(self) -> None:
        result = map_dtype(np.dtype("S10"), "/test")
        assert result == "str"

    def test_vlen_string_maps_to_str(self) -> None:
        result = map_dtype(h5py.string_dtype(), "/test")
        assert result == "str"

    def test_numeric_compound_passthrough(self) -> None:
        dt = np.dtype([("x", np.float32), ("y", np.float32), ("id", np.int32)])
        result = map_dtype(dt, "/test")
        assert result is dt

    def test_unsupported_dtype_raises(self) -> None:
        with pytest.raises(UnsupportedDtypeError, match="Unsupported dtype"):
            map_dtype(np.dtype("V8"), "/test")

    def test_unsupported_dtype_stores_path(self) -> None:
        with pytest.raises(UnsupportedDtypeError) as exc_info:
            map_dtype(np.dtype("V8"), "/group/dataset")
        assert exc_info.value.path == "/group/dataset"
