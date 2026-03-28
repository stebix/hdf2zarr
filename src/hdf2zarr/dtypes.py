"""HDF5-to-Zarr dtype mapping and validation.

Handles the translation of HDF5 dataset dtypes to Zarr-compatible numpy
dtypes, including edge cases like variable-length strings, fixed-length
byte strings, and compound types.
"""

import h5py
import numpy as np
import structlog

from hdf2zarr.exceptions import UnsupportedDtypeError

log = structlog.get_logger()

# HDF5 special dtype kinds (from h5py)
_VLEN_STRING_KIND = h5py.string_dtype().metadata

_NUMERIC_KINDS = frozenset(
    {
        "i",  # signed integer
        "u",  # unsigned integer
        "f",  # floating-point
        "c",  # complex
        "b",  # boolean
    }
)


def is_vlen_string(dtype: np.dtype[np.void]) -> bool:
    """Check if a dtype represents an HDF5 variable-length string.

    Parameters
    ----------
    dtype : np.dtype
        The numpy dtype to check.

    Returns
    -------
    bool
        ``True`` if the dtype is a variable-length string type.
    """
    vlen_meta = h5py.check_vlen_dtype(dtype)
    return vlen_meta is not None and vlen_meta in (str, bytes)


def is_fixed_length_string(dtype: np.dtype[np.bytes_]) -> bool:
    """Check if a dtype represents an HDF5 fixed-length byte string.

    Parameters
    ----------
    dtype : np.dtype
        The numpy dtype to check.

    Returns
    -------
    bool
        ``True`` if the dtype is a fixed-length string (``|Sn``).
    """
    return dtype.kind == "S"


def is_numeric(dtype: np.dtype[np.generic]) -> bool:
    """Check if a dtype is a numeric type passable directly to Zarr.

    Parameters
    ----------
    dtype : np.dtype
        The numpy dtype to check.

    Returns
    -------
    bool
        ``True`` for integer, float, complex, and boolean dtypes.
    """
    return dtype.kind in _NUMERIC_KINDS


def is_supported(dtype: np.dtype[np.generic]) -> bool:
    """Check whether an HDF5 dtype can be converted to Zarr.

    Parameters
    ----------
    dtype : np.dtype
        The numpy dtype to check.

    Returns
    -------
    bool
        ``True`` if the dtype is supported for conversion.
    """
    if is_numeric(dtype):
        return True
    if is_fixed_length_string(dtype):  # type: ignore[arg-type]
        return True
    if is_vlen_string(dtype):  # type: ignore[arg-type]
        return True
    # Compound types with all-numeric fields
    if dtype.names is not None:
        return all(is_numeric(dtype.fields[name][0]) for name in dtype.names)  # type: ignore[union-attr,index]
    return False


def map_dtype(dtype: np.dtype[np.generic], path: str) -> np.dtype[np.generic] | str:
    """Map an HDF5 dtype to a Zarr-compatible dtype.

    Parameters
    ----------
    dtype : np.dtype
        The HDF5 dataset's numpy dtype.
    path : str
        HDF5 path of the dataset (used for error messages).

    Returns
    -------
    np.dtype or str
        The Zarr-compatible dtype. Returns the string ``"str"`` for
        variable-length and fixed-length string types.

    Raises
    ------
    UnsupportedDtypeError
        If the dtype cannot be mapped.
    """
    if is_numeric(dtype):
        return dtype

    if is_fixed_length_string(dtype):  # type: ignore[arg-type]
        log.debug("mapping_fixed_string_to_str", path=path, original_dtype=str(dtype))
        return "str"  # type: ignore[return-value]

    if is_vlen_string(dtype):  # type: ignore[arg-type]
        log.debug("mapping_vlen_string_to_str", path=path)
        return "str"  # type: ignore[return-value]

    # Compound types with all-numeric fields: pass through as structured dtype
    if dtype.names is not None:
        if all(is_numeric(dtype.fields[name][0]) for name in dtype.names):  # type: ignore[union-attr,index]
            log.debug("passing_through_compound_dtype", path=path, dtype=str(dtype))
            return dtype
        raise UnsupportedDtypeError(dtype, path)

    raise UnsupportedDtypeError(dtype, path)
