"""Core HDF5-to-Zarr conversion logic."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numcodecs
import numpy as np
import structlog
import zarr
import zarr.codecs
import zarr.storage

from hdf2zarr.dtypes import map_dtype
from hdf2zarr.exceptions import ConversionError, UnsupportedDtypeError
from hdf2zarr.models import ConversionConfig, ConversionResult, Timer

if TYPE_CHECKING:
    from hdf2zarr.progress import ProgressTracker

log = structlog.get_logger()

#: Datasets larger than this (in bytes) trigger chunk-by-chunk copy.
MEMORY_THRESHOLD = 256 * 1024 * 1024  # 256 MB


def _build_compressor_v3(codec_name: str, level: int) -> list[Any]:
    """Build zarr v3 native codec compressor list.

    Parameters
    ----------
    codec_name : str
        Compression codec name.
    level : int
        Compression level.

    Returns
    -------
    list
        Codec list for zarr v3 ``compressors`` parameter.
    """
    if codec_name == "none":
        return []
    if codec_name == "zstd":
        return [zarr.codecs.ZstdCodec(level=level)]
    if codec_name == "gzip":
        return [zarr.codecs.GzipCodec(level=level)]
    if codec_name == "lz4":
        return [zarr.codecs.BloscCodec(cname="lz4", clevel=level)]
    return [zarr.codecs.ZstdCodec(level=level)]


def _build_compressor_v2(codec_name: str, level: int) -> Any | None:
    """Build a numcodecs compressor for zarr v2.

    Parameters
    ----------
    codec_name : str
        Compression codec name.
    level : int
        Compression level.

    Returns
    -------
    Any or None
        A numcodecs codec instance, or ``None`` for no compression.
    """
    if codec_name == "none":
        return None
    if codec_name == "zstd":
        return numcodecs.Zstd(level=level)
    if codec_name == "gzip":
        return numcodecs.GZip(level=level)
    if codec_name == "lz4":
        return numcodecs.LZ4()
    return numcodecs.Zstd(level=level)


def build_compressor(
    config: ConversionConfig,
) -> list[Any] | Any | None:
    """Build a zarr compressor from the conversion config.

    Returns a list of v3 codecs when ``zarr_format=3``, or a single
    numcodecs codec (or None) when ``zarr_format=2``.

    Parameters
    ----------
    config : ConversionConfig
        The conversion configuration.

    Returns
    -------
    list or Any or None
        Compressor(s) appropriate for the configured zarr format.
    """
    codec_name = config.compression
    if codec_name == "preserve":
        codec_name = "zstd"

    if config.zarr_format == 2:
        return _build_compressor_v2(codec_name, config.compression_level)
    return _build_compressor_v3(codec_name, config.compression_level)


def _map_hdf5_compression(
    dataset: h5py.Dataset, config: ConversionConfig
) -> list[Any] | Any | None:
    """Map HDF5 dataset compression to zarr codecs.

    When ``config.compression`` is ``"preserve"``, attempts to use the same
    codec as the source dataset. Falls back to zstd if the original codec
    has no direct zarr equivalent.

    Parameters
    ----------
    dataset : h5py.Dataset
        The HDF5 source dataset.
    config : ConversionConfig
        The conversion configuration.

    Returns
    -------
    list or Any or None
        Compressor(s) for the zarr array.
    """
    if config.compression != "preserve":
        return build_compressor(config)

    compression = dataset.compression
    compression_opts = dataset.compression_opts

    if compression is None:
        return [] if config.zarr_format == 3 else None

    if compression == "gzip":
        level = compression_opts if isinstance(compression_opts, int) else 4
        if config.zarr_format == 2:
            return numcodecs.GZip(level=level)
        return [zarr.codecs.GzipCodec(level=level)]

    if compression == "lzf":
        log.info(
            "lzf_fallback_to_lz4",
            path=dataset.name,
            reason="lzf has no zarr v3 equivalent",
        )
        if config.zarr_format == 2:
            return numcodecs.LZ4()
        return [zarr.codecs.BloscCodec(cname="lz4", clevel=5)]

    if compression == "szip":
        log.info(
            "szip_fallback_to_zstd",
            path=dataset.name,
            reason="szip has no zarr v3 equivalent",
        )
        if config.zarr_format == 2:
            return numcodecs.Zstd(level=config.compression_level)
        return [zarr.codecs.ZstdCodec(level=config.compression_level)]

    # Unknown compression; fall back to zstd
    log.warning(
        "unknown_compression_fallback", path=dataset.name, compression=compression
    )
    if config.zarr_format == 2:
        return numcodecs.Zstd(level=config.compression_level)
    return [zarr.codecs.ZstdCodec(level=config.compression_level)]


def _resolve_chunks(
    dataset: h5py.Dataset, config: ConversionConfig
) -> tuple[int, ...] | None:
    """Determine output chunk shape for a dataset.

    Parameters
    ----------
    dataset : h5py.Dataset
        The HDF5 source dataset.
    config : ConversionConfig
        The conversion configuration.

    Returns
    -------
    tuple[int, ...] or None
        Chunk shape, or ``None`` to let zarr auto-chunk.
    """
    if config.chunking == "auto":
        return None

    # "preserve" strategy
    if dataset.chunks is not None:
        return dataset.chunks

    # Contiguous dataset: let zarr decide
    return None


def _create_zarr_array(
    zarr_group: zarr.Group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic] | str,
    chunks: tuple[int, ...] | None,
    compressors: list[Any] | Any | None,
    zarr_format: int,
) -> zarr.Array:
    """Create a zarr array with format-aware compressor handling.

    Parameters
    ----------
    zarr_group : zarr.Group
        Parent group.
    name : str
        Array name.
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype or str
        Array dtype.
    chunks : tuple[int, ...] or None
        Chunk shape. ``None`` lets zarr auto-chunk.
    compressors : list or Any or None
        Compressor(s) — a v3 codec list or a v2 numcodecs codec.
    zarr_format : int
        Zarr format version (2 or 3).

    Returns
    -------
    zarr.Array
        The created array.
    """
    kwargs: dict[str, Any] = {
        "shape": shape,
        "dtype": dtype,
    }

    if chunks is not None:
        kwargs["chunks"] = chunks

    if zarr_format == 2:
        kwargs["compressor"] = compressors
    else:
        if compressors is None:
            kwargs["compressors"] = []
        else:
            kwargs["compressors"] = compressors

    return zarr_group.create_array(name, **kwargs)


def _copy_attributes(
    source: h5py.Group | h5py.Dataset, target: zarr.Group | zarr.Array
) -> int:
    """Copy HDF5 attributes to a zarr group or array.

    Performs type conversions for numpy scalars, arrays, and byte strings.
    Object references are skipped with a warning.

    Parameters
    ----------
    source : h5py.Group or h5py.Dataset
        The HDF5 object to copy attributes from.
    target : zarr.Group or zarr.Array
        The zarr object to copy attributes to.

    Returns
    -------
    int
        Number of attributes successfully copied.
    """
    copied = 0
    for key, value in source.attrs.items():
        try:
            converted = _convert_attr_value(value)
            if converted is _SKIP_SENTINEL:
                log.debug(
                    "skipping_attr",
                    path=source.name,
                    key=key,
                    reason="unsupported type",
                )
                continue
            target.attrs[key] = converted  # type: ignore[assignment]
            copied += 1
        except Exception as exc:
            log.warning("attr_copy_failed", path=source.name, key=key, error=str(exc))
    return copied


_SKIP_SENTINEL = object()


def _convert_attr_value(value: object) -> object:
    """Convert an HDF5 attribute value to a JSON-serializable Python type.

    Parameters
    ----------
    value : object
        The attribute value from HDF5.

    Returns
    -------
    object
        A Python-native value, or ``_SKIP_SENTINEL`` if the value should
        be skipped.
    """
    if isinstance(value, h5py.Reference):
        return _SKIP_SENTINEL

    if isinstance(value, np.void):
        # Compound scalar: convert to dict
        dtype = value.dtype
        if dtype.names is not None:
            return {name: _convert_attr_value(value[name]) for name in dtype.names}
        return _SKIP_SENTINEL

    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            # Object array (e.g., references)
            return _SKIP_SENTINEL
        if value.dtype.kind == "S":
            # Byte string array -> list of str
            return [v.decode("utf-8", errors="replace") for v in value.flat]
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    # Already a Python native (str, int, float, list, etc.)
    return value


def _iter_chunk_slices(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> list[tuple[slice, ...]]:
    """Generate slice tuples covering a dataset in chunk-sized blocks.

    Parameters
    ----------
    shape : tuple[int, ...]
        The dataset shape.
    chunks : tuple[int, ...]
        The chunk shape.

    Returns
    -------
    list[tuple[slice, ...]]
        A list of slice tuples, each covering one chunk.
    """
    import itertools

    ranges = []
    for dim_size, chunk_size in zip(shape, chunks):
        starts = range(0, dim_size, chunk_size)
        ranges.append([(s, min(s + chunk_size, dim_size)) for s in starts])

    slices = []
    for combo in itertools.product(*ranges):
        slices.append(tuple(slice(start, stop) for start, stop in combo))
    return slices


def _count_datasets(group: h5py.Group) -> int:
    """Count the total number of datasets in an HDF5 group tree.

    Parameters
    ----------
    group : h5py.Group
        Root group to count from.

    Returns
    -------
    int
        Total dataset count.
    """
    count = 0

    def _visitor(name: str, obj: object) -> None:
        nonlocal count
        if isinstance(obj, h5py.Dataset):
            count += 1

    group.visititems(_visitor)
    return count


class HDF5ToZarrConverter:
    """Converts a single HDF5 file to a Zarr store.

    Parameters
    ----------
    config : ConversionConfig
        Conversion settings (compression, chunking, etc.).
    progress : ProgressTracker or None
        Optional progress tracker implementing the ``ProgressTracker``
        protocol.
    """

    def __init__(
        self, config: ConversionConfig, progress: "ProgressTracker | None" = None
    ) -> None:
        self.config = config
        self.progress = progress

    def convert(self, source: Path, destination: Path) -> ConversionResult:
        """Convert a single HDF5 file to a Zarr store.

        Parameters
        ----------
        source : Path
            Path to the source HDF5 file.
        destination : Path
            Path for the output Zarr store directory.

        Returns
        -------
        ConversionResult
            The outcome of the conversion including stats and errors.

        Raises
        ------
        ConversionError
            If the source file cannot be opened.
        """
        result = ConversionResult(source=source, destination=destination)

        with Timer() as timer:
            try:
                with h5py.File(source, "r") as h5file:
                    # Notify progress tracker
                    if self.progress is not None:
                        total_ds = _count_datasets(h5file)
                        self.progress.start_file(source, total_ds)

                    store = zarr.storage.LocalStore(destination)
                    zarr_format = self.config.zarr_format  # 2 or 3
                    zarr_root = zarr.open_group(
                        store,
                        mode="w",
                        zarr_format=zarr_format,  # type: ignore[arg-type]
                    )
                    self._walk_hdf5(h5file, zarr_root, result, path="/")

                    if self.progress is not None:
                        self.progress.finish_file()
            except OSError as exc:
                raise ConversionError(str(source), str(exc)) from exc

        result.elapsed = timer.elapsed

        if result.datasets_skipped > 0 and result.datasets_copied > 0:
            result.success = True  # partial success
        elif result.datasets_skipped > 0 and result.datasets_copied == 0:
            result.success = False

        log.info(
            "conversion_complete",
            source=str(source),
            destination=str(destination),
            groups=result.groups_copied,
            datasets=result.datasets_copied,
            skipped=result.datasets_skipped,
            bytes=result.bytes_copied,
            elapsed=f"{result.elapsed:.2f}s",
        )

        return result

    def _walk_hdf5(
        self,
        h5group: h5py.Group,
        zarr_group: zarr.Group,
        result: ConversionResult,
        path: str,
    ) -> None:
        """Recursively walk HDF5 groups and copy to zarr.

        Parameters
        ----------
        h5group : h5py.Group
            Current HDF5 group.
        zarr_group : zarr.Group
            Corresponding zarr group.
        result : ConversionResult
            Accumulates conversion statistics.
        path : str
            Current path in the HDF5 hierarchy.
        """
        _copy_attributes(h5group, zarr_group)

        for name in h5group:
            item_path = f"{path}{name}" if path.endswith("/") else f"{path}/{name}"

            try:
                item = h5group[name]
            except Exception as exc:
                log.warning("cannot_access_item", path=item_path, error=str(exc))
                result.errors.append(f"Cannot access {item_path}: {exc}")
                continue

            if isinstance(item, h5py.Group):
                log.debug("copying_group", path=item_path)
                zg = zarr_group.create_group(name)  # type: ignore[arg-type]
                result.groups_copied += 1
                self._walk_hdf5(item, zg, result, path=item_path)

            elif isinstance(item, h5py.Dataset):
                self._convert_dataset(item, zarr_group, name, item_path, result)  # type: ignore[arg-type]
                if self.progress is not None:
                    self.progress.advance_dataset()

            else:
                log.warning(
                    "skipping_unsupported_item",
                    path=item_path,
                    type=type(item).__name__,
                )
                result.errors.append(f"Skipped unsupported item {item_path}")

    def _convert_dataset(
        self,
        dataset: h5py.Dataset,
        zarr_group: zarr.Group,
        name: str,
        path: str,
        result: ConversionResult,
    ) -> None:
        """Convert a single HDF5 dataset to a zarr array.

        Parameters
        ----------
        dataset : h5py.Dataset
            The HDF5 dataset.
        zarr_group : zarr.Group
            Parent zarr group to create the array in.
        name : str
            Name for the new zarr array.
        path : str
            Full HDF5 path (for logging).
        result : ConversionResult
            Accumulates conversion statistics.
        """
        try:
            mapped_dtype = map_dtype(dataset.dtype, path)
        except UnsupportedDtypeError as exc:
            log.warning("skipping_dataset", path=path, reason=str(exc))
            result.errors.append(str(exc))
            result.datasets_skipped += 1
            return

        chunks = _resolve_chunks(dataset, self.config)
        compressors = _map_hdf5_compression(dataset, self.config)
        shape = dataset.shape

        log.debug(
            "converting_dataset",
            path=path,
            shape=shape,
            dtype=str(mapped_dtype),
            chunks=chunks,
        )

        # Handle scalar datasets
        if shape == ():
            self._convert_scalar_dataset(
                dataset, zarr_group, name, path, mapped_dtype, result
            )
            return

        # Handle empty (zero-size) datasets
        if dataset.size == 0:
            _create_zarr_array(
                zarr_group,
                name,
                shape=shape,
                dtype=mapped_dtype,
                chunks=chunks,
                compressors=compressors,
                zarr_format=self.config.zarr_format,
            )
            _copy_attributes(dataset, zarr_group[name])
            result.datasets_copied += 1
            return

        # Create zarr array
        zarr_array = _create_zarr_array(
            zarr_group,
            name,
            shape=shape,
            dtype=mapped_dtype,
            chunks=chunks,
            compressors=compressors,
            zarr_format=self.config.zarr_format,
        )

        # Copy data
        nbytes = dataset.nbytes if hasattr(dataset, "nbytes") else 0
        is_string = isinstance(mapped_dtype, str) and mapped_dtype == "str"

        if is_string:
            self._copy_string_data(dataset, zarr_array, path)
        elif nbytes > MEMORY_THRESHOLD and chunks is not None:
            self._copy_chunked(dataset, zarr_array, chunks, path)
        else:
            zarr_array[:] = dataset[:]

        _copy_attributes(dataset, zarr_array)
        result.datasets_copied += 1
        result.bytes_copied += nbytes

    def _convert_scalar_dataset(
        self,
        dataset: h5py.Dataset,
        zarr_group: zarr.Group,
        name: str,
        path: str,
        mapped_dtype: np.dtype[np.generic] | str,
        result: ConversionResult,
    ) -> None:
        """Convert a scalar HDF5 dataset.

        Parameters
        ----------
        dataset : h5py.Dataset
            The scalar HDF5 dataset.
        zarr_group : zarr.Group
            Parent zarr group.
        name : str
            Array name.
        path : str
            Full HDF5 path for logging.
        mapped_dtype : np.dtype or str
            The mapped zarr dtype.
        result : ConversionResult
            Accumulates stats.
        """
        value = dataset[()]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.generic):
            value = value.item()

        zarr_array = _create_zarr_array(
            zarr_group,
            name,
            shape=(1,),
            dtype=mapped_dtype,
            chunks=(1,),
            compressors=build_compressor(self.config),
            zarr_format=self.config.zarr_format,
        )
        zarr_array[0] = value
        _copy_attributes(dataset, zarr_array)
        # Store a marker attribute so readers know this was originally scalar
        zarr_array.attrs["_hdf2zarr_scalar"] = True
        result.datasets_copied += 1
        log.debug("converted_scalar", path=path, value=repr(value))

    def _copy_string_data(
        self,
        dataset: h5py.Dataset,
        zarr_array: zarr.Array,
        path: str,
    ) -> None:
        """Copy string data from HDF5 to zarr, decoding bytes to str.

        Parameters
        ----------
        dataset : h5py.Dataset
            HDF5 source dataset.
        zarr_array : zarr.Array
            Zarr target array.
        path : str
            HDF5 path for logging.
        """
        data = dataset[:]
        if isinstance(data, np.ndarray) and data.dtype.kind == "S":
            decoded = np.array(
                [v.decode("utf-8", errors="replace") for v in data.flat],
            ).reshape(data.shape)
            zarr_array[:] = decoded
        elif isinstance(data, np.ndarray) and data.dtype.kind == "O":
            # Variable-length strings from h5py come as object arrays of str/bytes
            converted = np.array(
                [
                    v.decode("utf-8", errors="replace")
                    if isinstance(v, bytes)
                    else str(v)
                    for v in data.flat
                ],
            ).reshape(data.shape)
            zarr_array[:] = converted
        else:
            zarr_array[:] = data
        log.debug("copied_string_data", path=path, shape=dataset.shape)

    def _copy_chunked(
        self,
        dataset: h5py.Dataset,
        zarr_array: zarr.Array,
        chunks: tuple[int, ...],
        path: str,
    ) -> None:
        """Copy a large dataset chunk by chunk to limit memory usage.

        Parameters
        ----------
        dataset : h5py.Dataset
            HDF5 source dataset.
        zarr_array : zarr.Array
            Zarr target array.
        chunks : tuple[int, ...]
            Chunk shape used for iteration.
        path : str
            HDF5 path for logging.
        """
        chunk_slices = _iter_chunk_slices(dataset.shape, chunks)
        log.debug("chunked_copy_start", path=path, n_chunks=len(chunk_slices))

        for slices in chunk_slices:
            zarr_array[slices] = dataset[slices]

        log.debug("chunked_copy_complete", path=path)
