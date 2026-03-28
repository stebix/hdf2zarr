"""Tests for file discovery."""

from pathlib import Path

import pytest

from hdf2zarr.discovery import (
    discover_files,
    parse_patterns,
    resolve_output_path,
)
from hdf2zarr.exceptions import DiscoveryError


class TestParsePatterns:
    """Tests for pattern string parsing."""

    def test_single_pattern(self) -> None:
        assert parse_patterns("*.h5") == ["*.h5"]

    def test_multiple_patterns(self) -> None:
        result = parse_patterns("*.h5,*.hdf5,*.hdf")
        assert result == ["*.h5", "*.hdf5", "*.hdf"]

    def test_strips_whitespace(self) -> None:
        result = parse_patterns(" *.h5 , *.hdf5 ")
        assert result == ["*.h5", "*.hdf5"]

    def test_empty_string(self) -> None:
        assert parse_patterns("") == []

    def test_trailing_comma(self) -> None:
        result = parse_patterns("*.h5,")
        assert result == ["*.h5"]


class TestDiscoverFiles:
    """Tests for the discover_files function."""

    def test_single_file(self, simple_hdf5: Path) -> None:
        result = discover_files([simple_hdf5])
        assert result == [simple_hdf5.resolve()]

    def test_directory_default_patterns(self, multi_file_directory: Path) -> None:
        result = discover_files([multi_file_directory])
        names = {p.name for p in result}
        assert "file_a.h5" in names
        assert "file_b.hdf5" in names
        assert "file_c.hdf" in names
        assert "file_d.txt" not in names

    def test_directory_non_recursive(self, multi_file_directory: Path) -> None:
        result = discover_files([multi_file_directory], recursive=False)
        names = {p.name for p in result}
        # Should not find the nested file
        assert "nested.h5" not in names

    def test_directory_recursive(self, multi_file_directory: Path) -> None:
        result = discover_files([multi_file_directory], recursive=True)
        names = {p.name for p in result}
        assert "nested.h5" in names

    def test_custom_pattern(self, multi_file_directory: Path) -> None:
        result = discover_files([multi_file_directory], patterns=["*.h5"])
        names = {p.name for p in result}
        assert "file_a.h5" in names
        assert "file_b.hdf5" not in names

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = discover_files([tmp_path])
        assert result == []

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        fake = tmp_path / "does_not_exist"
        with pytest.raises(DiscoveryError, match="does not exist"):
            discover_files([fake])

    def test_mixed_files_and_dirs(
        self, simple_hdf5: Path, multi_file_directory: Path
    ) -> None:
        result = discover_files([simple_hdf5, multi_file_directory])
        assert len(result) >= 4  # 1 explicit file + 3 from directory

    def test_deduplication(self, simple_hdf5: Path) -> None:
        result = discover_files([simple_hdf5, simple_hdf5])
        assert len(result) == 1

    def test_sorted_output(self, multi_file_directory: Path) -> None:
        result = discover_files([multi_file_directory])
        assert result == sorted(result)


class TestResolveOutputPath:
    """Tests for output path resolution."""

    def test_alongside_source(self) -> None:
        source = Path("/data/experiment/scan.h5")
        result = resolve_output_path(source, output_dir=None)
        assert result == Path("/data/experiment/scan.zarr")

    def test_custom_output_dir(self) -> None:
        source = Path("/data/experiment/scan.h5")
        result = resolve_output_path(source, output_dir=Path("/output"))
        assert result == Path("/output/scan.zarr")

    def test_hdf5_extension(self) -> None:
        source = Path("/data/file.hdf5")
        result = resolve_output_path(source, output_dir=None)
        assert result == Path("/data/file.zarr")
