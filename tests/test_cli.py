"""Tests for the CLI interface."""

from pathlib import Path

import pytest
import zarr
import zarr.storage
from click.testing import CliRunner

from hdf2zarr.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


class TestConvertCommand:
    """Tests for the convert subcommand."""

    def test_no_args_shows_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "hdf2zarr" in result.output
        assert "convert" in result.output
        assert "info" in result.output

    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_convert_single_file(
        self, runner: CliRunner, simple_hdf5: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = runner.invoke(
            app, ["convert", str(simple_hdf5), "-o", str(output_dir)]
        )
        assert result.exit_code == 0, result.output
        assert "Converting" in result.output

        zarr_path = output_dir / "simple.zarr"
        assert zarr_path.exists()
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store, mode="r")
        assert "measurements" in root

    def test_convert_directory(
        self, runner: CliRunner, multi_file_directory: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = runner.invoke(
            app, ["convert", str(multi_file_directory), "-o", str(output_dir)]
        )
        assert result.exit_code == 0, result.output
        assert "Found" in result.output

    def test_convert_directory_recursive(
        self, runner: CliRunner, multi_file_directory: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = runner.invoke(
            app, ["convert", str(multi_file_directory), "-r", "-o", str(output_dir)]
        )
        assert result.exit_code == 0, result.output
        zarr_names = {p.name for p in output_dir.iterdir()}
        assert "nested.zarr" in zarr_names

    def test_dry_run(self, runner: CliRunner, simple_hdf5: Path) -> None:
        result = runner.invoke(app, ["convert", str(simple_hdf5), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "Dry Run" in result.output
        zarr_path = simple_hdf5.parent / "simple.zarr"
        assert not zarr_path.exists()

    def test_no_overwrite_refuses(
        self, runner: CliRunner, simple_hdf5: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "simple.zarr").mkdir()

        result = runner.invoke(
            app, ["convert", str(simple_hdf5), "-o", str(output_dir)]
        )
        assert result.exit_code == 1
        assert "already exist" in result.output

    def test_overwrite_succeeds(
        self, runner: CliRunner, simple_hdf5: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = runner.invoke(
            app, ["convert", str(simple_hdf5), "-o", str(output_dir)]
        )
        assert result.exit_code == 0

        result = runner.invoke(
            app, ["convert", str(simple_hdf5), "-o", str(output_dir), "--overwrite"]
        )
        assert result.exit_code == 0

    def test_compression_choice(
        self, runner: CliRunner, simple_hdf5: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = runner.invoke(
            app,
            [
                "convert",
                str(simple_hdf5),
                "-o",
                str(output_dir),
                "--compression",
                "gzip",
            ],
        )
        assert result.exit_code == 0

    def test_invalid_compression_rejected(
        self, runner: CliRunner, simple_hdf5: Path
    ) -> None:
        result = runner.invoke(
            app, ["convert", str(simple_hdf5), "--compression", "invalid"]
        )
        assert result.exit_code != 0

    def test_zarr_format_v2(
        self, runner: CliRunner, simple_hdf5: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = runner.invoke(
            app,
            ["convert", str(simple_hdf5), "-o", str(output_dir), "--zarr-format", "2"],
        )
        assert result.exit_code == 0

    def test_quiet_suppresses_output(
        self, runner: CliRunner, simple_hdf5: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        result = runner.invoke(
            app, ["convert", str(simple_hdf5), "-o", str(output_dir), "-q"]
        )
        assert result.exit_code == 0


class TestInfoCommand:
    """Tests for the info subcommand."""

    def test_info_shows_structure(self, runner: CliRunner, simple_hdf5: Path) -> None:
        result = runner.invoke(app, ["info", str(simple_hdf5)])
        assert result.exit_code == 0
        assert "measurements" in result.output
        assert "temperature" in result.output

    def test_info_shows_attrs(self, runner: CliRunner, simple_hdf5: Path) -> None:
        result = runner.invoke(app, ["info", str(simple_hdf5), "--show-attrs"])
        assert result.exit_code == 0
        assert "file_version" in result.output

    def test_info_hide_attrs(self, runner: CliRunner, simple_hdf5: Path) -> None:
        result = runner.invoke(app, ["info", str(simple_hdf5), "--hide-attrs"])
        assert result.exit_code == 0
        assert "file_version" not in result.output

    def test_info_depth_limit(self, runner: CliRunner, nested_hdf5: Path) -> None:
        result = runner.invoke(app, ["info", str(nested_hdf5), "--depth", "1"])
        assert result.exit_code == 0
        assert "level1" in result.output
        # level2 group itself should appear, but level3 content should be truncated
        assert "..." in result.output

    def test_info_nonexistent_file(self, runner: CliRunner, tmp_path: Path) -> None:
        fake = tmp_path / "nonexistent.h5"
        result = runner.invoke(app, ["info", str(fake)])
        assert result.exit_code != 0
