# hdf2zarr

[![CI](https://github.com/stebix/hdf2zarr/actions/workflows/ci.yml/badge.svg)](https://github.com/stebix/hdf2zarr/actions/workflows/ci.yml)

Quick python utility to convert HDF5 files to Zarr format.

## Installation

Requires Python 3.13+. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

hdf2zarr provides two commands: `convert` and `info`.

### Converting files

Convert one or more HDF5 files to Zarr:

```bash
# Convert a single file (output goes alongside the source)
hdf2zarr convert experiment.h5

# Convert to a specific output directory
hdf2zarr convert experiment.h5 -o /data/zarr_output

# Convert all HDF5 files in a directory
hdf2zarr convert /data/hdf5_files/ -o /data/zarr_output

# Recursively scan subdirectories
hdf2zarr convert /data/hdf5_files/ -r -o /data/zarr_output

# Preview what would be converted without doing it
hdf2zarr convert /data/hdf5_files/ -r --dry-run
```

### Compression and chunking

```bash
# Use gzip compression instead of the default zstd
hdf2zarr convert data.h5 --compression gzip --compression-level 6

# Preserve the original HDF5 compression (mapped to nearest zarr equivalent)
hdf2zarr convert data.h5 --compression preserve

# No compression
hdf2zarr convert data.h5 --compression none

# Let zarr auto-chunk instead of preserving the original HDF5 chunk layout
hdf2zarr convert data.h5 --chunking auto
```

### Zarr format version

Output defaults to Zarr v3. Use `--zarr-format 2` for compatibility with older tooling:

```bash
hdf2zarr convert data.h5 --zarr-format 2
```

### Overwriting and error handling

```bash
# Overwrite existing zarr stores
hdf2zarr convert data.h5 -o /output --overwrite

# Halt immediately on the first error instead of skipping and continuing
hdf2zarr convert /data/hdf5_files/ --fail-on-error
```

### Inspecting HDF5 files

Render the structure of an HDF5 file as a tree:

```bash
# Show full structure with attributes
hdf2zarr info experiment.h5

# Hide attributes
hdf2zarr info experiment.h5 --hide-attrs

# Limit tree depth
hdf2zarr info experiment.h5 --depth 2
```

### Verbosity and logging

```bash
# Increase verbosity (-v for INFO, -vv for DEBUG)
hdf2zarr convert data.h5 -v

# Suppress all output except errors
hdf2zarr convert data.h5 -q

# Emit structured JSON logs to stderr (useful for log aggregation)
hdf2zarr convert data.h5 --json-log
```

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check src/ tests/
uv run pyright src/
```
