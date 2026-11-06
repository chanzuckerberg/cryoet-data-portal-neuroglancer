# cryoet-data-portal-neuroglancer

CryoET Data Portal Neuroglancer configuration helper

## Installation

Running the following commands will clone the repository and install the required dependencies.

```bash
git clone https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer.git
cd cryoet-data-portal-neuroglancer
poetry install
```

To specify this as a dependency in a python project with poetry, add the following to your `pyproject.toml` file:

```
cryoet-data-portal-neuroglancer = { git = "https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer.git", branch = "main" }
```

As the main branch is not always stable, it is recommended to specify a commit hash or use a tag instead of a branch name.

## OME-Zarr input

`cryoet_data_portal_neuroglancer.io` reads OME-Zarr images with zarr-python 3: Zarr v2 / OME-NGFF 0.4 and
Zarr v3 / OME-NGFF 0.5 (including sharded arrays), from local paths or URLs (`s3://`, `https://`; pass
`storage_options`, e.g. `{"anon": True}`, for fsspec).

```python
from cryoet_data_portal_neuroglancer.io import load_omezarr_data, open_omezarr

image = open_omezarr("s3://bucket/run/tomogram.zarr", storage_options={"anon": True})
image.axes, image.scale(0), image.scale_factor, image.add_offset

codes = load_omezarr_data("s3://bucket/run/tomogram.zarr", resolution_level=-1, persist=False,
                          storage_options={"anon": True})               # stored values, 3-D
values = load_omezarr_data(..., decode=True)                              # CF-decoded float32
```

- `squeeze=True` (default) drops `t` and `c` axes of size 1, so 5-D `[t, c, z, y, x]` volumes load as 3-D.
- Sharded arrays are read in shard-aligned dask blocks (`chunks="auto"`).
- Volumes stored as integer codes with netCDF/CF `scale_factor`/`add_offset` are returned as codes unless
  `decode=True`. Neuroglancer also displays the codes, so contrast limits for such layers must be in code
  units: `precompute.contrast_limits.physical_to_code` converts, and `contrast_limits_from_statistics`
  derives limits from stored image statistics.
- `state_generator.generate_image_layer(..., extra_axes=("t", "c"))` maps the `t` and `c` axes of 5-D
  sources to local layer dimensions; `zarr_version=3` emits `zarr3://` instead of `zarr://`.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.
Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).
