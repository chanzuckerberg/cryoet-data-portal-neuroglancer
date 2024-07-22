import dask.array as da
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def load_omezarr_data(input_filepath: str, resolution_level: int = 0) -> da.Array:
    """Load the OME-Zarr data and return a dask array

    Parameters
    ----------
        input_filepath: str
            Path to the OME-Zarr file.
        resolution_level: int, optional
            Resolution level to load.
            By default 0 - the highest resolution.
    """
    url = parse_url(input_filepath)
    if not url:
        raise ValueError(f"Input file {input_filepath} is not a ZARR file")
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[resolution_level]
    return dask_data.persist()
