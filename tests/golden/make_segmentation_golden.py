"""Record the output of ``encode_segmentation`` of a released version as golden hashes.

Run it in an environment with the *reference* release installed (v1.7.2: zarr 2, ome-zarr 0.9, numpy 1.26):

    python tests/golden/make_segmentation_golden.py tests/golden/segmentation_v1.7.2.json

``tests/test_segmentation_golden.py`` encodes the same inputs, stored as Zarr v2 / OME-NGFF 0.4 and as
Zarr v3 / OME-NGFF 0.5, and requires byte-identical output.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# shape (z, y, x), zarr chunks, encode_segmentation keyword arguments
CASES = {
    "labels_default": ((40, 53, 70), (32, 32, 32), {}),
    "binary_blocks16": ((33, 64, 17), (16, 32, 16), {"block_size": (16, 16, 16), "convert_non_zero_to": 1}),
    "instance_uint16": ((24, 30, 36), (24, 30, 36), {"block_size": (8, 8, 8)}),
}


def case_data(name: str) -> np.ndarray:
    """Deterministic label volumes (no RNG, so every numpy version builds the same array)."""
    shape = CASES[name][0]
    z, y, x = np.indices(shape)
    if name == "labels_default":
        return (((3 * z + 5 * y + 7 * x) // 11) % 4).astype(np.uint8) * ((z + y) % 5 != 0)
    if name == "binary_blocks16":
        return (((z - 16) ** 2 + (y - 30) ** 2 + (x - 8) ** 2) < 120).astype(np.int8)
    labels = 1 + (z // 6) * 100 + (y // 7) * 10 + (x // 9)
    return (labels * ((x + y + z) % 3 != 0)).astype(np.uint16)


def output_hashes(out: Path) -> dict[str, str]:
    return {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }


def write_v2_ome_zarr_legacy(path: Path, data: np.ndarray, chunks: tuple[int, ...]) -> None:
    """Write a Zarr v2 / OME-NGFF 0.4 image with zarr-python 2 (the reference environment)."""
    import zarr

    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("0", data=data, chunks=chunks, dimension_separator="/")
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "angstrom"} for a in "zyx"],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
        },
    ]


def main(dest: str) -> None:
    from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import encode_segmentation

    golden = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (_, chunks, kwargs) in CASES.items():
            src = Path(tmp) / f"{name}.zarr"
            out = Path(tmp) / f"{name}_out"
            write_v2_ome_zarr_legacy(src, case_data(name), chunks)
            encode_segmentation(str(src), out, resolution=(1.0, 1.0, 1.0), **kwargs)
            golden[name] = output_hashes(out)
    Path(dest).write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print({k: len(v) for k, v in golden.items()})


if __name__ == "__main__":
    main(sys.argv[1])
