import numpy as np
import pytest

from cryoet_data_portal_neuroglancer.state_generator import (
    combine_json_layers,
    generate_image_layer,
    generate_image_volume_layer,
    generate_oriented_point_layer,
    generate_point_layer,
    generate_segmentation_mask_layer,
)


def test__generate_image_layer_default_values():
    state = generate_image_layer(source="mysource", scale=1.5, size={"a": 2.0})

    assert "blend" in state
    assert state["blend"] == "additive"
    assert state["opacity"] == 1.0
    assert "codeVisible" in state
    assert state["codeVisible"] is False


def test__generate_segmentation_layer_default_values():
    state = generate_segmentation_mask_layer(source="mysource")

    assert "pick" in state
    assert state["pick"] is False


def test__generate_configuration_default_values():
    state = combine_json_layers(layers=[{"type": "image", "volumeRendering": "OK", "name": "myname"}], scale=1.0)

    assert "toolPalettes" in state
    assert len(state["toolPalettes"]) == 1

    palette = state["toolPalettes"]["Position"]
    assert palette.get("side") == "bottom"
    assert palette.get("row") == 1
    assert palette.get("query") == "type:dimension"
    assert palette.get("size") == 100
    assert palette.get("visible")
    assert not palette.get("verticalStacking")


def test__generate_oriented_point_layer_default_values():
    state = generate_oriented_point_layer(source="mysource")

    assert "codeVisible" in state
    assert state["codeVisible"] is False


def test__generate_point_layer_default_values():
    state = generate_point_layer(source="mysource")

    assert "codeVisible" in state
    assert state["codeVisible"] is False


def np_is_close(array1, array2):
    """Helper function for list of float comparison."""
    return np.isclose(np.array(array1), np.array(array2)).all()


def test__combine_json_layers_default_values():
    image_json = generate_image_layer(
        source="mysource",
        scale=[1.5, 0.5, 2.5],
        size={"x": 400, "y": 200, "z": 150},
        start={"x": 100, "y": 50, "z": 25},
    )
    combined_json = combine_json_layers(layers=[image_json], scale=[1.0, 1.2, 0.1])

    expected_dimensions = {"x": [1.0, "m"], "y": [1.2, "m"], "z": [0.1, "m"]}
    expected_projection_orientation = [0.3826834, 0, 0, 0.9238796]
    expected_position = [250.0, 125.0, 88.0]

    assert combined_json["layers"][0] == image_json
    assert combined_json["dimensions"] == expected_dimensions
    assert np_is_close(
        combined_json["projectionOrientation"],
        expected_projection_orientation,
    )
    assert combined_json["layout"] == "4panel"
    assert combined_json["showSlices"]
    assert np_is_close(combined_json["position"], expected_position)

    # Check memory setting
    assert combined_json["gpuMemoryLimit"] == 1500000000


def test__generate_image_layer_source_is_unchanged_by_default():
    """The Portal frontend matches tomogram layers by their ``zarr://`` URL: the default must not change."""
    state = generate_image_layer(source="https://files.example/run/tomo.zarr", scale=1e-9, size={"x": 2.0})
    assert state["source"]["url"] == "zarr://https://files.example/run/tomo.zarr"
    transform = state["source"]["transform"]
    assert list(transform["inputDimensions"]) == ["x", "y", "z"]
    assert list(transform["outputDimensions"]) == ["x", "y", "z"]


def test__generate_image_layer_5d_source():
    state = generate_image_layer(
        source="https://files.example/run/tomo.zarr",
        scale=1e-9,
        size={"x": 2.0},
        extra_axes=("t", "c"),
    )
    transform = state["source"]["transform"]
    # every source dimension is listed, named as Neuroglancer names them, with the rank of the output
    assert transform["inputDimensions"] == {
        "x": [1e-9, "m"],
        "y": [1e-9, "m"],
        "z": [1e-9, "m"],
        "t": [1, "s"],
        "c'": [1, ""],
    }
    # the global coordinate space stays x, y, z; t and c become local (primed) dimensions
    assert transform["outputDimensions"] == {
        "x": [1e-9, "m"],
        "y": [1e-9, "m"],
        "z": [1e-9, "m"],
        "t'": [1, "s"],
        "c'": [1, ""],
    }
    assert state["source"]["url"].startswith("zarr://")


def test__generate_image_layer_zarr_version():
    assert generate_image_layer(source="a.zarr", scale=1, size={}, zarr_version=3)["source"]["url"] == "zarr3://a.zarr"
    assert generate_image_layer(source="a.zarr", scale=1, size={}, zarr_version=2)["source"]["url"] == "zarr2://a.zarr"


def test__generate_image_layer_rejects_unknown_axis():
    with pytest.raises(ValueError, match="Unsupported extra axis"):
        generate_image_layer(source="a.zarr", scale=1, size={}, extra_axes=("w",))


def test__generate_image_volume_layer_runs():
    state = generate_image_volume_layer(source="mysource.zarr", scale=(1.0, 1.0, 1.0))
    assert state["source"]["url"] == "zarr://mysource.zarr"
