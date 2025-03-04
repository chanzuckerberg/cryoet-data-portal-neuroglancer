import numpy as np

from cryoet_data_portal_neuroglancer.state_generator import (
    combine_json_layers,
    generate_image_layer,
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

    # Assertions
    assert combined_json["layers"][0] == image_json
    assert combined_json["dimensions"] == expected_dimensions
    assert np_is_close(
        combined_json["projectionOrientation"],
        expected_projection_orientation,
    )
    assert combined_json["layout"] == "4panel"
    assert combined_json["showSlices"]
    assert np_is_close(combined_json["position"], expected_position)
