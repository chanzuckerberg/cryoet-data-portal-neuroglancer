from cryoet_data_portal_neuroglancer.state_generator import (
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
