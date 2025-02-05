from cryoet_data_portal_neuroglancer.state_generator import (
    combine_json_layers,
    generate_image_layer,
    generate_segmentation_mask_layer,
)


def test__generate_image_layer_default_values():
    state = generate_image_layer(source="mysource", scale=1.5, size={"a": 2.0})

    assert "blend" in state
    assert state["blend"] == "additive"
    assert state["opacity"] == 1.0


def test__generate_segmentation_layer_default_values():
    state = generate_segmentation_mask_layer(source="mysource")

    assert "pick" in state
    assert state["pick"] is False


def test__generate_configuration_default_values():
    state = combine_json_layers(layers=[{"type": "image", "volumeRendering": "OK", "name": "myname"}], scale=1.0)

    assert "toolPalettes" in state
    assert len(state["toolPalettes"]) == 1

    palette = state["toolPalettes"]["Dimensions"]
    assert palette.get("side") == "bottom"
    assert palette.get("row") == 1
    assert palette.get("query") == "type:dimension"
    assert palette.get("size") == 120
    assert not palette.get("verticalStacking")
