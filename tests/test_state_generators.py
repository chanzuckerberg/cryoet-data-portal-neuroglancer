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

    assert "enableLayerColorWidget" in state
    assert state["enableLayerColorWidget"] is True

    assert "toolPalettes" in state
    assert len(state["toolPalettes"]) == 1

    palette = state["toolPalettes"]["Dimensions"]
    assert palette.get("side") == "bottom"
    assert palette.get("row") == 1
    assert len(palette.get("tools", [])) == 3

    tools = palette["tools"]
    assert "type" in tools[0]
    assert "type" in tools[1]
    assert "type" in tools[2]

    assert tools[0]["type"] == "dimension"
    assert tools[1]["type"] == "dimension"
    assert tools[2]["type"] == "dimension"

    assert "dimension" in tools[0]
    assert "dimension" in tools[1]
    assert "dimension" in tools[2]

    assert tools[0]["dimension"] == "x"
    assert tools[1]["dimension"] == "y"
    assert tools[2]["dimension"] == "z"
