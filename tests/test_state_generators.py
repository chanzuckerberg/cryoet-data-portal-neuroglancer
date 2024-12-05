from cryoet_data_portal_neuroglancer.state_generator import generate_image_layer


def test__generate_image_layer_default_values():
    state = generate_image_layer(source="mysource", scale=1.5, size={"a": 2.0})

    assert "blend" in state
    assert state["blend"] == "additive"
    assert state["opacity"] == 1.0
