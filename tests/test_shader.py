from cryoet_data_portal_neuroglancer.models.shader_builder import OrientedPointShaderBuilder


def test_oriented_point_shader():
    builder = OrientedPointShaderBuilder()
    output = builder.build_shader()
    shader = output["shader"]
    shader_controls = output["shaderControls"]

    assert shader_controls == {
        "pointScale": 1.0,
        "lineWidth": 1.0,
    }
    assert (
        shader
        == """#uicontrol float pointScale slider(min=0.01, max=10.0)
#uicontrol float lineWidth slider(min=0.01, max=10.0)

void main() {
  setLineWidth(lineWidth());
  setLineColor(prop_line_color());
  setEndpointMarkerSize(prop_diameter() * pointScale(), 0.0);
  setPointMarkerColor(prop_point_color());
}"""
    )
