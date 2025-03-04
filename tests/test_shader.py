from cryoet_data_portal_neuroglancer.shaders.annotation import NonOrientedPointShaderBuilder, OrientedPointShaderBuilder
from cryoet_data_portal_neuroglancer.shaders.image import ImageShaderBuilder, ImageWithVolumeRenderingShaderBuilder
from cryoet_data_portal_neuroglancer.shaders.shader_builder import ShaderBuilder


def test_get_default_image_vr_shader():
    contrast_limits = (0.0, 1.0)
    window_limits = (0.0, 1.0)
    threedee_contrast_limits = (1.0, -1.0)
    threedee_window_limits = None
    contrast_name = "contrast"
    threedee_contrast_name = "contrast3D"
    expected_shader = """
#uicontrol invlerp contrast
#uicontrol bool invert_contrast checkbox
#uicontrol invlerp contrast3D(clamp=false)
#uicontrol bool invert_contrast3D checkbox
#uicontrol bool hide_values_outside_limits_3D checkbox

float get_contrast() {
  float value = invert_contrast ? 1.0 - contrast() : contrast();
  return value;
}
float get_contrast3D() {
  float value = invert_contrast3D ? 1.0 - contrast3D() : contrast3D();
  value = (hide_values_outside_limits_3D && value > 1.0) ? 0.0 : clamp(value, 0.0, 1.0);
  return value;
}

void main() {
  float outputValue;
  if (VOLUME_RENDERING) {
    outputValue = get_contrast3D();
    emitIntensity(outputValue);
  } else {
    outputValue = get_contrast();
  }
  emitGrayscale(outputValue);
}
"""
    shader_builder = ImageWithVolumeRenderingShaderBuilder(
        contrast_limits=contrast_limits,
        window_limits=window_limits,
        threedee_contrast_limits=threedee_contrast_limits,
        threedee_window_limits=threedee_window_limits,
        contrast_name=contrast_name,
        threedee_contrast_name=threedee_contrast_name,
    )
    shader = shader_builder.build()
    actual_shader = shader["shader"]
    assert actual_shader == expected_shader.strip()

    shader_controls = shader["shaderControls"]
    contrast_control = shader_controls[contrast_name]
    assert contrast_control["range"] == list(contrast_limits)
    assert contrast_control["window"] == list(window_limits)

    contrast_threedee_control = shader_controls[threedee_contrast_name]
    assert contrast_threedee_control["range"] == list(threedee_contrast_limits)
    assert contrast_threedee_control["window"] == [-5.0, 5.0]  # Window bigger than range

    checkbox_control = shader_controls[f"invert_{threedee_contrast_name}"]
    assert checkbox_control is True
    checkbox_control = shader_controls[f"invert_{contrast_name}"]
    assert checkbox_control is False


def test_get_default_image_shader():
    contrast_limits = (0.0, 1.0)
    window_limits = (0.0, 1.0)
    contrast_name = "contrast"
    expected_shader = """
#uicontrol invlerp contrast
#uicontrol bool invert_contrast checkbox

float get_contrast() {
  float value = invert_contrast ? 1.0 - contrast() : contrast();
  return value;
}

void main() {
  float outputValue;
  outputValue = get_contrast();
  emitGrayscale(outputValue);
}
"""
    shader_builder = ImageShaderBuilder(
        contrast_limits=contrast_limits,
        window_limits=window_limits,
        contrast_name=contrast_name,
    )
    shader = shader_builder.build()
    actual_shader = shader["shader"]
    assert actual_shader == expected_shader.strip()

    shader_controls = shader["shaderControls"]
    contrast_control = shader_controls[contrast_name]
    assert contrast_control["range"] == list(contrast_limits)
    assert contrast_control["window"] == list(window_limits)


def test_shader_builder():
    expected_shader = """
#uicontrol test

void main() {
  test_main
}
"""
    shader_components = (
        ShaderBuilder().add_to_shader_controls("#uicontrol test").add_to_shader_main("test_main").build()
    )
    assert shader_components["shader"] == expected_shader.strip()
    assert shader_components["shaderControls"] == {}


def test_oriented_point_shader_builder():
    point_size_multiplier = 2.0
    shader_builder = OrientedPointShaderBuilder(point_size_multiplier=point_size_multiplier)
    shader = shader_builder.build()
    expected_shader = """
#uicontrol float pointScale slider(min=0.01, max=2.0, step=0.01)
#uicontrol float lineWidth slider(min=0.01, max=4.0, step=0.01)
#uicontrol float opacity slider(min=0.0, max=1.0, step=0.01)

void main() {
  if (opacity == 0.0) discard;
  setLineWidth(lineWidth);
  setLineColor(vec4(prop_line_color(), opacity));
  setEndpointMarkerSize(pointScale * prop_diameter(), pointScale * 0.5 * prop_diameter());
  setEndpointMarkerColor(vec4(defaultColor(), opacity));
  setEndpointMarkerBorderWidth(0.1);
  setEndpointMarkerBorderColor(vec4(0.0, 0.0, 0.0, opacity));
}
"""
    assert shader["shader"] == expected_shader.strip()
    shader_controls = shader["shaderControls"]
    assert shader_controls["pointScale"] == 2.0
    assert shader_controls["lineWidth"] == 1.0
    assert shader_controls["opacity"] == 1.0


def test_non_oriented_point_shader_builder():
    shader_builder = NonOrientedPointShaderBuilder()
    shader = shader_builder.build()
    expected_shader = """
#uicontrol float pointScale slider(min=0.01, max=2.0, step=0.01)
#uicontrol float opacity slider(min=0.0, max=1.0, step=0.01)

void main() {
  if (opacity == 0.0) discard;
  setColor(vec4(defaultColor(), opacity));
  setPointMarkerSize(pointScale * prop_diameter());
  setPointMarkerBorderWidth(0.1);
  setPointMarkerBorderColor(vec4(0.0, 0.0, 0.0, opacity));
}
"""
    assert shader["shader"] == expected_shader.strip()
    shader_controls = shader["shaderControls"]
    assert shader_controls["pointScale"] == 1.0
    assert shader_controls["opacity"] == 1.0


def test_non_oriented_point_shader_builder_instance_segmentation():
    shader_builder = NonOrientedPointShaderBuilder(is_instance_segmentation=True)
    shader = shader_builder.build()

    expected_shader = """
#uicontrol float pointScale slider(min=0.01, max=2.0, step=0.01)
#uicontrol float opacity slider(min=0.0, max=1.0, step=0.01)

void main() {
  if (opacity == 0.0) discard;
  setColor(vec4(prop_color(), opacity));
  setPointMarkerSize(pointScale * prop_diameter());
  setPointMarkerBorderWidth(0.1);
  setPointMarkerBorderColor(vec4(0.0, 0.0, 0.0, opacity));
}
"""
    assert shader["shader"] == expected_shader.strip()
    shader_controls = shader["shaderControls"]
    assert shader_controls["pointScale"] == 1.0
    assert shader_controls["opacity"] == 1.0
