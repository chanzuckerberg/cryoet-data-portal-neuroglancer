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
#uicontrol invlerp contrast3D
#uicontrol bool invert_contrast3D checkbox

float get_contrast() {
  return invert_contrast ? 1.0 - contrast() : contrast();
}
float get_contrast3D() {
  return invert_contrast3D ? 1.0 - contrast3D() : contrast3D();
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
    shader = shader_builder.build_shader()
    actual_shader = shader["shader"]
    assert actual_shader == expected_shader.strip()

    shader_controls = shader["shaderControls"]
    contrast_control = shader_controls[contrast_name]
    assert contrast_control["range"] == list(contrast_limits)
    assert contrast_control["window"] == list(window_limits)

    contrast_threedee_control = shader_controls[threedee_contrast_name]
    assert contrast_threedee_control["range"] == list(threedee_contrast_limits)
    assert contrast_threedee_control["window"] == [-1.2, 1.2]


def test_get_default_image_shader():
    contrast_limits = (0.0, 1.0)
    window_limits = (0.0, 1.0)
    contrast_name = "contrast"
    expected_shader = """
#uicontrol invlerp contrast
#uicontrol bool invert_contrast checkbox

float get_contrast() {
  return invert_contrast ? 1.0 - contrast() : contrast();
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
    shader = shader_builder.build_shader()
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
        ShaderBuilder().add_to_shader_controls("#uicontrol test").add_to_shader_main("test_main").build_shader()
    )
    assert shader_components["shader"] == expected_shader.strip()
    assert shader_components["shaderControls"] == {}
