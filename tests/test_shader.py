from cryoet_data_portal_neuroglancer.models.shader_builder import ImageVolumeShaderBuilder


def test_get_default_image_shader():
    contrast_limits = (0.0, 1.0)
    window_limits = (0.0, 1.0)
    threedee_contrast_limits = (1.0, -1.0)
    threedee_window_limits = None
    expected_shader = """
#uicontrol invlerp contrast(range=[0.0, 1.0], window=[0.0, 1.0])
#uicontrol bool invert_contrast checkbox(default=false)
float contrast_get() { return invert_contrast ? 1.0 - contrast() : contrast(); }
#uicontrol invlerp contrast3D(range=[1.0, -1.0], window=[-1.2, 1.2])
#uicontrol bool invert_contrast3D checkbox(default=false)
float contrast3D_get() { return invert_contrast3D ? 1.0 - contrast3D() : contrast3D(); }

void main() {
  float outputValue;
  if (VOLUME_RENDERING) {
    outputValue = contrast3D_get();
    emitIntensity(outputValue);
  } else {
    outputValue = contrast_get();
  }
  emitGrayscale(outputValue);
}
"""
    shader_builder = ImageVolumeShaderBuilder(
        contrast_limits=contrast_limits,
        window_limits=window_limits,
        threedee_contrast_limits=threedee_contrast_limits,
        threedee_window_limits=threedee_window_limits,
    )
    actual_shader = shader_builder.make_shader()
    print(actual_shader)
    assert actual_shader.strip() == expected_shader.strip()
