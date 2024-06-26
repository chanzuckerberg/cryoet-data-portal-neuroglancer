from typing import Optional

from cryoet_data_portal_neuroglancer.shaders.shader_builder import ShaderBuilder
from cryoet_data_portal_neuroglancer.utils import get_window_limits_from_contrast_limits


class ImageShaderBuilder(ShaderBuilder):
    def __init__(
        self,
        contrast_limits: tuple[float, float],
        threedee_contrast_limits: tuple[float, float],
        window_limits: Optional[tuple[float, float]] = None,
        threedee_window_limits: Optional[tuple[float, float]] = None,
        contrast_name="contrast",
        threedee_contrast_name="contrast3D",
    ):
        super().__init__()
        self._contrast_limits = contrast_limits
        self._window_limits = (
            window_limits if window_limits is not None else get_window_limits_from_contrast_limits(contrast_limits)
        )
        self._threedee_contrast_limits = threedee_contrast_limits
        self._threedee_window_limits = (
            threedee_window_limits
            if threedee_window_limits is not None
            else get_window_limits_from_contrast_limits(threedee_contrast_limits)
        )
        self._contrast_name = contrast_name
        self._threedee_contrast_name = threedee_contrast_name

        self._make_default_shader()

    def _make_default_shader(self):
        self.add_to_shader_controls(
            self.make_invertible_invlerp_component(
                self._contrast_name,
                self._contrast_limits,
                self._window_limits,
            ),
        )
        self.add_to_shader_controls(
            self.make_invertible_invlerp_component(
                self._threedee_contrast_name,
                self._threedee_contrast_limits,
                self._threedee_window_limits,
            ),
        )
        self.add_to_shader_main("float outputValue;")
        self._add_cross_section_and_vr_code(
            [
                f"outputValue = get_{self._threedee_contrast_name}();",
                "emitIntensity(outputValue);",
            ],
            [
                f"outputValue = get_{self._contrast_name}();",
            ],
        )
        self.add_to_shader_main("emitGrayscale(outputValue);")

    def _add_cross_section_and_vr_code(
        self,
        volume_rendering_code: str | list[str],
        cross_section_code: str | list[str],
    ):
        self.add_to_shader_main("if (VOLUME_RENDERING) {")
        self.add_to_shader_main(volume_rendering_code, indent=2)
        self.add_to_shader_main("} else {")
        self.add_to_shader_main(cross_section_code, indent=2)
        self.add_to_shader_main("}")
