"""Create GLSL shaders for Neuroglancer."""

from typing import Optional

from cryoet_data_portal_neuroglancer.utils import get_window_limits_from_contrast_limits

TAB = "  "


class ShaderBuilder:
    def __init__(self):
        self._shader_pre_main = ""
        self._shader_main_function = ""

    def add_to_shader_controls(self, code: str | list[str]):
        if isinstance(code, str):
            self._shader_pre_main += code
        else:
            self._shader_pre_main += "\n".join(code)
        self._shader_pre_main += "\n"

    def add_to_shader_main(self, code: str | list[str], indent: int = 1):
        if isinstance(code, str):
            self._shader_main_function += TAB * indent + code
        else:
            self._shader_main_function += "\n".join([TAB * indent + line for line in code])
        self._shader_main_function += "\n"

    def _make_main(self) -> str:
        return f"void main() {{\n{self._shader_main_function}}}"

    def make_shader(self) -> str:
        return self._shader_pre_main + "\n" + self._make_main()

    @staticmethod
    def make_invlerp_component(
        name: str,
        contrast_limits: tuple[float, float],
        window_limits: tuple[float, float],
    ) -> str:
        return f"#uicontrol invlerp {name}(range=[{contrast_limits[0]}, {contrast_limits[1]}], window=[{window_limits[0]}, {window_limits[1]}])"

    @staticmethod
    def make_invertible_invlerp_component(
        name: str,
        contrast_limits: tuple[float, float],
        window_limits: tuple[float, float],
    ) -> list[str]:
        invlerp_component = ShaderBuilder.make_invlerp_component(name, contrast_limits, window_limits)
        checkbox_part = f"#uicontrol bool invert_{name} checkbox(default=false)"
        data_value_getter = f"float {name}_get() {{ return invert_{name} ? 1.0 - {name}() : {name}(); }}"
        return [invlerp_component, checkbox_part, data_value_getter]


class ImageVolumeShaderBuilder(ShaderBuilder):
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
            self.make_invertible_invlerp_component(self._contrast_name, self._contrast_limits, self._window_limits),
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
                f"outputValue = {self._threedee_contrast_name}_get();",
                "emitIntensity(outputValue);",
            ],
            [
                f"outputValue = {self._contrast_name}_get();",
            ],
        )
        self.add_to_shader_main("emitGrayscale(outputValue);")

    def _add_cross_section_and_vr_code(
        self,
        volume_rendering_code: list[str],
        cross_section_code: list[str],
    ):
        self.add_to_shader_main("if (VOLUME_RENDERING) {")
        self.add_to_shader_main(volume_rendering_code, indent=2)
        self.add_to_shader_main("} else {")
        self.add_to_shader_main(cross_section_code, indent=2)
        self.add_to_shader_main("}")
