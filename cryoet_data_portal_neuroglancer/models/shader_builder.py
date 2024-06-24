"""Create GLSL shaders for Neuroglancer."""

from typing import Any, Optional

from cryoet_data_portal_neuroglancer.utils import get_window_limits_from_contrast_limits

TAB = "  "


class ShaderBuilder:
    def __init__(self):
        self._shader_pre_main = ""
        self._shader_main_function = ""
        self._shader_controls = {}

    def add_to_shader_controls(self, code: str | list[str]):
        if isinstance(code, str):
            self._shader_pre_main += code
        else:
            self._shader_pre_main += "\n".join(code)
        self._shader_pre_main += "\n"
        return self

    def add_to_shader_main(self, code: str | list[str], indent: int = 1):
        if isinstance(code, str):
            self._shader_main_function += TAB * indent + code
        else:
            self._shader_main_function += "\n".join([TAB * indent + line for line in code])
        self._shader_main_function += "\n"
        return self

    def _make_main(self) -> str:
        return f"void main() {{\n{self._shader_main_function}}}"

    def _make_pre_main(self) -> str:
        """Sort the preamble for more visually appealing code"""
        # Extract all the #uicontrol lines and put them at the top
        uicontrol_lines = []
        pre_main_lines = []
        for line in self._shader_pre_main.split("\n"):
            if line.startswith("#uicontrol"):
                uicontrol_lines.append(line)
            else:
                pre_main_lines.append(line)
        # Create a blank line between the uicontrols and the rest of the code
        if len(pre_main_lines) > 1:
            pre_main_lines.insert(0, "")
        return "\n".join(uicontrol_lines + pre_main_lines)

    def build_shader(self) -> dict[str, str | dict[str, Any]]:
        return {
            "shader": self._make_pre_main() + "\n" + self._make_main(),
            "shaderControls": self._shader_controls,
        }

    def make_invlerp_component(
        self,
        name: str,
        contrast_limits: tuple[float, float],
        window_limits: tuple[float, float],
    ) -> str:
        controls = self._shader_controls.setdefault(name, {})
        controls["range"] = list(contrast_limits)
        controls["window"] = list(window_limits)
        return f"#uicontrol invlerp {name}"

    def make_invertible_invlerp_component(
        self,
        name: str,
        contrast_limits: tuple[float, float],
        window_limits: tuple[float, float],
    ) -> list[str]:
        invlerp_component = self.make_invlerp_component(name, contrast_limits, window_limits)
        checkbox_part = f"#uicontrol bool invert_{name} checkbox"
        data_value_getter = f"float {name}_get() {{ return invert_{name} ? 1.0 - {name}() : {name}(); }}"
        return [invlerp_component, checkbox_part, data_value_getter]

    def make_color_component(self, name: str, default_color: str) -> str:
        self._shader_controls[name] = default_color
        return f"#uicontrol vec3 {name} color"


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
        volume_rendering_code: str | list[str],
        cross_section_code: str | list[str],
    ):
        self.add_to_shader_main("if (VOLUME_RENDERING) {")
        self.add_to_shader_main(volume_rendering_code, indent=2)
        self.add_to_shader_main("} else {")
        self.add_to_shader_main(cross_section_code, indent=2)
        self.add_to_shader_main("}")