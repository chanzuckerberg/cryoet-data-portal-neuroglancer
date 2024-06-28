"""Create GLSL shaders for Neuroglancer."""

from typing import Any

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

    def build(self) -> dict[str, str | dict[str, Any]]:
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
        data_value_getter = [
            f"float get_{name}()" + " {",
            f"{TAB}return invert_{name} ? 1.0 - {name}() : {name}();",
            "}",
        ]
        return [invlerp_component, checkbox_part, *data_value_getter]

    def make_color_component(self, name: str, default_color: str) -> str:
        self._shader_controls[name] = default_color
        return f"#uicontrol vec3 {name} color"
