"""Create GLSL shaders for Neuroglancer."""

from typing import Any, Iterable, Optional

TAB = "  "


class ShaderBuilder:
    def __init__(self):
        self._shader_pre_main = ""
        self._shader_main_function = ""
        self._shader_controls = {}

    def add_to_shader_controls(self, code: str | Iterable[str]):
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

    def sort_shader_preamble(self, sorting: lambda x: x):
        self._shader_pre_main = "\n".join(sorted(self._shader_pre_main.split("\n"), key=sorting))

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
        clamp: bool = True,
    ) -> str:
        controls = self._shader_controls.setdefault(name, {})
        controls["range"] = list(contrast_limits)
        controls["window"] = list(window_limits)
        if clamp:
            return f"#uicontrol invlerp {name}"
        else:
            return f"#uicontrol invlerp {name}(clamp=false)"

    def make_invertible_invlerp_component(
        self,
        name: str,
        contrast_limits: tuple[float, float],
        window_limits: tuple[float, float],
        checked_by_default: bool = False,
        can_hide_noise: bool = False,
        noise_name: str | None = None,
    ) -> list[str]:
        """
        Create an invertible invlerp component with a checkbox to invert the values.

        Parameters
        ----------
        name : str
            The name of the component.
        contrast_limits : tuple[float, float]
            The minimum and maximum values for the contrast values.
        window_limits : tuple[float, float]
            The minimum and maximum values for the window control.
        checked_by_default : bool, optional
            Whether the checkbox should be checked by default, by default False.
        can_hide_noise : bool, optional
            Whether the high noise can be hidden, by default False.
        noise_name : str, optional
            The name of the noise control, by default None.
        """
        invlerp_component = self.make_invlerp_component(
            name,
            contrast_limits,
            window_limits,
            clamp=not can_hide_noise,
        )
        checkbox_part = f"#uicontrol bool invert_{name} checkbox"
        data_value_getter = [
            f"float get_{name}()" + " {",
            f"{TAB}float value = invert_{name} ? 1.0 - {name}() : {name}();",
        ]
        self._shader_controls[f"invert_{name}"] = checked_by_default
        if can_hide_noise:
            if noise_name is None:
                noise_name = name
            checkbox_part += f"\n#uicontrol bool hide_noise_{noise_name} checkbox"
            self._shader_controls[f"hide_noise_{noise_name}"] = False
            data_value_getter += [
                f"{TAB}value = (hide_noise_{noise_name} && value > 1.0) ? 0.0 : clamp(val, 0.0, 1.0);",
            ]
        data_value_getter += [f"{TAB}return value;", "}"]
        return [invlerp_component, checkbox_part, *data_value_getter]

    def make_slider_component(
        self,
        name: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        default_value: Optional[float] = None,
    ) -> str:
        if default_value is not None:
            self._shader_controls[name] = default_value
        return f"#uicontrol float {name} slider(min={min_value}, max={max_value}, step=0.01)"

    def make_color_component(self, name: str, default_color: str) -> str:
        self._shader_controls[name] = default_color
        return f"#uicontrol vec3 {name} color"
