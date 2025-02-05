from cryoet_data_portal_neuroglancer.shaders.shader_builder import ShaderBuilder


class PointShaderBuilder(ShaderBuilder):
    def __init__(self, point_size_multiplier: float = 1.0, is_instance_segmentation: bool = False):
        super().__init__()
        self.point_size_multiplier = point_size_multiplier
        self.is_instance_segmentation = is_instance_segmentation

    def _make_default_shader(self):
        self.add_to_shader_controls(
            self.make_slider_component(
                "pointScale",
                min_value=0.01,
                max_value=2.0,
                default_value=self.point_size_multiplier,
            ),
        )
        self.add_to_shader_controls(
            self.make_slider_component("opacity", min_value=0.0, max_value=1.0, default_value=1.0),
        )
        self.add_to_shader_main("if (opacity == 0.0) discard;")

    def _get_color_setter(self):
        return "prop_color()" if self.is_instance_segmentation else "defaultColor()"


class NonOrientedPointShaderBuilder(PointShaderBuilder):
    def __init__(self, point_size_multiplier: float = 1.0, is_instance_segmentation: bool = False):
        super().__init__(point_size_multiplier=point_size_multiplier, is_instance_segmentation=is_instance_segmentation)
        print(self._get_color_setter())
        self._make_default_shader()

    def _make_default_shader(self):
        super()._make_default_shader()

        self.add_to_shader_main(
            (
                [
                    f"setColor(vec4({self._get_color_setter()}, opacity));",
                    "setPointMarkerSize(pointScale * prop_diameter());",
                    "setPointMarkerBorderWidth(0.1);",
                    "setPointMarkerBorderColor(vec4(0.0, 0.0, 0.0, opacity));",
                ]
            ),
        )


class OrientedPointShaderBuilder(PointShaderBuilder):
    def __init__(
        self,
        point_size_multiplier: float = 1.0,
        is_instance_segmentation: bool = False,
        line_width: float = 1.0,
    ):
        super().__init__(
            point_size_multiplier=point_size_multiplier,
            is_instance_segmentation=is_instance_segmentation,
        )
        self.line_width = line_width
        self._make_default_shader()

    def _make_default_shader(self):
        super()._make_default_shader()
        self.add_to_shader_controls(
            (
                self.make_slider_component(
                    "lineWidth",
                    min_value=0.01,
                    max_value=4.0,
                    default_value=self.line_width,
                ),
            ),
        )
        self.add_to_shader_main(
            (
                "setLineWidth(lineWidth);",
                "setLineColor(vec4(prop_line_color(), opacity));",
                "setEndpointMarkerSize(pointScale * prop_diameter(), pointScale * 0.5 * prop_diameter());",
                f"setEndpointMarkerColor(vec4({self._get_color_setter()}, opacity));",
                "setEndpointMarkerBorderWidth(0.1);",
                "setEndpointMarkerBorderColor(vec4(0.0, 0.0, 0.0, opacity));",
            ),
        )
        # Sort the shader preamble to ensure that the sliders are ordered
        mapping = {"pointScale": 0, "lineWidth": 1, "opacity": 2}

        def sorting(x):
            for m in mapping:
                if m in x:
                    return mapping[m]
            return len(mapping)

        self.sort_shader_preamble(sorting)
