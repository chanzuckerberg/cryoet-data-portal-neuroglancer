from typing import Optional

from cryoet_data_portal_neuroglancer.shaders.shader_builder import ShaderBuilder
from cryoet_data_portal_neuroglancer.utils import get_window_limits_from_contrast_limits


class ImageShaderBuilder(ShaderBuilder):
    """Create a shader for Neuroglancer to display an image.

    The shader will have a contrast control that can be adjusted by the user.
    The contrast control is invertible.
    There is no separate volume rendering control for contrast.
    """

    def __init__(
        self,
        contrast_limits: tuple[float, float],
        window_limits: Optional[tuple[float, float]] = None,
        contrast_name="contrast",
        create_default_shader=True,
    ):
        """Create a shader for Neuroglancer to display an image.

        Parameters
        ----------
        contrast_limits : tuple[float, float]
            The minimum and maximum values for the contrast control.
        window_limits : tuple[float, float], optional
            The minimum and maximum values for the window control, by default None.
            If None, the window limits will be calculated from the contrast limits.
        contrast_name : str, optional
            The name of the contrast control, by default "contrast".
        create_default_shader : bool, optional
            Whether to create the default shader, by default True.
            This is primarily turned off by subclasses.
            A subclass will call the _make_default_shader method to create the shader.
            But when initializing the base class, the default shader is usually intended to be created.
        """
        super().__init__()
        self._contrast_limits = contrast_limits
        self._window_limits = (
            window_limits if window_limits is not None else get_window_limits_from_contrast_limits(contrast_limits)
        )
        self._contrast_name = contrast_name

        # This is a hack to suppress the call to _make_default_shader in the super class
        if create_default_shader:
            self._make_default_shader()

    def _make_default_shader(self, suppress_emission=False):
        self.add_to_shader_controls(
            self.make_invertible_invlerp_component(
                self._contrast_name,
                self._contrast_limits,
                self._window_limits,
            ),
        )
        self.add_to_shader_main("float outputValue;")

        if not suppress_emission:
            self.add_to_shader_main(f"outputValue = get_{self._contrast_name}();")
            self.add_to_shader_main("emitGrayscale(outputValue);")


class ImageWithVolumeRenderingShaderBuilder(ImageShaderBuilder):
    """Create a shader for Neuroglancer to display an image.

    The shader will have a contrast control that can be adjusted by the user.
    The contrast control is invertible.
    There is a separate volume rendering control for contrast.

    """

    def __init__(
        self,
        contrast_limits: tuple[float, float],
        threedee_contrast_limits: tuple[float, float],
        contrast_name="contrast",
        window_limits: Optional[tuple[float, float]] = None,
        threedee_window_limits: Optional[tuple[float, float]] = None,
        threedee_contrast_name="contrast3D",
    ):
        """Create a shader for Neuroglancer to display an image.

        Parameters
        ----------
        contrast_limits : tuple[float, float]
            The minimum and maximum values for the contrast control.
        threedee_contrast_limits : tuple[float, float]
            The minimum and maximum values for the contrast control for volume rendering.
        contrast_name : str, optional
            The name of the contrast control, by default "contrast".
        window_limits : tuple[float, float], optional
            The minimum and maximum values for the window control, by default None.
            If None, the window limits will be calculated from the contrast limits.
        threedee_window_limits : tuple[float, float], optional
            The minimum and maximum values for the window control for volume rendering, by default None.
            If None, the window limits will be calculated from the contrast limits.
        threedee_contrast_name : str, optional
            The name of the contrast control for volume rendering, by default "contrast3D".
        """
        super().__init__(
            contrast_limits=contrast_limits,
            window_limits=window_limits,
            contrast_name=contrast_name,
            create_default_shader=False,
        )
        self._threedee_contrast_limits = threedee_contrast_limits
        self._threedee_window_limits = (
            threedee_window_limits
            if threedee_window_limits is not None
            else get_window_limits_from_contrast_limits(threedee_contrast_limits)
        )
        self._threedee_contrast_name = threedee_contrast_name

        self._make_default_shader()

    def _make_default_shader(self):
        super()._make_default_shader(suppress_emission=True)
        self.add_to_shader_controls(
            self.make_invertible_invlerp_component(
                self._threedee_contrast_name,
                self._threedee_contrast_limits,
                self._threedee_window_limits,
            ),
        )

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
