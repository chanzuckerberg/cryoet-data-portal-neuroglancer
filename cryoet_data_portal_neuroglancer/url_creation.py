import webbrowser
from types import SimpleNamespace
from typing import Optional, Any

import neuroglancer
import neuroglancer.cli
from neuroglancer.url_state import to_json_dump, to_url

from .utils import get_resolution


def launch_nglancer(server_kwargs) -> neuroglancer.Viewer:
    neuroglancer.cli.handle_server_arguments(SimpleNamespace(**server_kwargs))
    viewer = neuroglancer.Viewer()
    return viewer


def open_browser(viewer: neuroglancer.Viewer, hang: bool = False):
    print(viewer)
    webbrowser.open_new(viewer.get_viewer_url())
    if hang:
        input("Press Enter to continue...")


def dump_url_and_state(viewer: neuroglancer.Viewer):
    with viewer.txn() as s:
        url = to_url(s)
        json_state = to_json_dump(s, indent=2)
    print(url, json_state)


def loop_json_and_url(viewer: neuroglancer.Viewer):
    while input("Press Enter to print url and json or q to quit...") != "q":
        dump_url_and_state(viewer)


def viewer_to_url(**server_kwargs):
    viewer = launch_nglancer(server_kwargs)
    open_browser(viewer)
    loop_json_and_url(viewer)
    return 0


def load_jsonstate_to_browser(json_content: dict[str, Any], **server_kwargs):
    state = neuroglancer.viewer_state.ViewerState(json_content)

    viewer = launch_nglancer(server_kwargs)
    viewer.set_state(state)

    open_browser(viewer)
    loop_json_and_url(viewer)
    return 0


def combine_json_layers(
    layers: list[dict[str, Any]],
    resolution: Optional[tuple[float, float, float] | list[float]] = None,
    units: str = "m",
) -> dict[str, Any]:
    image_layers = [layer for layer in layers if layer["type"] == "image"]
    resolution = get_resolution(resolution)
    dimensions = {dim: [res, units] for dim, res in zip("xyz", resolution)}

    combined_json = {
        "dimensions": dimensions,
        "crossSectionScale": 1.8,
        "projectionOrientation": [
            0.0,
            0.655,
            0.0,
            -0.756,
        ],
        "layers": layers,
        "selectedLayer": {
            "visible": True,
            "layer": layers[0]["name"],
        },
        "crossSectionBackgroundColor": "#000000",
        "layout": "4panel",

    }
    if image_layers is not None:
        combined_json["position"] = image_layers[0]["_position"]
        combined_json["crossSectionScale"] = image_layers[0]["_crossSectionScale"]

    return combined_json
