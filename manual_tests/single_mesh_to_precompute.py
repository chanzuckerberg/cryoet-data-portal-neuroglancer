from pathlib import Path

import trimesh
from cloudvolume import CloudVolume

from cryoet_data_portal_neuroglancer.precompute.mesh import generate_multiresolution_mesh


def generate_basic_trimesh():
    # Create a torus
    mesh = trimesh.creation.torus(
        major_radius=50.0,
        minor_radius=25,
        major_sections=2 ^ 14,
        minor_sections=2 ^ 14,
        transform=trimesh.transformations.translation_matrix([100, 100, 100]),
    )

    return trimesh.Scene(mesh)


def main(output_folder="test_glb", port=1030, num_lods=4):
    import logging

    logging.basicConfig(level=logging.INFO)
    here = Path(__file__).parent
    output_folder = here / output_folder
    scene = generate_basic_trimesh()
    generate_multiresolution_mesh(scene, output_folder, num_lods - 1, 8, bounding_box_size=[500, 500, 500])

    print(
        f"Go to https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B10.890692710876465%2C-2.4535868167877197%2C50.0444221496582%5D%2C%22crossSectionScale%22:1%2C%22projectionOrientation%22:%5B-0.9983203411102295%2C-0.04334384202957153%2C0.009803229942917824%2C-0.037171632051467896%5D%2C%22projectionScale%22:642.7371815420057%2C%22layers%22:%5B%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://http://localhost:{port}%22%2C%22subsources%22:%7B%22default%22:true%2C%22mesh%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22toolBindings%22:%7B%22B%22:%22selectSegments%22%7D%2C%22tab%22:%22rendering%22%2C%22hoverHighlight%22:false%2C%22ignoreNullVisibleSet%22:false%2C%22meshRenderScale%22:475.1228767657264%2C%22crossSectionRenderScale%22:2%2C%22segments%22:%5B%221%22%2C%222%22%2C%223%22%2C%224%22%2C%225%22%2C%226%22%2C%227%22%2C%228%22%5D%2C%22segmentQuery%22:%228%22%2C%22name%22:%22localhost:{port}%22%7D%5D%2C%22showAxisLines%22:false%2C%22showScaleBar%22:false%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22size%22:605%2C%22visible%22:true%2C%22layer%22:%22localhost:{port}%22%7D%2C%22layout%22:%223d%22%7D to view",
    )

    cv = CloudVolume(f"file://{output_folder.resolve()}")
    cv.viewer(port=port)


if __name__ == "__main__":
    main()
