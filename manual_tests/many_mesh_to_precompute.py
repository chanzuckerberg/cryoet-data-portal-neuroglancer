from pathlib import Path

import trimesh

from cryoet_data_portal_neuroglancer.precompute.mesh import generate_multilabel_multiresolution_mesh


def generate_basic_trimeshes():
    # Create a dictionary of trimeshes from trimesh primitives
    mesh_dict = {}

    # Create a cube
    mesh_dict[1] = trimesh.creation.box(
        [100, 100, 100],
        transform=trimesh.transformations.translation_matrix([0, 0, 0]),
    )

    # Create a torus
    mesh_dict[2] = trimesh.creation.torus(
        major_radius=50.0,
        minor_radius=25,
        transform=trimesh.transformations.translation_matrix([100, 100, 100]),
    )

    # Create a sphere
    mesh_dict[3] = trimesh.creation.icosphere(
        radius=50,
        subdivisions=3,
    ).apply_translation([-100, -100, -100])

    # Create a cylinder
    mesh_dict[4] = trimesh.creation.cylinder(
        radius=50,
        height=100,
        transform=trimesh.transformations.translation_matrix([100, -100, 100]),
    )

    # Create a cone
    mesh_dict[5] = trimesh.creation.cone(
        radius=50,
        height=100,
        transform=trimesh.transformations.translation_matrix([-100, 100, -100]),
    )

    # Create a capsule
    mesh_dict[6] = trimesh.creation.capsule(
        radius=50,
        height=100,
        transform=trimesh.transformations.translation_matrix([100, 100, -100]),
    )

    # Create an annulus
    mesh_dict[7] = trimesh.creation.annulus(
        50.0,
        25,
        40.0,
        transform=trimesh.transformations.translation_matrix([-100, -100, 100]),
    )

    # Create an icosahedron
    mesh_dict[8] = trimesh.creation.icosahedron().apply_translation([10, -10, -10]).apply_scale(10.0)

    return {k: trimesh.Scene(v) for k, v in mesh_dict.items()}


def main(output_folder="test_glb", port=1030, num_lods=2):
    here = Path(__file__).parent
    output_folder = here / output_folder
    mesh_dict = generate_basic_trimeshes()
    string_labels = [
        "cube",
        "torus",
        "sphere",
        "cylinder",
        "cone",
        "capsule",
        "annulus",
        "icosahedron",
    ]
    string_labels = dict(zip(mesh_dict.keys(), string_labels, strict=False))
    generate_multilabel_multiresolution_mesh(
        mesh_dict,
        output_folder,
        num_lods - 1,
        16,
        [500, 500, 500],
        string_labels,
    )

    print(
        f"Go to https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B10.890692710876465%2C-2.4535868167877197%2C50.0444221496582%5D%2C%22crossSectionScale%22:1%2C%22projectionOrientation%22:%5B-0.9983203411102295%2C-0.04334384202957153%2C0.009803229942917824%2C-0.037171632051467896%5D%2C%22projectionScale%22:642.7371815420057%2C%22layers%22:%5B%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://http://localhost:{port}%22%2C%22subsources%22:%7B%22default%22:true%2C%22mesh%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22toolBindings%22:%7B%22B%22:%22selectSegments%22%7D%2C%22tab%22:%22rendering%22%2C%22hoverHighlight%22:false%2C%22ignoreNullVisibleSet%22:false%2C%22meshRenderScale%22:475.1228767657264%2C%22crossSectionRenderScale%22:2%2C%22segments%22:%5B%221%22%2C%222%22%2C%223%22%2C%224%22%2C%225%22%2C%226%22%2C%227%22%2C%228%22%5D%2C%22segmentQuery%22:%228%22%2C%22name%22:%22localhost:{port}%22%7D%5D%2C%22showAxisLines%22:false%2C%22showScaleBar%22:false%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22size%22:605%2C%22visible%22:true%2C%22layer%22:%22localhost:{port}%22%7D%2C%22layout%22:%223d%22%7D to view",
    )
    print(f"Run npx http-server test__glb --cors=authorization -p={port} to serve the files first")


if __name__ == "__main__":
    main()
