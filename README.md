# cryoet-data-portal-neuroglancer

CryoET Data Portal Neuroglancer configuration helper

## Installation

Running the following commands will clone the repository and install the required dependencies.

```bash
git clone https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer.git
cd cryoet-data-portal-neuroglancer
poetry install
```

To specify this as a dependency in a python project with poetry, add the following to your `pyproject.toml` file:

```
cryoet-data-portal-neuroglancer = { git = "https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer.git", branch = "main" }
```

As the main branch is not always stable, it is recommended to specify a commit hash or use a tag instead of a branch name.
