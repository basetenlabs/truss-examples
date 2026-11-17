import json
import subprocess
import sys

import yaml


def compute_library_models_to_deploy(git_diff_arg):
    command = ["git", "--no-pager", "diff", "--name-only", git_diff_arg]
    git_diff_output = subprocess.run(
        command, capture_output=True, text=True
    ).stdout.strip()

    files_changed = git_diff_output.split("\n")

    with open("library.config.yaml", "r") as f:
        library_contents = f.read()
        library_models = yaml.safe_load(library_contents)

    library_models_to_deploy = set()
    for library_model_slug, library_model in library_models.items():
        for file_changed in files_changed:
            if file_changed.startswith(library_model["path"]):
                library_models_to_deploy.add(library_model_slug)

    return library_models_to_deploy


if __name__ == "__main__":
    git_diff_arg = sys.argv[1]

    library_models_to_deploy = compute_library_models_to_deploy(git_diff_arg)

    print(json.dumps(list(library_models_to_deploy)))
