import os
import subprocess


def clone_repo():
    git_repo_url = "https://github.com/xuebinqin/DIS"
    commit_hash = "ec4a4f4f8d967f744bf857149d5ee343b59766b0"
    git_clone_command = ["git", "clone", git_repo_url]

    # clone the repo
    subprocess.run(git_clone_command, check=True)
    print("Git repository cloned successfully!")

    os.chdir(os.path.join(os.getcwd(), "DIS", "IS-Net"))

    # Pin repo to a specific commit
    checkout_command = ["git", "checkout", commit_hash]
    subprocess.run(checkout_command, check=True)
