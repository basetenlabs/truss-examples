# TRTLLM

To run this model on Baseten, you'll need to download the engine into the `data` directory. To do so correctly, run the following code snippet

```
from huggingface_hub import snapshot_download

snapshot_download(
    "huggingface_repo_id_of_engine",
    local_dir="path_to_data_directory",
    local_dir_use_symlinks=False,
    max_workers=4
)
```