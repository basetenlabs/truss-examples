from pathlib import Path

def move_all_files(src: Path, dest: Path):
    """
    Moves all files from `src` to `dest` recursively.
    """
    for item in src.iterdir():
        dest_item = dest / item.name
        if item.is_dir():
            dest_item.mkdir(parents=True, exist_ok=True)
            move_all_files(item, dest_item)
        else:
            item.rename(dest_item)

def prepare_model_repository(data_dir: Path):
    """
    Moves all files from `data_dir` to the model repository directory.
    """
    # Ensure the destination directory exists
    dest_dir = Path("/packages/inflight_batcher_llm/tensorrt_llm/1")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure empty version directory for `ensemble` model exists
    ensemble_dir = Path("/packages/inflight_batcher_llm/ensemble/1")
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Move all files and directories from data_dir to dest_dir
    move_all_files(data_dir, dest_dir)
