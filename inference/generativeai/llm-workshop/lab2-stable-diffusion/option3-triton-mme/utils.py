from huggingface_hub import snapshot_download
from pathlib import Path
import shutil


def download_model(model_name, local_model_path):
    local_model_path = Path(local_model_path)
    local_model_path.mkdir(exist_ok=True)
    local_cache_path = Path("./tmp_cache")
    snapshot_download(
        repo_id=model_name,
        local_dir_use_symlinks=False,
        revision="fp16",
        cache_dir=local_cache_path,
        local_dir=local_model_path,
        ignore_patterns=["*.ckpt", "*.safetensors"],
    )
    shutil.rmtree(local_cache_path)

    return local_model_path