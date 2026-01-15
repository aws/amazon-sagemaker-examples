import torch
import tarfile
import tempfile
import os

input_nemo = os.getenv("PATH_TO_TRAINED_MODEL")
output_nemo = input_nemo.replace(".nemo", "_fixed.nemo")

if os.path.exists(output_nemo):
    # Check if output is more recent than input
    output_mtime = os.path.getmtime(output_nemo)
    input_mtime = os.path.getmtime(input_nemo)
    
    if output_mtime > input_mtime:
        print(f"Fixed .nemo already exists and is more recent: {output_nemo}")
        exit(0)
    else:
        print(f"Fixed .nemo exists but is older than input, will regenerate: {output_nemo}")

print(f"Fixing keys in: {input_nemo}")

with tempfile.TemporaryDirectory() as tmpdir:
    with tarfile.open(input_nemo, "r") as tar:
        tar.extractall(tmpdir)
    
    for rank in range(8):
        ckpt_path = f"{tmpdir}/mp_rank_{rank:02d}/model_weights.ckpt"
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Change 'model.decoder.' to 'model.module.decoder.'
        new_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('model.decoder.'):
                new_key = key.replace('model.decoder.', 'model.module.decoder.', 1)
            else:
                new_key = key
            new_checkpoint[new_key] = value
        
        if rank == 0:
            print(f"\nSample transformed keys (rank 0):")
            for key in sorted(list(new_checkpoint.keys()))[:3]:
                print(f"  {key}")
        
        torch.save(new_checkpoint, ckpt_path)
    
    with tarfile.open(output_nemo, "w") as tar:
        for item in os.listdir(tmpdir):
            tar.add(os.path.join(tmpdir, item), arcname=item)

print(f"\nFixed .nemo saved to: {output_nemo}")