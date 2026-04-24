#!/usr/bin/env python3
"""Patch verl qwen3_vl.py for transformers >= 4.57.0 compatibility."""
import re

verl_path = __import__('verl').__path__[0]
qwen3_vl_file = f"{verl_path}/models/transformers/qwen3_vl.py"

with open(qwen3_vl_file, 'r') as f:
    content = f.read()

# Pattern to match the visual encoder call
old_pattern = r'image_embeds,\s*dummy_deepstack_image_embeds\s*=\s*model\.visual\(pixel_values,\s*grid_thw=image_grid_thw\)'

# New code handles BaseModelOutputWithDeepstackFeatures which has:
# - last_hidden_state (the image embeddings)
# - deepstack_image_embeds
new_code = '''visual_output = model.visual(pixel_values, grid_thw=image_grid_thw)
        # Handle both old tuple format and new object format (transformers >= 4.57.0)
        # New format returns BaseModelOutputWithDeepstackFeatures with last_hidden_state
        if hasattr(visual_output, 'last_hidden_state'):
            # New format: BaseModelOutputWithDeepstackFeatures
            image_embeds = visual_output.last_hidden_state
            dummy_deepstack_image_embeds = getattr(visual_output, 'deepstack_image_embeds', None)
        elif hasattr(visual_output, 'image_embeds'):
            # Alternative new format with image_embeds attribute
            image_embeds = visual_output.image_embeds
            dummy_deepstack_image_embeds = getattr(visual_output, 'deepstack_image_embeds', None)
        elif isinstance(visual_output, tuple):
            if len(visual_output) == 2:
                image_embeds, dummy_deepstack_image_embeds = visual_output
            elif len(visual_output) == 3:
                image_embeds, _video_embeds, dummy_deepstack_image_embeds = visual_output
            else:
                raise ValueError(f"Unexpected visual output tuple length: {len(visual_output)}")
        else:
            raise ValueError(f"Unexpected visual output type: {type(visual_output)}")'''

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_code, content)
    print("Pattern found and replaced!")
else:
    print("WARNING: Pattern not found, trying alternative...")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'model.visual' in line:
            print(f"Line {i}: {line}")

with open(qwen3_vl_file, 'w') as f:
    f.write(content)

print("Patch complete!")
