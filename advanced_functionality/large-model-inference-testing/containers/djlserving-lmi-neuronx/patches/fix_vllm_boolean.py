#!/usr/bin/env python3
"""
Patch DJL Python's vllm_rb_properties.py to fix boolean False handling in vLLM args.
This adds support for --no- prefix flags and underscore-to-hyphen conversion.
Also adds explicit boolean type conversion for string values.

This script patches the source file BEFORE the JAR is built.
"""

# Path to the source file in the cloned repository
file_path = '/usr/src/djl-serving/engines/python/setup/djl_python/properties_manager/vllm_rb_properties.py'

print(f"Patching {file_path}...")

# Read the file
try:
    with open(file_path, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print(f"✗ ERROR: File not found: {file_path}")
    print("Make sure this script runs after git clone and before gradlew build")
    exit(1)

# Define the old function text (exactly as it appears in the file)
old_function = '''def construct_vllm_args_list(vllm_engine_args: dict):
    # Modified from https://github.com/vllm-project/vllm/blob/94666612a938380cb643c1555ef9aa68b7ab1e53/vllm/utils/argparse_utils.py#L441
    args_list = []
    for key, value in vllm_engine_args.items():
        if str(value).lower() in {'true', 'false'}:
            if str(value).lower() == 'true':
                args_list.append("--" + key)
        elif isinstance(value, bool):
            if value:
                args_list.append("--" + key)
        elif isinstance(value, list):
            if value:
                args_list.append("--" + key)
                for item in value:
                    args_list.append(str(item))
        else:
            args_list.append("--" + key)
            args_list.append(str(value))
    return args_list'''

# Define the new function text with our fixes and debug logging
new_function = '''def construct_vllm_args_list(vllm_engine_args: dict):
    # Modified from https://github.com/vllm-project/vllm/blob/94666612a938380cb643c1555ef9aa68b7ab1e53/vllm/utils/argparse_utils.py#L441
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=== construct_vllm_args_list DEBUG ===")
    logger.info(f"Input vllm_engine_args keys: {list(vllm_engine_args.keys())}")
    
    args_list = []
    for key, value in vllm_engine_args.items():
        # Convert underscores to hyphens for CLI args
        cli_key = key.replace('_', '-')
        
        original_value = value
        original_type = type(value).__name__
        
        # Convert string booleans to actual booleans
        if isinstance(value, str) and value.lower() in {'true', 'false'}:
            value = value.lower() == 'true'
            logger.info(f"  {key}: converted string '{original_value}' to bool {value}")
        
        if isinstance(value, bool):
            if value:
                args_list.append("--" + cli_key)
                logger.info(f"  {key} ({original_type}={original_value}) -> --{cli_key}")
            else:
                args_list.append("--no-" + cli_key)
                logger.info(f"  {key} ({original_type}={original_value}) -> --no-{cli_key}")
        elif isinstance(value, list):
            if value:
                args_list.append("--" + cli_key)
                for item in value:
                    args_list.append(str(item))
                logger.info(f"  {key} ({original_type}) -> --{cli_key} {' '.join(str(v) for v in value)}")
        else:
            args_list.append("--" + cli_key)
            args_list.append(str(value))
            logger.info(f"  {key} ({original_type}={original_value}) -> --{cli_key} {value}")
    
    logger.info(f"Final args_list: {args_list}")
    logger.info("=== END construct_vllm_args_list DEBUG ===")
    return args_list'''

# Replace the function
if old_function in content:
    content = content.replace(old_function, new_function)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✓ Successfully patched vllm_rb_properties.py with debug logging")
    print("  This patched version will be included in the JAR during build")
else:
    print("✗ ERROR: Could not find the exact function to patch!")
    print("The function signature or content may have changed.")
    exit(1)
