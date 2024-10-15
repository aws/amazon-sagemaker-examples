"""Train.py."""

import os
os.environ["NVTE_TORCH_COMPILE"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_FLASH_ATTN"] = "0"

from arguments import parse_args
import train_lib

def main():
    """Main function to train GPT."""
    args, _ = parse_args()
    train_lib.main(args)


if __name__ == "__main__":
    main()
