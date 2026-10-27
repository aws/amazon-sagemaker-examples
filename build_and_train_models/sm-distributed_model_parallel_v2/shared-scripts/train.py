"""Train.py."""

import os

from arguments import parse_args
import train_lib

def main():
    """Main function to train GPT."""
    args, _ = parse_args()
    train_lib.main(args)


if __name__ == "__main__":
    main()
