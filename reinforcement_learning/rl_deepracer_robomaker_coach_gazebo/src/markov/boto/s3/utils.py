"""This module implement s3 utils"""

import os


def get_s3_key(s3_prefix, postfix):
    """Parse hyperparameters S3 prefix and postfix into key

    Args:
        s3_prefix(str): s3 prefix
        postfix(str): postfix

    Returns:
        str: s3 key by joining prefix and postfix

    """

    # parse S3 prefix and postfix into key
    s3_key = os.path.normpath(os.path.join(s3_prefix, postfix))
    return s3_key


def is_power_of_two(n):
    """Return True if n is a power of two."""
    if n <= 0:
        return False
    else:
        return n & (n - 1) == 0
