_S3_PREFIX = "s3://"


def is_s3_source(src):
    return src.startswith(_S3_PREFIX)


def parse_s3_address(address):
    address = address[len(_S3_PREFIX) :]
    return address.split("/", 1)
