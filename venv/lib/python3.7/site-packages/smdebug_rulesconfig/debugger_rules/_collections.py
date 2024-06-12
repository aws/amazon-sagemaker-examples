from ._utils import _get_collection_config


def get_collection(collection_name):
    return _get_collection_config(collection_name)
