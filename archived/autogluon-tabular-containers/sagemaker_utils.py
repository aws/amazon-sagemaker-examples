from distutils.version import StrictVersion

from sagemaker import image_uris


def retrieve_available_framework_versions(framework_type="training", details=False):
    """Get available versions of autogluon

    Args:
        framework_type (str, optional):
            Type of framework. Options: 'training', 'inference'.
            Defaults to 'training'.
        details (bool, optional):
            Whether to get detailed information of each versions.
            Defaults to False.

    Returns:
        (Union(list, dict)):
            returns a list of versions if detailed == False.
            returns a dict containing information related to each version if detailed == True.
    """
    assert framework_type in ["training", "inference"]
    config = image_uris.config_for_framework("autogluon")
    versions_details = config[framework_type]["versions"]
    if details:
        return versions_details
    versions = list(config[framework_type]["versions"].keys())
    return versions


def retrieve_py_versions(framework_version, framework_type="training"):
    versions_details = retrieve_available_framework_versions(
        framework_type, details=True
    )
    return versions_details[framework_version]["py_versions"]


def retrieve_latest_framework_version(framework_type="training"):
    """Get latest version of autogluon framework and its py_versions

    Args:
        framework_type (str, optional):
            Type of framework. Options: 'training', 'inference'.
            Defaults to 'training'.

    Returns:
        (str, list):
            version number of latest autogluon framework, and its py_versions as a list
    """
    versions = retrieve_available_framework_versions(framework_type)
    versions.sort(key=StrictVersion)
    versions = [(v, retrieve_py_versions(v, framework_type)) for v in versions]
    return versions[-1]
