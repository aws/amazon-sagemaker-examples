from ._version import __version__  # noqa
from .style import JupyterStyle  # noqa


def _jupyter_labextension_paths():
    return [{
        "src": "labextension",
        "dest": "jupyterlab_pygments"
    }]
