from __future__ import unicode_literals

from .base import DummyOutput, Output
from .color_depth import ColorDepth
from .defaults import create_output, get_default_output, set_default_output

__all__ = [
    # Base.
    'Output',
    'DummyOutput',

    # Color depth.
    'ColorDepth',

    # Defaults.
    'create_output',
    'get_default_output',
    'set_default_output',
]
