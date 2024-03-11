"""General utility methods"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import inspect
from typing import Any, Callable, Optional

from jupyter_core.utils import ensure_async, run_sync  # noqa: F401


async def run_hook(hook: Optional[Callable], **kwargs: Any) -> None:
    """Run a hook callback."""
    if hook is None:
        return
    res = hook(**kwargs)
    if inspect.isawaitable(res):
        await res
