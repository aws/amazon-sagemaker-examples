# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import asyncio
import atexit
import errno
import inspect
import os
import threading
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union


def ensure_dir_exists(path, mode=0o777):
    """ensure that a directory exists
    If it doesn't exist, try to create it, protecting against a race condition
    if another process is doing the same.
    The default permissions are determined by the current umask.
    """
    try:
        os.makedirs(path, mode=mode)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.isdir(path):
        raise OSError("%r exists but is not a directory" % path)


T = TypeVar("T")


class _TaskRunner:
    """A task runner that runs an asyncio event loop on a background thread."""

    def __init__(self):
        self.__io_loop: Optional[asyncio.AbstractEventLoop] = None
        self.__runner_thread: Optional[threading.Thread] = None
        self.__lock = threading.Lock()
        atexit.register(self._close)

    def _close(self):
        if self.__io_loop:
            self.__io_loop.stop()

    def _runner(self):
        loop = self.__io_loop
        assert loop is not None
        try:
            loop.run_forever()
        finally:
            loop.close()

    def run(self, coro):
        """Synchronously run a coroutine on a background thread."""
        with self.__lock:
            name = f"{threading.current_thread().name} - runner"
            if self.__io_loop is None:
                self.__io_loop = asyncio.new_event_loop()
                self.__runner_thread = threading.Thread(target=self._runner, daemon=True, name=name)
                self.__runner_thread.start()
        fut = asyncio.run_coroutine_threadsafe(coro, self.__io_loop)
        return fut.result(None)


_runner_map = {}
_loop_map = {}


def run_sync(coro: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """Runs a coroutine and blocks until it has executed.

    Parameters
    ----------
    coro : coroutine
        The coroutine to be executed.
    Returns
    -------
    result :
        Whatever the coroutine returns.
    """

    def wrapped(*args, **kwargs):
        name = threading.current_thread().name
        inner = coro(*args, **kwargs)
        try:
            # If a loop is currently running in this thread,
            # use a task runner.
            asyncio.get_running_loop()
            if name not in _runner_map:
                _runner_map[name] = _TaskRunner()
            return _runner_map[name].run(inner)
        except RuntimeError:
            pass

        # Run the loop for this thread.
        if name not in _loop_map:
            _loop_map[name] = asyncio.new_event_loop()
        loop = _loop_map[name]
        return loop.run_until_complete(inner)

    wrapped.__doc__ = coro.__doc__
    return wrapped


async def ensure_async(obj: Union[Awaitable[Any], Any]) -> Any:
    """Convert a non-awaitable object to a coroutine if needed,
    and await it if it was not already awaited.
    """
    if inspect.isawaitable(obj):
        try:
            result = await obj
        except RuntimeError as e:
            if str(e) == "cannot reuse already awaited coroutine":
                # obj is already the coroutine's result
                return obj
            raise
        return result
    # obj doesn't need to be awaited
    return obj
