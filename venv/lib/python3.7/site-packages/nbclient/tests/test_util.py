import asyncio
from unittest.mock import MagicMock

import pytest
import tornado

from nbclient.util import run_hook, run_sync


@run_sync
async def some_async_function():
    await asyncio.sleep(0.01)
    return 42


def test_nested_asyncio_with_existing_ioloop():
    async def _test():
        assert some_async_function() == 42
        return asyncio.get_running_loop()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    event_loop = loop.run_until_complete(_test())
    assert event_loop is loop


def test_nested_asyncio_with_no_ioloop():
    asyncio.set_event_loop(None)
    assert some_async_function() == 42


def test_nested_asyncio_with_tornado():
    # This tests if tornado accepts the pure-Python Futures, see
    # https://github.com/tornadoweb/tornado/issues/2753
    asyncio.set_event_loop(asyncio.new_event_loop())
    ioloop = tornado.ioloop.IOLoop.current()

    async def some_async_function():
        future: asyncio.Future = asyncio.ensure_future(asyncio.sleep(0.1))
        # the asyncio module, check if tornado likes it:
        ioloop.add_future(future, lambda f: f.result())  # type:ignore
        await future
        return 42

    def some_sync_function():
        return run_sync(some_async_function)()

    async def run():
        # calling some_async_function directly should work
        assert await some_async_function() == 42
        assert some_sync_function() == 42

    ioloop.run_sync(run)


@pytest.mark.asyncio
async def test_run_hook_sync():
    some_sync_function = MagicMock()
    await run_hook(some_sync_function)
    assert some_sync_function.call_count == 1


@pytest.mark.asyncio
async def test_run_hook_async():
    hook = MagicMock(return_value=some_async_function())
    await run_hook(hook)
    assert hook.call_count == 1
