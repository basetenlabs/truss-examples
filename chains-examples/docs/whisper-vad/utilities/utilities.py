import asyncio
import time
from contextlib import contextmanager
from typing import Iterable, TypeVar


@contextmanager
def measure(tag):
    """Utility to measure time taken by a block of code."""
    st = time.time()
    yield
    print("elapsed ", tag, (time.time() - st) * 1000)


_T = TypeVar("_T")


async def gather(tasks: Iterable[asyncio.Task[_T]]) -> list[_T]:
    return await asyncio.gather(*tasks)
