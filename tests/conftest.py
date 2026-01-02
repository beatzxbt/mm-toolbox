import asyncio
import glob
import os
import pytest


# Ignore all .pyx files from collection - need absolute paths
_test_root = os.path.dirname(os.path.abspath(__file__))
collect_ignore = [
    os.path.relpath(f, _test_root)
    for f in glob.glob(os.path.join(_test_root, "**/*.pyx"), recursive=True)
]


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for tests that request 'event_loop'."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()
