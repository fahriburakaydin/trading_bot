"""
Root conftest — ensure an asyncio event loop exists before ib_insync is imported.

ib_insync (via eventkit) calls ``asyncio.get_event_loop()`` at *import time*.
Python 3.12+ raises RuntimeError when there is no current event loop in the
main thread.  Creating one here, before any test module loads, prevents that.
"""

import asyncio

# Force-create a persistent event loop for the main thread so that
# ib_insync / eventkit can import successfully at any point during the
# test session.
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

# Pre-import ib_insync while the event loop is guaranteed to exist.
# This caches the module so later test-time imports don't trigger the
# eventkit RuntimeError.
try:
    import ib_insync  # noqa: F401
except Exception:
    pass
