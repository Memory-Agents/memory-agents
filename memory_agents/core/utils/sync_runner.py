import asyncio
import threading
from typing import Coroutine


class ThreadedSyncRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        """Start the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_async_task(self, task: Coroutine):
        """Run an async task in the separate event loop thread.

        Args:
            task: The coroutine to run in the event loop.

        Returns:
            The result of the coroutine execution.
        """
        future = asyncio.run_coroutine_threadsafe(task, self.loop)
        return future.result()
