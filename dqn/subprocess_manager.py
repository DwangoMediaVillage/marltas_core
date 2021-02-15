"""Helper class to run workers on child processes"""
import logging
import multiprocessing as mp
from typing import Callable


class SubprocessManager:
    """Helper class to run workers on child processes"""
    def __init__(self, logger: logging.Logger = logging.getLogger(__name__)):
        self.logger = logger
        self.ps = []

    def append_worker(self, f: Callable[[], None]) -> None:
        """Run function `f` on subprocess.

        Args:
            f: Function to be run.
        """
        p = mp.Process(target=f, daemon=True)
        p.start()
        self.ps.append(p)

    @property
    def workers_alive(self) -> bool:
        """Returns all the processes are alive or not"""
        assert len(self.ps), 'Empty worker'
        return all([p.is_alive() for p in self.ps])

    def finalize(self) -> None:
        """Kill forked subprocesses"""
        [p.terminate() for p in self.ps]
        [p.join() for p in self.ps]
