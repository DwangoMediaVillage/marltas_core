"""Helper class to run an actor on child process."""
import logging
import multiprocessing as mp
from functools import partial
from queue import Queue
from typing import Callable


class ActorRunner:
    """Actor runner. To start actors' sampling task you need to not only init Runner object but also call `start()` to kick gRPC client initialization and start the main task.

    Args:
        n_processes: Number of child processes,
        run_actor_func: Function to init an actor.
    """
    def __init__(self,
                 n_processes: int,
                 run_actor_func: Callable[[int, Queue], None],
                 logger: logging.Logger = logging.getLogger(__name__)):
        self.n_processes = n_processes
        self.logger = logger

        # init actor subprocesses
        self.ps = []
        self.start_queue = []
        for i in range(n_processes):
            queue = mp.Queue(maxsize=1)
            p = mp.Process(target=partial(run_actor_func, process_index=i, start_queue=queue), daemon=True)
            p.start()
            self.ps.append(p)
            self.start_queue.append(queue)

    @property
    def workers_alive(self) -> bool:
        """Returns actor worker processes are alive or not."""
        return self.n_processes == 0 or all([p.is_alive() for p in self.ps])

    def start(self) -> None:
        """Run child process tasks."""
        if self.n_processes > 0:
            [q.put(True) for q in self.start_queue]

    def finalize(self) -> None:
        """Finalize processes."""
        [p.terminate() for p in self.ps]
        [p.join() for p in self.ps]
