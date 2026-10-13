"""Lifecycle helpers for subprocess groups owned by evaluation coordinators."""
import os
import signal
import subprocess
import time


def stop_processes(processes, timeout=5.0):
    """Terminate only our own process groups, then reap every direct child."""
    processes = [p for p in processes if p is not None]
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def handle_termination(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')
