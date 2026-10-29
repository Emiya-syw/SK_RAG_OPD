"""Shared request pacing and adaptive concurrency for online judge workers."""
import threading
import time


def terminal_error(error):
    return 'data_inspection_failed' in str(error)


class RequestGate:
    def __init__(self, maximum=16, initial=8, interval=.5):
        self.maximum = maximum
        self.capacity = min(initial, maximum)
        self.interval = interval
        self.active = self.successes = 0
        self.next_start = self.cooldown = 0.
        self.condition = threading.Condition()

    def acquire(self):
        with self.condition:
            while True:
                now = time.monotonic()
                delay = max(self.next_start, self.cooldown) - now
                if self.active < self.capacity and delay <= 0:
                    self.active += 1
                    self.next_start = now + self.interval
                    return
                self.condition.wait(timeout=max(.05, min(delay, 1.)) if delay > 0 else .2)

    def release(self, success=False, limited=False, retry_after=0.):
        with self.condition:
            self.active -= 1
            if limited:
                self.capacity = max(1, self.capacity // 2)
                self.interval = min(10., self.interval * 2)
                self.cooldown = max(self.cooldown, time.monotonic() + max(5., retry_after))
                self.successes = 0
                print(f'[JUDGE] 429 backoff: concurrency={self.capacity}, interval={self.interval}s', flush=True)
            elif success:
                self.successes += 1
                if self.successes >= 32:
                    self.capacity = min(self.maximum, self.capacity + 2)
                    self.interval = max(.5, self.interval * .9)
                    self.successes = 0
                    print(f'[JUDGE] concurrency={self.capacity}, interval={self.interval:.2f}s', flush=True)
            self.condition.notify_all()
