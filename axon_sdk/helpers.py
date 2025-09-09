import contextlib
import time


class Timing(contextlib.ContextDecorator):
    def __init__(self, its=None):
        self.its = its

    def __enter__(self):
        self.start = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter_ns() - self.start
        print(f"Execution time: {self.elapsed * 1e-6:6.2f} ms")
        if self.its:
            print(f"  {(self.elapsed * 1e-6)/self.its:6.2f} ms/it")
