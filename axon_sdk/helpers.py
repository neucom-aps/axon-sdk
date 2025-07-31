import contextlib
import time


class Timing(contextlib.ContextDecorator):
    def __enter__(self):
        self.start = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.end = time.perf_counter_ns() - self.start
        print(f"Execution time: {self.end * 1e-6:6.2f} ms")
