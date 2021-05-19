import time

class VerboseTimer:
    def __init__(self, auto_print=True):
        self.time_start = None
        self.time_stop = None
        self.time_diff = None

        self.auto_print = auto_print
        self.print_prefix = ''
        self.print_suffix = ''
        self.print_kwargs = {'flush': True}

    def start(self):
        self.time_start = time.perf_counter()

    def stop(self):
        self.time_stop = time.perf_counter()

        if self.time_start is None:
            raise Exception('Timer cannot be stopped. Timer was not started.')

        self.time_diff = self.time_stop - self.time_start
        self.reset()

        if self.auto_print:
            self.print()

    def reset(self):
        self.time_start = None
        self.time_stop = None

    def print(self):
        print(self.print_prefix + str(self.time_diff) + self.print_suffix, **self.print_kwargs)

verbose_timer = VerboseTimer()
def start():
    global verbose_timer
    verbose_timer.start()

def stop():
    global verbose_timer
    verbose_timer.stop()