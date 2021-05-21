import time

class VerboseTimer:
    def __init__(self, auto_print=True):
        self.status = 'stop' # 'start', 'pause'
        self.time_start = 0.0
        self.time_stop = 0.0
        self.time_diff = 0.0

        self.auto_print = auto_print
        self.print_prefix = ''
        self.print_suffix = ''
        self.print_kwargs = {'flush': True}

    def start(self):
        time_now = time.perf_counter()

        if self.status == 'start':
            raise Exception('Timer had been started, it cannot be started again.')
        elif self.status == 'pause' or self.status == 'stop':
            self.time_start = time_now

        self.status = 'start'

    def pause(self, show=False):
        time_now = time.perf_counter()

        if self.status == 'start':
            self.time_diff += time_now - self.time_start
        elif self.status == 'pause':
            raise Exception('Timer cannot be paused. Timer had been paused.')
        elif self.status == 'stop':
            raise Exception('Timer cannot be paused. Timer was not started.')

        if show:
            self.print()
        self.status = 'pause'

    def stop(self, show=None, prefix=None, suffix=None):
        time_now = time.perf_counter()

        if self.status == 'start':
            self.time_diff += time_now - self.time_start
        if self.status == 'pause':
            self.time_diff += 0.0
        if self.status == 'stop':
            raise Exception('Timer cannot be stopped. Timer had been stopped.')

        if show is True:
            self.print()
        elif show is False:
            pass
        elif self.auto_print:
            self.print(prefix=prefix, suffix=suffix)

        self.reset()
        self.status = 'stop'

    def reset(self):
        self.status = 'stop'
        self.time_start = 0.0
        self.time_stop = 0.0
        self.time_diff = 0.0

    def print(self, prefix=None, suffix=None):
        if prefix is None:
            prefix = self.print_prefix
        if suffix is None:
            suffix = self.print_suffix
        print(prefix + str(self.time_diff) + suffix, **self.print_kwargs)

verbose_timer = VerboseTimer()
def start(*args, **kwargs):
    global verbose_timer
    verbose_timer.start()

def pause(*args, **kwargs):
    global verbose_timer
    verbose_timer.pause()

def stop(*args, **kwargs):
    global verbose_timer
    verbose_timer.stop(*args, **kwargs)

def reset(*args, **kwargs):
    global verbose_timer
    verbose_timer.reset(*args, **kwargs)