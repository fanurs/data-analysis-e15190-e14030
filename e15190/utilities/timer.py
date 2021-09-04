import copy
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
            raise RuntimeError('Timer had been started, it cannot be started again.')
        elif self.status == 'pause' or self.status == 'stop':
            self.time_start = time_now

        self.status = 'start'

    def pause(self, show=False):
        time_now = time.perf_counter()

        if self.status == 'start':
            self.time_diff += time_now - self.time_start
        elif self.status == 'pause':
            raise RuntimeError('Timer cannot be paused. Timer had been paused.')
        elif self.status == 'stop':
            raise RuntimeError('Timer cannot be paused. Timer was not started.')

        if show:
            self.print()
        self.status = 'pause'

    def stop(self, prefix=None, suffix=None, show=None):
        time_now = time.perf_counter()

        if self.status == 'start':
            self.time_diff += time_now - self.time_start
        if self.status == 'pause':
            self.time_diff += 0.0
        if self.status == 'stop':
            raise RuntimeError('Timer cannot be stopped. Timer had been stopped.')

        if show is True:
            self.print()
        elif show is False:
            pass
        elif self.auto_print:
            self.print(prefix=prefix, suffix=suffix)

        elapse = copy.copy(self.time_diff)
        self.reset()
        self.status = 'stop'
        return elapse

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

# ---------------------------
# Some global functions for using the default timer
# ---------------------------
verbose_timer = VerboseTimer()
def start():
    """Start the timer.
    """
    global verbose_timer
    verbose_timer.start()

def pause(show=False):
    """Pause the timer.
    
    Parameters:
        show: bool, optional
            Whether to print the elapsed time accummulated at the time of
            pausing.
    """
    global verbose_timer
    verbose_timer.pause()

def stop(prefix=None, suffix=None, show=True):
    """Stop the timer.

    Parameters:
        prefix: str, optional
            Prefix to print before the elapsed time.
        suffix: str, optional
            Suffix to print after the elapsed time.
        show: bool, optional
            Whether to print the elapsed time.
        
    Returns:
        float: Elapsed time in seconds.
    """
    global verbose_timer
    return verbose_timer.stop(prefix=prefix, suffix=suffix, show=show)

def reset():
    """Reset the timer.
    """
    global verbose_timer
    verbose_timer.reset()