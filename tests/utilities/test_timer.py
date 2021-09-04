import pytest

import time

from e15190.utilities import timer

class TestGlobalFunctions:
    def test_start_and_stop(self):
        timer.start()
        time.sleep(0.01)
        elapsed_time = timer.stop()

        # check if elapsed time is close to 0.01 seconds
        assert 0.01 - 0.005 < elapsed_time < 0.01 + 0.005
    
    def test_start_and_pause_and_resume(self):
        timer.start()
        time.sleep(0.01)
        timer.pause()
        time.sleep(0.01)
        timer.start()
        time.sleep(0.01)
        elapsed_time = timer.stop()

        # check if elapsed time is close to 0.02 seconds
        assert 0.02 - 0.005 < elapsed_time < 0.02 + 0.005

    def test_reset(self):
        timer.start()
        time.sleep(0.01)
        timer.reset()
        timer.start()
        time.sleep(0.01)
        elapsed_time = timer.stop()

        # check if elapsed time is close to 0.01 seconds
        assert 0.01 - 0.005 < elapsed_time < 0.01 + 0.005
    
    def test_stopping_without_starting(self):
        with pytest.raises(RuntimeError) as err:
            timer.stop()
    
    def test_starting_without_stopping(self):
        timer.start()
        with pytest.raises(RuntimeError) as err:
            timer.start()
