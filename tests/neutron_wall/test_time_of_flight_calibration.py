import pytest

from pathlib import Path

from e15190.neutron_wall import time_of_flight_calibration as tic

class TestTimeOfFlightCalibrator:
    def test_get_input_path(self):
        func = tic.TimeOfFlightCalibrator.get_input_path

        assert func(1).name == 'CalibratedData_0001.root'
        assert func(12).name == 'CalibratedData_0012.root'
        assert func(123).name == 'CalibratedData_0123.root'
        assert func(1234).name == 'CalibratedData_1234.root'

        assert func(1, directory='.') == Path('.', 'CalibratedData_0001.root')
        assert func(12, directory='.') == Path('.', 'CalibratedData_0012.root')
        assert func(123, directory='.') == Path('.', 'CalibratedData_0123.root')
        assert func(1234, directory='.') == Path('.', 'CalibratedData_1234.root')

        assert func(1, filename_fmt='testrun_%03d.root').name == 'testrun_001.root'
        assert func(12, filename_fmt='testrun_%03d.root').name == 'testrun_012.root'
        assert func(123, filename_fmt='testrun_%03d.root').name == 'testrun_123.root'


