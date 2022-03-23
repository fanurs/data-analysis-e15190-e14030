import pytest

from pathlib import Path

import numpy as np
import pandas as pd

from e15190.runlog import downloader
from e15190.runlog.data_cleansing import ElogCleanser, MySqlCleanser
from .. import conftest

@pytest.fixture
def empty_elog_cleanser():
    return ElogCleanser(elog_path='', read_elog=False)

class TestElogCleanser:
    @pytest.fixture(scope='class', autouse=True)
    def setup_teardown(self):
        rel_path = Path(downloader.ElogDownloader.DOWNLOAD_PATH)
        rel_path = Path(*rel_path.parts[1:])
        if not conftest.copy_from_default_database(rel_path):
            dl = downloader.ElogDownloader()
            dl.download()

        yield

        pass

    def test___init__(self):
        cleanser = ElogCleanser(read_elog=False)
        assert isinstance(cleanser.elog_path, Path)
    
    def test__set_first_row_as_headers(self):
        df = pd.DataFrame(
            [
                ['run', 'time', 'event'],
                ['1', '12:00:00', 'start'],
                ['2', '12:05:00', 'stop'],
            ],
        )
        ElogCleanser._set_first_row_as_headers(df)
        assert df.columns.tolist() == ['run', 'time', 'event']
        assert len(df) == 2
    
    def test__convert_all_entries_to_str(self):
        df = pd.DataFrame(
            [
                [1, pd.to_datetime('12:00:00'), 5, 6, 'remark', 'comment'],
                [2, pd.to_datetime('12:05:00'), np.nan, None, np.nan, None],
            ],
            columns=['run', 'time', 'num1', 'num2', 'str1', 'str2'],
        )
        ElogCleanser._convert_all_entries_to_str(df)
        assert df.dtypes.tolist() == ['object'] * 6
        assert df.loc[0, 'run'] == '1'
        assert df.loc[1, 'run'] == '2'
        assert '12:00:00' in df.loc[0, 'time']
        assert '12:05:00' in df.loc[1, 'time']
        assert df.loc[0, 'num1'] == '5.0'
        assert df.loc[1, 'num1'] == 'nan'
        assert df.loc[0, 'num2'] == '6.0'
        assert df.loc[1, 'num2'] == 'nan' # None -> float('nan') -> 'nan'
        assert df.loc[0, 'str1'] == 'remark'
        assert df.loc[1, 'str1'] == 'nan'
        assert df.loc[0, 'str2'] == 'comment'
        assert df.loc[1, 'str2'] == 'None' # None -> str(None) -> 'None'
    
    def test__rename_headers(self):
        content = lambda ncols: [
            [f'dummy{i}' for i in range(ncols)],
            [f'dimmy{i}' for i in range(ncols)],
        ]
        df = pd.DataFrame(
            content(11),
            columns=[
                'RUN',
                'Begin time',
                'End time',
                'Elapse',
                'DAQ',
                'Type',
                'Target',
                'Beam',
                'Shadow bars',
                'Trigger rate',
                'Comments',
            ],
        )
        ElogCleanser._rename_headers(df)
        assert list(df.columns) == [
            'run',
            'begin_time',
            'end_time',
            'elapse',
            'daq',
            'type',
            'target',
            'beam',
            'shadow_bar',
            'trigger_rate',
            'comment',
        ]

        df = pd.DataFrame(content(2), columns=['RUN', 'Begin time'])
        with pytest.raises(ValueError) as excinfo:
            ElogCleanser._rename_headers(df)
            assert 'Expected columns' in str(excinfo.value)
            assert 'but got' in str(excinfo.value)

        df = pd.DataFrame(content(2), columns=['dummy', 'Begin time'])
        with pytest.raises(ValueError) as excinfo:
            ElogCleanser._rename_headers(df)
            assert 'Expected columns' in str(excinfo.value)
            assert 'but got' in str(excinfo.value)

    def test__split_runs_and_events(self):
        df = pd.DataFrame(
            [
                ['E1', '11:59:00', 'ready'],
                ['1', '12:00:00', 'start'],
                ['2', '12:02:00', 'stop'],
                ['3', '12:05:00', 'start'],
                ['E2', '12:06:00', 'error'],
            ],
            columns=['run', 'time', 'event'],
        )
        runs, events = ElogCleanser._split_runs_and_events(df)

        assert len(runs) == 3
        assert runs.loc[1, 'run'] == '1'
        assert runs.loc[2, 'time'] == '12:02:00'
        assert runs.loc[3, 'event'] == 'start'

        assert len(events) == 2
        assert events.loc[0, 'run'] == 'E1'
        assert events.loc[4, 'event'] == 'error'
    
    def test_smoke(self):
        cleanser = ElogCleanser(read_elog=True)
        cleanser.cleanse()
        cleanser.filtered_runs()
        paths = cleanser.save_cleansed_elog()
        for path in paths:
            assert path.is_file()
            assert 'runlog/cleansed' in str(path)
        for ext in ['h5', 'csv']:
            path = cleanser.save_filtered_runs(ext)
            assert path.is_file()
            assert 'runlog' in str(path)
            assert 'runlog/cleansed' not in str(path)

@pytest.mark.skipif(True, reason='to-do')
class TestMySqlCleanser:
    def test___init__(self):
        cleanser = MySqlCleanser()