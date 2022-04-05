import pytest

from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import uproot

from e15190.neutron_wall.cache import RunCache

@pytest.fixture
def runs():
    return [2048, 2222, 4096, 4444]

@pytest.fixture
def io_directories(runs):
    tmp_inroot_dir = tempfile.TemporaryDirectory()
    tmp_outroot_dir = tempfile.TemporaryDirectory()
    for run in runs:
        src = Path(__file__).parent / '../_samples/root_files/sample1.root'
        dst = Path(tmp_inroot_dir.name) / f'CalibratedData_{run:04d}.root'
        shutil.copy(src, dst)
    return tmp_inroot_dir, tmp_outroot_dir

@pytest.fixture
def run_cache(io_directories):
    return RunCache(
        str(Path(io_directories[0].name) / r'CalibratedData_{run:04d}.root'),
        str(Path(io_directories[1].name) / r'run-{run:04d}.db'),
        max_workers=1,
    )

class TestRunCache:
    def test___init__(self):
        rc = RunCache(
            r'../CalibratedData_{run:04d}.root',
            r'run-{run:04d}.db',
            max_workers=-1,
        )
        assert rc.SRC_PATH_FMT == r'../CalibratedData_{run:04d}.root'
        assert rc.CACHE_PATH_FMT == r'run-{run:04d}.db'
        assert rc.max_workers == -1
    
    def test_infer_tree_name(self):
        btypes = {'branch': int}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'one_tree.root'
            with uproot.recreate(path) as file:
                file.mktree('tree', btypes)
            with uproot.open(path) as file:
                assert RunCache.infer_tree_name(file) == 'tree'

            path = Path(directory) / 'one_tree_one_hist.root'
            with uproot.recreate(path) as file:
                file['hist'] = np.histogram(np.random.normal(size=100), bins=5, range=[-2, 2])
                file.mktree('tree', btypes)
            with uproot.open(path) as file:
                assert RunCache.infer_tree_name(file) == 'tree'

            path = Path(directory) / 'two_tree.root'
            with uproot.recreate(path) as file:
                file.mktree('tree1', btypes)
                file.mktree('tree2', btypes)
            with uproot.open(path) as file:
                with pytest.raises(ValueError) as excinfo:
                    RunCache.infer_tree_name(file)
                assert 'could not infer' in str(excinfo.value).lower()
    
    def test_read_run_from_root(self, run_cache, runs):
        rc = run_cache
        for run in runs:
            df = rc.read_run_from_root(run, ['i_evt', 'multi_0', 'multi_1', 'x_1', 'y_1'])
            assert isinstance(df, pd.DataFrame)
            assert all(df.columns == ['i_evt', 'multi_0', 'multi_1', 'x_1', 'y_1'])
            assert 0 not in set(df['multi_1']) # CAUTION!

        for run in set(np.random.choice(range(1000, 10000), size=5, replace=False)) - set(runs):
            with pytest.raises(FileNotFoundError):
                rc.read_run_from_root(run, ['i_evt'])
        
        for run in runs:
            with pytest.raises(ValueError) as excinfo:
                rc.read_run_from_root(run, ['i_evt', 'multi_0', 'x_0', 'multi_1', 'x_1'])
            assert 'cannot' in str(excinfo.value).lower()
            assert 'single dataframe' in str(excinfo.value).lower()
    
    def test_read_run_from_root_rename(self, run_cache, runs):
        rc = run_cache
        for run in runs:
            df = rc.read_run_from_root(
                run,
                {
                    'i_evt': 'i',
                    'multi_0': 'm0',
                    'multi_1': 'm1',
                    'x_1': 'x',
                    'y_1': 'y',
                }
            )
            assert all(df.columns == ['i', 'm0', 'm1', 'x', 'y'])
    
    def test__get_table_name(self):
        assert RunCache._get_table_name(1) == 'run0001'
        assert RunCache._get_table_name(12) == 'run0012'
        assert RunCache._get_table_name(123) == 'run0123'
        assert RunCache._get_table_name(1234) == 'run1234'
        assert RunCache._get_table_name(12345) == 'run12345'

    def test_save_run_to_sqlite(self, run_cache, runs):
        rc = run_cache
        _runs = [
            *runs,
            runs[-1], # to test if overwriting works; it should
        ]
        for run in _runs:
            df = rc.read_run_from_root(run, ['i_evt', 'multi_0', 'x_0', 'multi_1'])

            path = rc.save_run_to_sqlite(run, df)
            assert path.is_file()
            assert path.suffix == '.db'
            assert path.name == Path(rc.CACHE_PATH_FMT.format(run=run)).name

            with sqlite3.connect(str(path)) as conn:
                c = conn.cursor()
                tables = c.execute('SELECT name FROM sqlite_schema WHERE type="table"').fetchall()
                assert len(tables) == 1
                assert len(tables[0]) == 1
                table = tables[0][0]
                df_sqlite = pd.read_sql(f'SELECT * FROM "{table}"', conn)
                assert len(df_sqlite) == len(df)
                assert set(df.columns).issubset(set(df_sqlite.columns))
                assert set(df_sqlite.columns) - set(df.columns) == {'entry', 'subentry'}
                assert np.allclose(df_sqlite[df.columns].to_numpy(), df.to_numpy())
    
    def test_save_run_to_sqlite_mkdir(self, io_directories, runs):
        rc = RunCache(
            str(Path(io_directories[0].name) / r'CalibratedData_{run:04d}.root'),
            str(Path(io_directories[1].name) / 'dummy' / r'run-{run:04d}.db'),
            max_workers=1,
        )
        for run in runs:
            df = rc.read_run_from_root(run, ['i_evt', 'multi_0', 'x_0', 'multi_1'])
            path = rc.save_run_to_sqlite(run, df)
            assert 'dummy' in str(path)
            assert f'run-{run:04d}' in str(path)

    def test_read_run_from_sqlite(self, run_cache, runs):
        rc = run_cache
        _runs = [
            *runs,
            runs[-1], # to test if overwriting works; it should
        ]
        for run in _runs:
            df = rc.read_run_from_root(run, ['i_evt', 'multi_0', 'x_0', 'multi_1'])
            rc.save_run_to_sqlite(run, df)
            df_sqlite = rc.read_run_from_sqlite(run)
            assert len(df_sqlite) == len(df)
            assert all(df_sqlite.columns == df.columns)
            assert df_sqlite.index.names == df.index.names
            assert np.allclose(df_sqlite.to_numpy(), df.to_numpy())
            assert 0 in set(df_sqlite['multi_1'])
        for run in _runs:
            df = rc.read_run_from_root(run, ['i_evt', 'multi_0', 'x_0', 'multi_1'])
            rc.save_run_to_sqlite(run, df)
            df_sqlite = rc.read_run_from_sqlite(run, sql_cmd='WHERE multi_1 > 0')
            assert 0 not in set(df_sqlite['multi_1'])
    
    def test_read_run(self, run_cache, runs):
        rc = run_cache

        # when no cache yet
        for run in runs:
            assert not Path(rc.CACHE_PATH_FMT.format(run=run)).is_file()
            df = rc.read_run(run, ['i_evt', 'multi_0', 'x_0', 'multi_1'])
            assert 0 not in set(df['multi_0'])
            assert ['entry', 'subentry'] == list(df.index.names)
        
        # when cache has been created
        for run in runs:
            assert Path(rc.CACHE_PATH_FMT.format(run=run)).is_file()
            df = rc.read_run(run, ['i_evt', 'multi_1', 'x_1'])
            assert 'multi_0' in df.columns
            assert 'x_1' not in df.columns
            assert ['entry', 'subentry'] == list(df.index.names)
        
    def test_read_run_cache_options(self, run_cache, runs):
        rc = run_cache

        # when no cache yet
        for run in runs:
            df = rc.read_run(run, ['i_evt', 'multi_0', 'x_0', 'multi_1'], save_cache=False)
            assert all(df.columns == ['i_evt', 'multi_0', 'x_0', 'multi_1'])

        # there should be still no cache
        for run in runs:
            df = rc.read_run(run, ['i_evt', 'multi_1', 'x_1'])
            assert 'multi_0' not in df.columns
            assert 'multi_1' in df.columns
            assert 'x_1' in df.columns
        
        # cache has been create, but we don't read from it
        for run in runs:
            df = rc.read_run(run, ['i_evt', 'multi_0', 'x_0'], from_cache=False)
            assert 'multi_0' in df.columns
            assert 'x_0' in df.columns
            assert 'multi_1' not in df.columns
        
        # see if cache has been overwritten
        for run in runs:
            df = rc.read_run(run, None)
            assert 'multi_0' in df.columns
            assert 'x_0' in df.columns
            assert 'multi_1' not in df.columns

    def test_read(self, run_cache, runs):
        rc = run_cache

        df_single = dict()
        for run in runs:
            df_single[run] = rc.read_run(run, ['i_evt', 'multi_0', 'x_0', 'multi_1'])
        df = rc.read(runs, None)
        assert len(df) == sum([len(df_single[run]) for run in runs])
        assert all(df.columns == ['i_evt', 'multi_0', 'x_0', 'multi_1'])
        assert np.allclose(df['x_0'], np.ravel([df_single[run]['x_0'] for run in runs]))
        assert {'entry', 'subentry'} == set(df.index.names)
        assert 0 in set(df['multi_1'])
        assert 'run' not in df.columns

        # test single run
        df = rc.read(runs[0], None)
        assert np.allclose(df.to_numpy(), df_single[runs[0]].to_numpy())

    def test_read_options(self, run_cache, runs, capsys):
        rc = run_cache
        branches = ['i_evt', 'multi_0', 'x_0', 'multi_1']

        # test sql_cmd
        df = rc.read(runs, branches, sql_cmd='WHERE multi_1 > 0')
        assert 0 not in set(df['multi_1'])

        # test drop_columns
        df = rc.read(runs, branches, drop_columns=['i_evt', 'multi_1'])
        assert all(df.columns == ['multi_0', 'x_0'])

        # test reset_index = True
        df = rc.read(runs, branches, reset_index=True)
        assert 'entry' not in df.index.names
        assert 'subentry' not in df.index.names
        assert all(df.index == range(len(df)))

        # test insert_run_column = True
        df = rc.read(runs, branches, insert_run_column=True)
        assert list(df.columns).index('run') == 0

        # test verbose = True
        rc.read(runs, branches, verbose=True)
        stdout = capsys.readouterr().out
        assert len(stdout.split('\n')) == len(runs) + 1
        for run in runs:
            assert f'reading run {run}' in stdout.lower()
    
    def test_read_options_that_dont_affect_cache(self, run_cache, runs):
        rc = run_cache
        run = runs[0]

        # sql_cmd
        df = rc.read(
            run,
            ['i_evt', 'multi_0', 'multi_1', 'x_1', 'y_1'],
            sql_cmd='WHERE multi_0 > 0',
            from_cache=True,
            save_cache=True,
        )
        assert 0 not in set(df['multi_0'])
        df = rc.read_run_from_sqlite(run)
        assert 0 in set(df['multi_0'])

        # drop_columns
        df = rc.read(
            run,
            ['i_evt', 'multi_0', 'multi_1', 'x_1', 'y_1'],
            drop_columns=['i_evt'],
            from_cache=True,
            save_cache=True,
        )
        assert 'i_evt' not in df.columns
        df = rc.read_run_from_sqlite(run)
        assert 'i_evt' in df.columns
