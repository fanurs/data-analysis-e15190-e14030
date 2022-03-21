import pytest

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from e15190.runlog import downloader

class TestMySqlDownloader:
    def test___init__(self):
        dl = downloader.MySqlDownloader(auto_connect=False)
        assert isinstance(dl, downloader.MySqlDownloader)
    
    def test_decorate(self):
        def func():
            return (
                (1, 1),
                (2, 3, 5),
                (8),
            )
        deco_func = downloader.MySqlDownloader.decorate(func)
        result = deco_func()
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == [1, 1]
        assert result[1] == [2, 3, 5]
        assert result[2] == 8 # reduced to a single value if length equals to 1
    
    def test_connect_and_disconnect(self, capsys):
        # without context manager
        dl = downloader.MySqlDownloader(auto_connect=False)
        dl.connect()
        dl.disconnect()
        stdout = capsys.readouterr().out.split('\n')
        assert 'established' in stdout[0]
        assert 'closed' in stdout[1]
    
        # with context manager
        with downloader.MySqlDownloader(auto_connect=False) as dl:
            pass
        assert 'no connection' in capsys.readouterr().out.lower()

        with downloader.MySqlDownloader(auto_connect=False) as dl:
                dl.connect()
        stdout = capsys.readouterr().out.split('\n')
        assert 'established' in stdout[0]
        assert 'closed' in stdout[1]

        with downloader.MySqlDownloader(auto_connect=True) as dl:
            pass
        stdout = capsys.readouterr().out.split('\n')
        assert 'established' in stdout[0]
        assert 'closed' in stdout[1]

    def test_get_all_table_names(self):
        with downloader.MySqlDownloader(auto_connect=True) as dl:
            tnames = dl.get_all_table_names()
            assert set(tnames) == set([
                'comments',
                'location',
                'module',
                'mtypes',
                'runbeam',
                'runbeamintensity',
                'runcomment',
                'runinfo',
                'runinfo2',
                'runinfo3',
                'runlog',
                'runscalernames',
                'runscalers',
                'runtarget',
                'users',
                'vendors',
            ])

    def test_get_table(self):
        with downloader.MySqlDownloader(auto_connect=True) as dl:
            df = dl.get_table('runtarget')
            assert isinstance(df, pd.DataFrame)
            assert list(df['tid']) == list(range(1, 8 + 1))

            with pytest.raises(ValueError) as excinfo:
                df = dl.get_table('dummy')
                assert 'dummy' in str(excinfo.value)
                assert 'not found' in str(excinfo.value)
        return df
    
    @pytest.mark.parametrize(
        'path',
        [None, NamedTemporaryFile(suffix='.h5').name]
    )
    def test_download_fresh(self, path):
        with downloader.MySqlDownloader(auto_connect=True) as dl:
            path = dl.download(
                download_path=path,
                table_names=['vendors', 'users', 'mtypes'],
            )
        
        with pd.HDFStore(path) as file:
            assert '/vendors' in file.keys()
            assert len(file.get('vendors')) == 21
            assert '/users' in file.keys()
            assert len(file.get('users')) == 11
            assert '/mtypes' in file.keys()
            assert len(file.get('mtypes')) == 18
            assert '/runlog' not in file.keys()
            assert '/runinfo' not in file.keys()
            assert '/runbeamintensity' not in file.keys()
    
    def test_download_file_already_exist(self, capsys, monkeypatch):
        path = NamedTemporaryFile(suffix='.h5').name
        path = Path(path)
        path.touch(exist_ok=True)

        monkeypatch.setattr('builtins.input', lambda _: 'y')
        with downloader.MySqlDownloader(auto_connect=True) as dl:
            dl.download(download_path=path, table_names=['vendors'])
        stdout = capsys.readouterr().out
        assert 'attempting to download run log' in stdout.lower()
        assert 'no re-download will be performed' not in stdout.lower()
        assert 'converting and saving' in stdout.lower()
        assert 'all tables have been saved' in stdout.lower()

        monkeypatch.setattr('builtins.input', lambda _: 'n')
        with downloader.MySqlDownloader(auto_connect=True) as dl:
            dl.download(download_path=path, table_names=['vendors'])
        stdout = capsys.readouterr().out
        assert 'attempting to download run log' in stdout.lower()
        assert 'no re-download will be performed' in stdout.lower()
        assert 'converting and saving' not in stdout.lower()
        assert 'all tables have been saved' not in stdout.lower()
    
    def test_download_auto_disconnect(self, capsys):
        path = NamedTemporaryFile(suffix='.h5').name
        dl = downloader.MySqlDownloader(auto_connect=False)
        dl.connect()
        stdout = capsys.readouterr().out
        assert 'established' in stdout

        dl.download(download_path=path, table_names=['vendors'], auto_disconnect=True)
        stdout = capsys.readouterr().out
        assert 'closed' in stdout

class TestElogDownloader:
    def test___init__(self):
        dl = downloader.ElogDownloader()
        assert isinstance(dl, downloader.ElogDownloader)
    
    @pytest.mark.parametrize(
        'path',
        [None, NamedTemporaryFile(suffix='.html').name]
    )
    def test_download(self, path):
        dl = downloader.ElogDownloader()
        nbytes = 1024
        path = dl.download(path, read_nbytes=nbytes)
        assert Path(path).stat().st_size == pytest.approx(nbytes)
        with open(path, 'r') as file:
            assert '<html>' in file.read()