import pytest

import contextlib
import io
import pathlib
import tempfile

import pandas as pd

from e15190.runlog import downloader

class TestMySqlDownloader:
    def test___init__(self):
        dl = downloader.MySqlDownloader(
            auto_connect=False,
            key_path=None,
        )
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
    
    def test_connect_and_disconnect(self):
        # connecting and disconnecting without context manager
        dl = downloader.MySqlDownloader(
            auto_connect=False,
            key_path=None,
        )
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            dl.connect()
            dl.disconnect()
        stdout = sio.getvalue().split('\n')
        assert 'established' in stdout[0]
        assert 'closed' in stdout[1]
    
        # connecting and disconnecting with context manager
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            with downloader.MySqlDownloader(auto_connect=False) as dl:
                pass
        assert sio.getvalue() == ''

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            with downloader.MySqlDownloader(auto_connect=False) as dl:
                dl.connect()
        stdout = sio.getvalue().split('\n')
        assert 'established' in stdout[0]
        assert 'closed' in stdout[1]

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            with downloader.MySqlDownloader(auto_connect=True) as dl:
                pass
        stdout = sio.getvalue().split('\n')
        assert 'established' in stdout[0]
        assert 'closed' in stdout[1]

        # connect without key
        with pytest.raises(FileNotFoundError):
            downloader.MySqlDownloader(
                auto_connect=True,
                key_path='non_existing_file',
            )

    def test_get_all_table_names(self):
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
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
    
    def test_download(self):
        sio = io.StringIO()
        tmp_path = tempfile.NamedTemporaryFile(suffix='.h5').name
        with contextlib.redirect_stdout(sio):
            with downloader.MySqlDownloader(auto_connect=True) as dl:
                dl.download(
                    download_path=tmp_path,
                    table_names=['vendors', 'users', 'mtypes'],
                )
        with pd.HDFStore(tmp_path, 'r') as file:
            assert '/vendors' in file.keys()
            assert len(file.get('vendors')) == 21
            assert '/users' in file.keys()
            assert len(file.get('users')) == 11
            assert '/mtypes' in file.keys()
            assert len(file.get('mtypes')) == 18
            assert '/runlog' not in file.keys()
            assert '/runinfo' not in file.keys()
            assert '/runbeamintensity' not in file.keys()

class TestElogDownloader:
    def test___init__(self):
        dl = downloader.ElogDownloader()
        assert isinstance(dl, downloader.ElogDownloader)
    
    def test_download(self):
        dl = downloader.ElogDownloader()

        sio = io.StringIO()
        tmp_path = tempfile.NamedTemporaryFile(suffix='.html').name
        with contextlib.redirect_stdout(sio):
            dl.download(download_path=tmp_path, read_nbytes=1024)
        assert pathlib.Path(tmp_path).stat().st_size == pytest.approx(1024)
        with open(tmp_path, 'r') as file:
            assert '<html>' in file.read()