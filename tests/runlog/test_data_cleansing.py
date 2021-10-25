import pytest

import contextlib
import io
import tempfile

from e15190.runlog import data_cleansing

class TestElogCleanser:
    def test___init__(self):
        cleanser = data_cleansing.ElogCleanser()
        assert cleanser.elog_path.name == 'elog.html'