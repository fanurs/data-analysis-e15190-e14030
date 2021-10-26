import pytest

import tempfile

import numpy as np
import pandas as pd

from e15190.runlog import query

class TestElogQuery:
    def test___init__(self):
        q = query.ElogQuery(
            load_run_batches=False,
            update_run_batches=False,
            save_run_batches=False,
        )
        assert isinstance(q, query.ElogQuery)
        assert isinstance(q.df, pd.DataFrame)
        assert isinstance(q.batch_unified_properties, list)
        assert isinstance(q.max_run_gap, np.timedelta64)

    def test_determine_run_batches(self):
        q = query.ElogQuery(
            load_run_batches=False,
            update_run_batches=False,
            save_run_batches=False,
        )
        assert q.df.index.names == [None]

        q.determine_run_batches()
        assert q.df.index.names == ['ibatch', 'irun']

        ibatches = list(q.df.index.get_level_values('ibatch').unique())
        assert ibatches == list(range(len(ibatches)))

        iruns = list(q.df.index.get_level_values('irun').unique())
        assert iruns == list(range(len(iruns)))

        for ibatch in range(len(ibatches)):
            df_batch = q.df.loc[ibatch]

            # check uniqueness of unified properties
            for property in q.batch_unified_properties:
                assert len(df_batch[property].unique()) == 1
        
            # check maximum run gap
            for i in range(len(df_batch) - 1):
                assert df_batch['begin_time'].iloc[i + 1] - df_batch['end_time'].iloc[i] <= q.max_run_gap
    
    def test_get_run_batches_summary(self):
        q = query.ElogQuery(
            load_run_batches=False,
            update_run_batches=False,
            save_run_batches=False,
        )
        q.determine_run_batches()
        assert q.df_batches is None

        df_batches = q.get_run_batches_summary()
        assert df_batches.equals(q.df_batches)
        assert isinstance(df_batches, pd.DataFrame)
        assert df_batches.index.names == ['ibatch']
        assert len(df_batches) == len(q.df.index.get_level_values('ibatch').unique())

    def test_save_run_batches(self):
        q = query.ElogQuery(
            load_run_batches=False,
            update_run_batches=True,
            save_run_batches=False,
        )

        path = tempfile.NamedTemporaryFile(suffix='.csv').name
        q.save_run_batches(path)

        df_batches = pd.read_csv(path)
        assert len(df_batches) == len(q.df.index.get_level_values('ibatch').unique())
        assert df_batches['ibatch'].to_list() == list(range(len(df_batches)))
    
    def test_load_run_batches(self):
        q = query.ElogQuery(
            load_run_batches=False,
            update_run_batches=False,
            save_run_batches=False,
        )

        assert q.df_batches is None
        q.load_run_batches()
        assert isinstance(q.df_batches, pd.DataFrame)
        assert q.df_batches['run_min'].dtype == int
        assert q.df_batches['trigger_rate_mean'].dtype == float
        assert 'datetime' in str(q.df_batches['begin_time'].dtype).lower()
        assert 'datetime' in str(q.df_batches['end_time'].dtype).lower()
        assert 'timedelta' in str(q.df_batches['total_elapse'].dtype).lower()
        assert 'timedelta' in str(q.df_batches['total_gap_time'].dtype).lower()



@pytest.fixture
def run_2200():
    return dict(
        run=2200,
        begin_time=pd.Timestamp('2018-02-13 18:09:02'),
        end_time=pd.Timestamp('2018-02-13 18:36:39'),
        elapse=pd.Timedelta('0 days 00:27:37'),
        target='Ni58',
        beam='Ca40',
        beam_energy=140.0,
        shadow_bar='out',
        trigger_rate=2112.0,
    )

@pytest.fixture
def run_4100():
    return dict(
        run=4100,
        begin_time=pd.Timestamp('2018-03-11 03:40:40'),
        end_time=pd.Timestamp('2018-03-11 04:12:19'),
        elapse=pd.Timedelta('0 days 00:31:39'),
        target='Ni64',
        beam='Ca48',
        beam_energy=140.0,
        shadow_bar='in',
        trigger_rate=2463.0,
    )

class TestQuery:
    def test__get_run_query(self, run_2200, run_4100):
        entry = query.Query._get_run_query(2200)
        assert isinstance(entry, pd.Series)
        for key, value in run_2200.items():
            assert entry[key] == value
        assert isinstance(entry['comment'], str)

        entry = query.Query._get_run_query(4100)
        assert isinstance(entry, pd.Series)
        for key, value in run_4100.items():
            assert entry[key] == value
        assert isinstance(entry['comment'], str)
    
    def test_get_run_info(run, run_2200, run_4100):
        info = query.Query.get_run_info(2200)
        assert isinstance(info, dict)
        for key, value in run_2200.items():
            assert info[key] == value
        assert isinstance(info['comment'], str)

        info = query.Query.get_run_info(4100)
        assert isinstance(info, dict)
        for key, value in run_4100.items():
            assert info[key] == value
        assert isinstance(info['comment'], str)

    def test__get_batch_query(self):
        for ibatch in query.Query.elog.df_batches.index.get_level_values('ibatch'):
            entry = query.Query._get_batch_query(ibatch)
            assert isinstance(entry, pd.DataFrame)
            assert entry.index.names == ['irun']
    
    def test_get_batch(self):
        for ibatch in query.Query.elog.df_batches.index.get_level_values('ibatch'):
            batch = query.Query.get_batch(ibatch)
            assert isinstance(batch, pd.DataFrame)
            assert batch.index.names == [None]
            assert batch.index.values.tolist() == list(range(len(batch)))
    
    def test_get_batch_info(self):
        for ibatch in query.Query.elog.df_batches.index.get_level_values('ibatch'):
            info = query.Query.get_batch_info(ibatch)
            assert isinstance(info, dict)
            assert info['ibatch'] == ibatch
            assert isinstance(info['comment'], (str, list))

            info = query.Query.get_batch_info(ibatch, include_comments=False)
            assert info['comment'] is None
    
    def test_get_n_batches(self):
        assert query.Query.get_n_batches() == len(query.Query.elog.df_batches)
        assert query.Query.get_n_batches() >= 60
    
    def test_targets(self):
        assert set(query.Query.targets()) == set(['Ni58', 'Ni64', 'Sn112', 'Sn124'])
    
    def test_beams(self):
        assert set(query.Query.beams()) == set(['Ca40', 'Ca48'])
    
    def test_beam_energies(self):
        assert set(query.Query.beam_energies()) == set([56.0, 140.0])
    
    def test_select_runs(self):
        # example 1
        runs = query.Query.select_runs('target == "Ni58"')
        assert 2134 in runs
        assert 2135 in runs
        assert 2136 not in runs
        assert 2142 in runs
        assert 4617 in runs
        assert 4618 in runs
        assert 4619 in runs
        assert min(runs) == runs[0] == 2134
        assert max(runs) == runs[-1] == 4619

        # example 2
        runs = query.Query.select_runs(
            ' & '.join([
                'beam == "Ca40"',
                'target == "Ni64"',
                'beam_energy == 56.0',
                'shadow_bar == "in"',
            ])
        )
        expected_runs = list(range(2512, 2523 + 1))
        expected_runs.remove(2516)
        assert runs == expected_runs

        # example 3
        runs = query.Query.select_runs(
            'beam == "Ca48" & beam_energy == 56',
            comment_cut='mb singles 301',
            case=False,
        )
        assert runs[:3] == [4593, 4594, 4595]
        assert runs[-3:] == [4659, 4660, 4661]

    def test_select_batches(self):
        ibatches = query.Query.select_batches('n_runs > 35')
        assert ibatches == [22, 33, 49]
