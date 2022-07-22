import collections
import pathlib
import os
import sqlite3
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import e15190 # always needed


class MySqlQuery:
    downloaded_path = '$DATABASE_DIR/runlog/cleansed/mysql_database.db'

    def __init__(self):
        self.downloaded_path = pathlib.Path(os.path.expandvars(self.downloaded_path))

    @staticmethod
    def fetchall(result):
        return [ele[0] for ele in result.fetchall()]

    @property
    def table_names(self):
        with sqlite3.connect(self.downloaded_path) as conn:
            result = conn.execute('SELECT name FROM sqlite_master WHERE type="table"')
            return self.fetchall(result)
    
    def get_table(self, table_name=None, commands=None, **kwargs):
        with sqlite3.connect(self.downloaded_path) as conn:
            if table_name is not None:
                df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn, **kwargs)
            elif commands is not None:
                df = pd.read_sql_query(commands, conn, **kwargs)
            return df

    @staticmethod
    def runscalers_source_id(source):
        if source == 'hira' or source == 0:
            return 0
        if source == 'nw' or source == 1:
            return 1
        raise ValueError(f'Unknown source: {source}')

class ElogQuery:
    def __init__(
        self,
        load_run_batches=True,
        update_run_batches=False,
        save_run_batches=False,
    ):
        """Query for the Elog database.

        Parameters
        ----------
        load_run_batches : bool, default True
            Invoke :py:func:`load_run_batches` upon initialization.
        update_run_batches : bool, default False
            Invoke :py:func:`determine_run_batches`, then
            :py:func:`get_run_batches_summary` upon initialization. If ``True``,
            ``load_run_batches`` will be ignored, and the run batches will be
            re-determined.
        save_run_batches : bool, default False
            Invoke :py:func:`save_run_batches` upon initialization. This is
            useful when the run batches are re-determined and we want the new
            run batches to be saved into a CSV file.
        """
        self.path = '$DATABASE_DIR/runlog/elog_runs_filtered.h5'

        self.df = None
        """The Elog database pandas dataframe."""
        with pd.HDFStore(os.path.expandvars(self.path), 'r') as file:
            self.df = file['runs']
            self.df.sort_values('run', inplace=True, ignore_index=True)
        
        # split beam into beam (isotope) and beam energy (float)
        beam_cols = self.df['beam'].str.split(' ', expand=True)
        self.df['beam'] = beam_cols[0]
        self.df.insert(
            self.df.columns.get_loc('beam') + 1,
            'beam_energy',
            beam_cols[1].astype(float),
        )

        self.batch_unified_properties = ['target', 'beam', 'beam_energy', 'shadow_bar']
        """Properties that are unified across all runs in a single batch."""

        self.max_run_gap = np.timedelta64(2, 'h')
        """Maximum allowed gap between runs in a single batch."""

        self.df_batches = None
        """A summary of batch properties rather than run properties."""

        self.run_batches_path = '$DATABASE_DIR/runlog/run_batches.csv'

        if update_run_batches:
            self.determine_run_batches()
            self.get_run_batches_summary()
        elif load_run_batches:
            self.load_run_batches()
        if save_run_batches:
            self.save_run_batches()
    
    def determine_run_batches(
        self,
        batch_unified_properties=None,
        max_run_gap=None,
        additional_runs=None,
    ):
        """Determine the run batches.

        A *run batch* or simply *batch* is a group of runs that share the same
        properties like beam energy and target, so that merging the events
        across runs within the same batch makes physical sense. Run batches are
        useful when analyzing quantities (e.g. PSD parameters) that require
        statistics from multiple runs.

        This function tries to determine the run batches by comparing the
        properties of the runs. Additional batches can also be specified by the
        argument ``additional_runs``. At the end, ``self.df`` will be updated to
        use a 2-level index, namely, ``ibatch`` and ``irun``. The ``ibatch``
        index is the enumeration of run batches; the ``irun`` index is a
        consecutive enumeration of all runs across all batches. Both indices
        start from 0.
        
        While we might expect physical properties of the runs within one single
        batch to be comparable, there is no guarantee for that. As an example,
        consider a batch consists of run 12, 13, 14, 15. All the runs have the
        same reaction systems, Ca40 + Ni58 at 140 MeV/u, without shadow bars.
        However, there is no guarantee that, say, the position calibration
        parameters are similar for all four runs. Some hardware differences or
        DAQ settings might cause one of the runs to be very different from the
        rest. Further analysis is required to study those differences across
        runs.

        Parameters
        ----------
        batch_unified_properties : list of str, default None
            A list of properties that are unified across all runs in a single
            batch. This means that whenever one of these properties changes, a
            breakpoint will be set to separate the runs into different batches.
            Any column in ``self.df`` can be used. If ``None``, the function
            uses :py:attr:`batch_unified_properties`.
        max_run_gap : np.timedelta64, default None
            Maximum allowed gap between runs in a single batch. If ``None``, the
            function uses :py:attr:`max_run_gap`.
        additional_runs : list of int, default None
            A list of run numbers that will be used to create additional
            batches. The runs should specify the *first* run of a batch.
        """
        # a breakpoint is defined as the index of the last run in a batch
        # the indices strictly follow the ones in ``self.df``
        breakpoints = None

        # breakpoints by batch unified properties
        if batch_unified_properties is not None:
            self.batch_unified_properties = batch_unified_properties
        cols = self.batch_unified_properties # alias
        breakpoints = (self.df[cols] != self.df[cols].shift(-1)).any('columns')

        # breakpoints by maximum inter-run time gap
        if max_run_gap is not None:
            self.max_run_gap = max_run_gap
        run_gaps = (self.df['begin_time'].shift(-1) - self.df['end_time'])
        breakpoints = np.any([breakpoints, run_gaps > self.max_run_gap], axis=0)
        breakpoints = pd.Series(breakpoints)

        # breakpoints by additional run numbers
        additional_runs = [] if additional_runs is None else additional_runs
        for run in additional_runs:
            closest_run_index = self.df.query('run >= @run').index[0]
            bp_index = min(max(closest_run_index - 1, 0), len(breakpoints) - 1)
            breakpoints[bp_index] = True

        # finalize multi indices: ibatch, irun
        ibatch = breakpoints.shift(1).cumsum()
        ibatch[0] = 0
        indices = np.vstack([ibatch.astype(int), self.df.index])
        indices = pd.MultiIndex.from_arrays(indices, names=['ibatch', 'irun'])
        self.df.set_index(indices, inplace=True)
    
    def get_run_batches_summary(self):
        """Updates and returns the run batches summary.

        This function updates the dataframe :py:attr:`df_batches`, which
        contains the following columns:

        - ``ibatch`` (int): the enumeration of run batches (starting from 0)
        - ``run_min`` (int): first run number
        - ``run_max`` (int): last run number
        - ``n_runs`` (int): number of runs
        - ``n_skipped_runs`` (int): number of runs that are skipped
        - ``begin_time`` (Timestamp): begin time of the first run
        - ``end_time`` (Timestamp): end time of the last run
        - ``total_elapse`` (Timedelta): total elapsed time of all the runs
        - ``total_gap_time`` (Timedelta): total gap time between all the runs
        - ``target`` (str): e.g. Ni58, Ni64, Sn112, Sn124.
        - ``beam`` (str): e.g. Ca40, Ca48
        - ``beam_energy`` (float): beam energy in MeV/u
        - ``shadow_bar`` (str): shadow bar configuration, i.e. "in" or "out"
        - ``trigger_rate_min`` (float): minimum trigger rate
        - ``trigger_rate_max`` (float): maximum trigger rate
        - ``trigger_rate_mean`` (float): mean trigger rate

        Returns
        -------
        df_batches : pandas.DataFrame
            A dataframe that lists the batch properties.
        """
        self.df_batches = []
        for ibatch, batch in self.df.groupby('ibatch'):
            entry = dict(ibatch=ibatch)

            # run numbers
            entry['run_min'] = batch['run'].min()
            entry['run_max'] = batch['run'].max()
            entry['n_runs'] = len(batch)
            entry['n_skipped_runs'] = entry['run_max'] - entry['run_min'] + 1 - entry['n_runs']

            # times and elapses
            entry['begin_time'] = batch['begin_time'].min()
            entry['end_time'] = batch['end_time'].max()
            entry['total_elapse'] = batch['elapse'].sum()
            entry['total_gap_time'] = (entry['end_time'] - entry['begin_time']) - entry['total_elapse']

            # experiment configuration
            entry['target'] = batch['target'].iloc[0]
            entry['beam'] = batch['beam'].iloc[0]
            entry['beam_energy'] = batch['beam_energy'].iloc[0]
            entry['shadow_bar'] = batch['shadow_bar'].iloc[0]

            # trigger rates
            entry['trigger_rate_min'] = batch['trigger_rate'].min()
            entry['trigger_rate_max'] = batch['trigger_rate'].max()
            entry['trigger_rate_mean'] = batch['trigger_rate'].mean()

            self.df_batches.append(list(entry.values()))
        self.df_batches = pd.DataFrame(self.df_batches, columns=list(entry.keys()))
        self.df_batches.set_index('ibatch', drop=True, inplace=True)
        return self.df_batches

    def save_run_batches(self, filepath=None):
        """Save the run batches to a CSV file.

        Parameters
        ----------
        filepath : str or pathlib.Path, default None
            The path to the CSV file. If ``None``, the function uses
            :py:attr:`run_batches_path`.
        """
        filepath = pathlib.Path(filepath or os.path.expandvars(self.run_batches_path))
        self.df_batches.to_csv(filepath)

    def load_run_batches(self, filepath=None):
        """Load the run batches from a CSV file.

        Parameters
        ----------
        filepath : str or pathlib.Path, default None
            The path to the CSV file. If ``None``, the function uses
            :py:attr:`run_batches_path`.
        """
        filepath = pathlib.Path(filepath or os.path.expandvars(self.run_batches_path))
        self.df_batches = pd.read_csv(filepath)
        self.df_batches.set_index('ibatch', drop=True, inplace=True)

        # update the column types of ``self.df_batches``
        self.df_batches['begin_time'] = pd.to_datetime(self.df_batches['begin_time'])
        self.df_batches['end_time'] = pd.to_datetime(self.df_batches['end_time'])
        self.df_batches['total_elapse'] = pd.to_timedelta(self.df_batches['total_elapse'])
        self.df_batches['total_gap_time'] = pd.to_timedelta(self.df_batches['total_gap_time'])

        # update the index of ``self.df``
        last_runs = self.df_batches['run_max'].to_list()
        breakpoints = self.df['run'].isin(last_runs)
        ibatch = breakpoints.shift(1).cumsum()
        ibatch[0] = 0
        indices = np.vstack([ibatch.astype(int), self.df.index])
        indices = pd.MultiIndex.from_arrays(indices, names=['ibatch', 'irun'])
        self.df.set_index(indices, inplace=True)


class CosmicQuery:
    def __init__(self):
        path = '$DATABASE_DIR/runlog/cosmic_runs.csv'
        path = pathlib.Path(os.path.expandvars(path))
        self.df = pd.read_csv(path)
        self.df['begin_time'] = pd.to_datetime(self.df['begin_time'])
        self.df['end_time'] = pd.to_datetime(self.df['end_time'])
        self.df['elapse'] = pd.to_timedelta(self.df['elapse'])
    
    def get_all_runs(self):
        return sorted(self.df['run'].to_list())


class AmBeQuery:
    def __init__(self):
        path = '$DATABASE_DIR/runlog/ambe_runs.csv'
        path = pathlib.Path(os.path.expandvars(path))
        self.df = pd.read_csv(path)
        self.df['begin_time'] = pd.to_datetime(self.df['begin_time'])
        self.df['end_time'] = pd.to_datetime(self.df['end_time'])
        self.df['elapse'] = pd.to_timedelta(self.df['elapse'])
    
    def get_all_runs(self):
        return sorted(self.df['run'].to_list())


class Query:
    """A class of query methods for the database.

    Runlog queries are made from both the Elog database and the MySQL database.

    For Elog queries, one can either directly make a query on the
    :py:attr:`elog.df` dataframe or use functions like :py:func:`get_run_info`
    to get the run information.
    
    For MySQL queries, one can utilize the class :py:func:`MySqlQuery` or
    functions like :py:func:`get_runscalers`. Bear in mind that the entire MySQL
    database is huge. If you just want to explore the database, use clauses like
    LIMIT and WHERE to select a subset of the database for inspection.

    Examples
    --------
    Single run info can easily be queried as following:

    >>> from e15190.runlog.query import Query
    >>> Query.get_run_info(4100)
    {'run': 4100,
     'begin_time': Timestamp('2018-03-11 03:40:40'),
     'end_time': Timestamp('2018-03-11 04:12:19'),
     'elapse': Timedelta('0 days 00:31:39'),
     'target': 'Ni64',
     'beam': 'Ca48',
     'beam_energy': 140.0,
     'shadow_bar': 'in',
     'trigger_rate': 2463.0,
     'comment': '140MeV 64Ni, coincidence trigger, uB DS, RF trig'}
    
    Batch info is queried as following:

    >>> from e15190.runlog.query import Query
    >>> Query.get_batch_info(2)
    {'ibatch': 2,
     'run_range': [2142, 2152],
     'n_runs': 9,
     'n_skipped_runs': 2,
     'begin_time': Timestamp('2018-02-12 22:06:15'),
     'end_time': Timestamp('2018-02-13 03:25:30'),
     'total_elapse': Timedelta('0 days 04:39:29'),
     'total_gap_time': Timedelta('0 days 00:39:46'),
     'target': 'Ni58',
     'beam': 'Ca40',
     'beam_energy': 140.0,
     'shadow_bar': 'out',
     'trigger_rate_range': [1861.0, 2044.0],
     'trigger_rate_stdev': 70.8092115790343,
     'comment': ['Ni 58 Ca40 140MeV/u, coincidence data delayed triggers 150ns',
      'Ni 58 Ca40 140MeV/u, coincidence data delayed triggers 150ns(1) Before
      this run: 1) delayed hira+NW master by 150 ns, 2) dleayed fast clear by
      150ns; 3)increased the fast busy by 150 ns.']}
    
    Run scalers can be queried as following:

    >>> from e15190.runlog.query import Query
    >>> Query.get_runscalers(4100, 'nw')
          run              datetime  VW-TOP-OR  VW-BOT-OR  VW-TOP-BOT-OR  VW-GATE \\
    0    4100   2018-03-11 03:41:16      22180      10953          15913    88002   
    1    4100   2018-03-11 03:41:18      22357      10921          16020    87246   
    2    4100   2018-03-11 03:41:20      22450      11015          16040    86622   
    3    4100   2018-03-11 03:41:22      22343      11032          16013    86232   
    4    4100   2018-03-11 03:41:24      22285      10985          15920    86526   
    ..    ...                   ...        ...        ...            ...      ...   
    881  4100   2018-03-11 04:12:03      21756      10761          15676    86893   
    882  4100   2018-03-11 04:12:05      21774      10732          15644    87262   
    883  4100   2018-03-11 04:12:08      21759      10772          15713    86451   
    884  4100   2018-03-11 04:12:10      22263      11032          16027    86564   
    885  4100   2018-03-11 04:12:12      22271      10727          16023    88343   
    ---
         VW-FCLR  MASTER  MASTER2  VW-DELAY   FA-OR  DSS-OR  NW-RAW  NW-LIVE \\
    0      85224    5023     5023         0  808973       0  213598    88002   
    1      84507    4940     4940         0  804762       0  210173    87246   
    2      83929    4972     4972         0  800139       0  210713    86622   
    3      83446    5040     5040         0  796532       0  211062    86232   
    4      83829    5024     5024         0  796509       0  211055    86526   
    ..       ...     ...      ...       ...     ...     ...     ...      ...   
    881    84172    4929     4929         0  793584       0  207542    86893   
    882    84657    4882     4882         0  796243       0  207361    87262   
    883    83769    4907     4907         0  790993       0  206080    86451   
    884    83803    4986     4986         0  792586       0  208633    86564   
    885    85664    4871     4871         0  808603       0  208472    88343   

    """
    elog = ElogQuery(load_run_batches=True)
    """Class attribute :py:class:`ElogQuery` object loaded with run batches."""

    _map_run_iloc = {run: iloc for iloc, run in enumerate(elog.df['run'].to_list())}

    @staticmethod
    def get_ibatch(run) -> int:
        """Returns the batch number of the run

        Parameters
        ----------
        run : int
        """
        return Query.elog.df.query(f'run == {run}').index.get_level_values('ibatch').values[0]

    @staticmethod
    def _get_run_query(run) -> pd.DataFrame:
        """Returns sub-dataframe of :py:attr:`ElogQuery.df` in
        :py:attr:`Query.elog` for a given run.

        Parameters
        ----------
        run : int
            Run number.
        """
        return Query.elog.df.iloc[Query._map_run_iloc[run]]
    
    @staticmethod
    def get_run_info(run, include_ibatch=True) -> Dict[str, Any]:
        """Returns a dictionary of properties for a given run.
        
        Parameters
        ----------
        run : int
        """
        result = dict(Query._get_run_query(run))
        if include_ibatch:
            result['ibatch'] = Query.get_ibatch(run)
        return result
    
    @staticmethod
    def _get_batch_query(ibatch) -> pd.DataFrame:
        """Returns a sub-dataframe of :py:attr:`ElogQuery.df` in
        :py:attr:`Query.elog` for a given batch number.

        Parameters
        ----------
        ibatch : int
            Batch number (0-indexed).
        """
        return Query.elog.df.loc[ibatch]

    @staticmethod
    def is_good(run) -> bool:
        """Returns whether a given run is good.

        Parameters
        ----------
        run : int
            Run number.
        
        Returns
        -------
        is_good_ : bool
            Whether the run is good. Not good would mean this run should be
            skipped in data analysis.
        """
        return run in Query.elog.df.run.to_list()
    
    @staticmethod
    def are_good(runs) -> List[bool]:
        """Returns whether a list of runs are good.

        Parameters
        ----------
        runs : list
            List of runs.
        
        Returns
        -------
        are_good_ : list of bool
            Whether the runs are good. Bad would mean the runs should be skipped
            in data analysis.
        """
        all_good_runs = Query.elog.df.run.to_list()
        return [run in all_good_runs for run in runs]

    @staticmethod
    def get_batch(ibatch) -> pd.DataFrame:
        """Returns a sub-dataframe of :py:attr:`ElogQuery.df` in
        :py:attr:`Query.elog` for a given batch number.

        The sub-dataframe has been re-indexed from zero.

        Parameters
        ----------
        ibatch : int
            Batch number (0-indexed).
        """
        df = Query._get_batch_query(ibatch)
        return df.reset_index(drop=True)

    @staticmethod
    def get_batch_runs(ibatch) -> List[int]:
        """Returns runs in a given batch.
        
        Parameters
        ----------
        ibatch : int
            Batch number (0-indexed).
        """
        return Query.elog.df.query(f'ibatch == {ibatch}').run.to_list()
    
    @staticmethod
    def get_batch_info(ibatch, include_comments=True, include_runs=True) -> Dict[str, Any]:
        """Returns a dictionary of properties for a given batch.

        This function basically reads off from the
        :py:class:`ElogQuery.df_batches` dataframe.

        Parameters
        ----------
        ibatch : int
            Batch number (0-indexed).
        include_comments : bool, default True
            Whether to include the comments in the returned dictionary. If True,
            the comments are included as a list of strings, sorted from the most
            common to the least common; if there is only one comment, the list
            is reduced into a string. If False, the comments are not included,
            i.e. the key ``'comment'`` is mapped to None.
        include_runs : bool, default True
            Whether to include the runs in the returned dictionary. If True,
            the runs are included as a list of integers, sorted from the smallest
            to the largest. If False, the runs are not included, and the key
            ``'runs'`` will not be found in the returned dictionary.
        """
        result = dict(
            ibatch=ibatch,
            **Query.elog.df_batches.loc[ibatch],
        )

        if include_comments:
            df = Query.get_batch(ibatch)
            comment = collections.Counter(df['comment'])
            comment = sorted(comment.items(), key=lambda x: x[1], reverse=True)
            comment = [cmt for cmt, _ in comment]
            if len(comment) == 1:
                comment = comment[0]
            result['comment'] = comment
        else:
            result['comment'] = None

        if include_runs:
            result['runs'] = Query.get_batch_runs(ibatch)
        
        return result
    
    @staticmethod
    def get_n_batches() -> int:
        """Returns the number of batches in the database."""
        return len(Query.elog.df_batches)
    
    @staticmethod
    def targets() -> List[str]:
        """Returns all distinct target names in the database."""
        return Query.elog.df['target'].unique().tolist()
    
    @staticmethod
    def beams() -> List[str]:
        """Returns all distinct beam names in the database."""
        return Query.elog.df['beam'].unique().tolist()
    
    @staticmethod
    def beam_energies() -> List[float]:
        """Returns all distinct beam energies in the database."""
        return Query.elog.df['beam_energy'].unique().tolist()

    @staticmethod
    def select_runs(cut, comment_cut=None, **kw_comment) -> List[int]:
        """Returns a list of run numbers that satisfy the given cuts.

        Parameters
        ----------
        cut : str
            A string that can be evaluated as a boolean expression. The
            following variables are can be specified in the expression:
            - ibatch (int)
            - run (int)
            - begin_time (Timestamp)
            - end_time (Timestamp)
            - elapse (Timedelta)
            - target (str)
            - beam (str)
            - beam_energy (float)
            - shadow_bar ("in" or "out")
            - trigger_rate (float)
            - comment (str)
        comment_cut : str, default None
            Character sequence or regular expression to match the comment.

            While the variable ``comment`` can be specified in the argument ``cut``,
            it is usually more helpful to use ``comment_cut``, which invokes a
            simple wrapper around the
            `Series.str.contains <https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html>`__
            method. This allows more complex string matching like case insensitive
            matching and regular expressions.
        **kw_comment : dict
            Keyword arguments to be passed to the
            `Series.str.contains <https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html>`__
            method.
        
        Returns
        -------
        runs : list
            A list of run numbers that satisfy the given cuts.

        Examples
        --------
        As our first example, we select all runs with target Ni58:

        >>> from e15190.runlog.query import Query
        >>> Query.select_runs('target == "Ni58"')
        [2134, 2135, 2142, ..., 4617, 4618, 4619]


        In the second example, we select all runs with Ca40 + Ni58 at 140 MeV/u
        with shadow bars:

        >>> from e15190.runlog.query import Query
        >>> Query.select_runs('beam == "Ca40" & target == "Ni64"\\
        ...  & beam_energy == 56 & shadow_bar == "in"')
        [2512, 2513, 2514, 2515, 2517, 2518, 2519, 2520, 2521, 2522, 2523]
        

        In the last example, we select runs that have comments that contain
        substring ``'mb singles 301'``:

        >>> from e15190.runlog.query import Query
        >>> Query.select_runs(
        ...     'beam == "Ca48" & beam_energy == 56',
        ...      comment_cut='mb singles 301',
        ...      case=False, # case insensitive matching
        ... )
        [4593, 4594, 4595, ..., 4659, 4660, 4661]

        """
        subdf = Query.elog.df.query(cut)
        if comment_cut is not None:
            subdf = subdf.loc[subdf['comment'].str.contains(comment_cut, **kw_comment)]
        return subdf['run'].to_list()
    
    @staticmethod
    def select_batches(cut) -> List[int]:
        """Returns a list of batch numbers that satisfy the given cuts.

        Comment cut is not supported. But users can easily use this function to
        first select the batches of interest, then invoke
        :py:func:`Query.select_runs` to further select on the comments with more
        complicated matching patterns.

        Parameters
        ----------
        cut : str
            A string that can be evaluated as a boolean expression. The
            following variables are can be specified in the expression:

            - ``ibatch`` (int): batch number (0-indexed)
            - ``run_min`` (int): first run numbe
            - ``run_max`` (int): last run number
            - ``n_runs`` (int): number of runs in the batch
            - ``n_skipped_runs`` (int): number of runs skipped in the batch
            - ``begin_time`` (Timestamp): start time of the first run
            - ``end_time`` (Timestamp): end time of the last run
            - ``total_elapse`` (Timedelta): total elapsed time of all the runs
            - ``total_gap_time`` (Timedelta): total gap time of between all the runs
            - ``target`` (str): e.g. Ni58, Ni64, Sn112, Sn124
            - ``beam`` (str): e.g. Ca40, Ca48
            - ``beam_energy`` (float): beam energy in MeV/u
            - ``shadow_bar`` (str): "in" or "out"
            - ``trigger_rate_min`` (float): minimum trigger rate
            - ``trigger_rate_max`` (float): maximum trigger rate
            - ``trigger_rate_mean`` (float): mean trigger rate
        
        Returns
        -------
        ibatches : list
            A list of batch numbers that satisfy the given cuts.
        
        Examples
        --------
        >>> from e15190.runlog.query import Query
        >>> Query.select_batches('n_runs > 35')
        [22, 33, 49]
        """
        subdf = Query.elog.df_batches.query(cut)
        return subdf.index.to_list()

    @staticmethod
    def get_all_cosmic_runs():
        return CosmicQuery().get_all_runs()
    
    @staticmethod
    def get_cosmic_run_info():
        return CosmicQuery().df
    
    @staticmethod
    def get_all_ambe_runs():
        return AmBeQuery().get_all_runs()

    @staticmethod
    def get_ambe_run_info():
        return AmBeQuery().df

    @staticmethod
    def get_runscalers(run, source) -> pd.DataFrame:
        """
        Parameters
        ----------
        run : int
            Run number.
        source : str or int
            Source name (``'hira'`` or ``'nw'``) or number (0 or 1), respectively.

        Returns
        -------
        scalers : pd.DataFrame
            A dataframe containing datetime and scaler values.
        """
        q = MySqlQuery()
        source_id = q.runscalers_source_id(source)
        df = q.get_table(
            commands=f'''
            SELECT * FROM runscalers_{source_id}
            WHERE run = {run};
            '''
        )
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    @staticmethod
    def get_runscalernames(source) -> List[str]:
        """
        Parameters
        ----------
        source : str or int
            Source name (``'hira'`` or ``'nw'``) or number (0 or 1), respectively.
        
        Returns
        -------
        scaler_names : list of str
            A list of scaler names, i.e. all the columns except ``'run'`` and ``'datetime'``.
        """
        q = MySqlQuery()
        source_id = q.runscalers_source_id(source)
        columns = q.get_table(
            commands=f'''
            SELECT * FROM runscalers_{source_id}
            LIMIT 1;
            '''
        ).columns
        return [col for col in columns if col not in ['run', 'datetime']]
    
    @staticmethod
    def get_runscalernames_channel_map() -> pd.DataFrame:
        """
        Returns
        -------
        channel_map : pd.DataFrame
            A dataframe containing the mapping between scaler names and
            channels. Other columns like ``'description'`` and ``'wmu_name'``
            are also included; ''`wmu_name`'' is the name used by Western
            Michigan University (WMU).
        """
        q = MySqlQuery()
        return q.get_table(table_name='runscalernames').drop(columns=['index']).set_index(['id', 'chn'])


class ReactionParser:
    beams = [
        ('Ca', 40),
        ('Ca', 48),
    ]
    targets = [
        ('Ni', 58),
        ('Ni', 64),
        ('Sn', 112),
        ('Sn', 124),
    ]
    energies = [
        56,
        140,
    ]
    dst_style = None

    def __init__(self, beams=None, targets=None, energies=None):
        if beams is not None:
            self.beams = beams

        if targets is not None:
            self.targets = targets
    
        if energies is not None:
            self.energies = energies

    @property
    def beam_target_styles(self):
        return [
            'aa10bb20',
            'Aa10Bb20',
            '10aa20bb',
            '10Aa20Bb',
        ]
    
    @property
    def beam_target_energy_styles_re(self):
        return [
            'aa10bb20e100',
            'Aa10Bb20E100',
        ]
    
    @staticmethod
    def _isotope_combinations(symbol, mass_number):
        return [
            f'{symbol.capitalize()}{mass_number}',
            f'{symbol.lower()}{mass_number}',
            f'{mass_number}{symbol.capitalize()}',
            f'{mass_number}{symbol.lower()}',
        ]
    
    @classmethod
    def read_beam(cls, string):
        for beam in cls.beams:
            for beam_str in cls._isotope_combinations(*beam):
                if beam_str in string:
                    return beam
    
    @classmethod
    def read_target(cls, string):
        for target in cls.targets:
            for target_str in cls._isotope_combinations(*target):
                if target_str in string:
                    return target
    
    @classmethod
    def read_energy(cls, string):
        for energy in cls.energies:
            if str(energy) in string:
                return int(energy)
    
    @classmethod
    def set_convert_style(cls, dst_style):
        if dst_style not in cls.beam_target_styles and dst_style not in cls.beam_target_energy_styles:
            raise ValueError(f'Unknown style: {dst_style}')
        cls.dst_style = dst_style
    
    @classmethod
    def convert(cls, string, dst_style=None):
        if dst_style is None:
            dst_style = cls.dst_style

        beam = cls.read_beam(string)
        target = cls.read_target(string)
        energy = cls.read_energy(string)

        if dst_style == 'aa10bb20':
            return f'{beam[0].lower()}{beam[1]}{target[0].lower()}{target[1]}'
        if dst_style == 'Aa10Bb20':
            return f'{beam[0].capitalize()}{beam[1]}{target[0].capitalize()}{target[1]}'
        if dst_style == '10aa20bb':
            return f'{beam[1]}{beam[0].lower()}{target[1]}{target[0].lower()}'
        if dst_style == '10Aa20Bb':
            return f'{beam[1]}{beam[0].capitalize()}{target[1]}{target[0].capitalize()}'

        if dst_style == 'aa10bb20e100':
            return f'{beam[0].lower()}{beam[1]}{target[0].lower()}{target[1]}e{energy}'
        if dst_style == 'Aa10Bb20E100':
            return f'{beam[0].capitalize()}{beam[1]}{target[0].capitalize()}{target[1]}E{energy}'
    