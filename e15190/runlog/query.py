#%%
import collections
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = 'simple_white'

from e15190 import PROJECT_DIR

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
        self.path = 'database/runlog/elog_runs_filtered.h5' # relative to PROJECT_DIR

        self.df = None
        """The Elog database pandas dataframe."""
        with pd.HDFStore(PROJECT_DIR / self.path, 'r') as file:
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

        self.run_batches_path = 'database/runlog/run_batches.csv' # relative to $PROJECT_DIR

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
        filepath = pathlib.Path(filepath or PROJECT_DIR / self.run_batches_path)
        self.df_batches.to_csv(filepath)

    def load_run_batches(self, filepath=None):
        """Load the run batches from a CSV file.

        Parameters
        ----------
        filepath : str or pathlib.Path, default None
            The path to the CSV file. If ``None``, the function uses
            :py:attr:`run_batches_path`.
        """
        filepath = pathlib.Path(filepath or PROJECT_DIR / self.run_batches_path)
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
    
    def get_figure_reaction_overview(self, append_trigger_rate=True, dim=480):
        showlegend_first_only = {56: True, 140: True}
        def routine(fig, df, rc):
            target_map = {'Ni58': 58, 'Ni64': 64, 'Sn112': 112, 'Sn124': 124}
            energy_map = {56: 'darkblue', 140: 'red'}
            shadow_map = {'in': 'black', 'out': 'gold'}

            ibatches = sorted(df.index.get_level_values('ibatch').unique())
            for ibatch in ibatches:
                subdf = df.loc[ibatch]

                # prepare unified properties of batch
                first_entry = subdf.iloc[0]
                target = first_entry['target']
                energy = int(first_entry['beam'].split()[1]) # e.g. Ca48 140 MeV/u
                shadow_bar = first_entry['shadow_bar']

                # visualize target, beam energy and shadow bar info
                scat = go.Scatter(
                    x=subdf['run'], y=[target_map[target]] * len(subdf),
                    mode='lines+markers',
                    line_color=energy_map[energy],
                    legendgroup=f'{energy} MeV/u',
                    showlegend=showlegend_first_only[energy],
                    name=f'{energy} MeV/u',
                    hovertemplate=f'batch-{ibatch:02d} ({shadow_bar})',
                )
                showlegend_first_only[energy] = False
                fig.add_trace(scat, **rc, secondary_y=False)
                fig.add_vrect(
                    x0=subdf['run'].min(), x1=subdf['run'].max(),
                    line_width=1, line_color='green',
                    opacity=0.2, fillcolor=shadow_map[shadow_bar],
                    **rc, secondary_y=False,
                )

        # create plotly figure
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[dict(secondary_y=True)]] * 2,
            shared_yaxes=True,
            vertical_spacing=0.14,
            subplot_titles=[
                '<b>Beam: <sup>40</sup>Ca</b>',
                '<b>Beam: <sup>48</sup>Ca</b>',
            ],
            x_title='<b>Run number</b>',
            y_title='<b>Target mass number</b> A',
        )
        fig.layout.annotations[0].update(x=0.05)
        fig.layout.annotations[1].update(x=0.05)

        rc = dict(row=1, col=1)
        df = self.df.query('run < 3000')
        routine(fig, df, rc)

        rc = dict(row=2, col=1)
        df = self.df.query('run > 4000')
        routine(fig, df, rc)

        # finalize attributes
        if dim == 480:
            dim = dict(width=854, height=480)
        elif dim == 720:
            dim = dict(width=1280, height=720)
        fig.update_layout(
            title='<b><i>Overview of reaction systems in E15190-E14030</i></b>',
            title_x=0.5,
            title_xanchor='center',
            margin=dict(l=60, r=20, t=60, b=50),
            **dim,
            font_family='Cambria',
            hovermode='x',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
            ),
        )
        return self.append_trigger_rate(fig, self.df) if append_trigger_rate else fig

    def append_trigger_rate(self, fig):
        def routine(fig, df, rc):
            shadow_map = {'in': 'black', 'out': 'gold'}
            ibatches = sorted(df.index.get_level_values('ibatch').unique())
            for ibatch in ibatches:
                subdf = df.loc[ibatch]

                # prepare unified properties of batch
                first_entry = subdf.iloc[0]
                shadow_bar = first_entry['shadow_bar']

                # visualize target, beam energy and shadow bar info
                scat = go.Scatter(
                    x=subdf['run'], y=subdf['trigger_rate'],
                    mode='lines',
                    line_color=shadow_map[shadow_bar],
                    showlegend=False,
                    name='',
                    hovertemplate='%{y} Hz',
                )
                fig.add_trace(scat, **rc, secondary_y=True)
        
        rc = dict(row=1, col=1)
        subdf = self.df.query('run < 3000')
        routine(fig, subdf, rc)

        rc = dict(row=2, col=1)
        subdf = self.df.query('run > 4000')
        routine(fig, subdf, rc)

        fig.update_yaxes(
            secondary_y=True,
            range=[1000, 4000],
            title='<b>Trigger rate</b> (Hz)',
        )
        return fig

    def append_nw_pos_calib_params(self, AB, fig, ignore_batches=True):
        ab = AB.lower()
        def get_calib_params(run):
            path = pathlib.Path(
                PROJECT_DIR,
                'database/neutron_wall/position_calibration/calib_params',
                f'run-{run:04d}-nw{ab}.dat',
            )
            if path.is_file():
                df_par = pd.read_csv(path, delim_whitespace=True, comment='#')
                df_par.set_index(f'nw{ab}-bar', drop=True, inplace=True)
                return df_par
            else:
                return None

        colors = {'p0': 'green', 'p1': 'purple'}
        y_range = [1e9, -1e9]
        showlegend_first_only = {bar: True for bar in range(25)}
        original_len = len(fig.data)
        itrace = original_len
        itraces = dict()
        def routine(fig, df, rc):
            nonlocal itrace, colors
            ibatches = sorted(df.index.get_level_values('ibatch').unique())
            for ibatch in ibatches:
                if ignore_batches:
                    subdf = df.copy()
                else:
                    subdf = df.loc[ibatch]

                # collect parameters
                runs = []
                df_par = {'p0': None, 'p1': None}
                for run in subdf['run']:
                    _df = get_calib_params(run)
                    if _df is None:
                        continue
                    for par in df_par:
                        if df_par[par] is None:
                            df_par[par] = _df[par].to_frame()
                        else:
                            df_par[par] = pd.concat([df_par[par], _df[par]], axis='columns')
                    runs.append(run)

                # construct and collect plotly traces
                for par in df_par:
                    if df_par[par] is None:
                        break
                    df_par[par] = df_par[par].transpose()

                    # update y_range
                    y_range[0] = min(y_range[0], df_par[par].min().min())
                    y_range[1] = max(y_range[1], df_par[par].max().max())

                    # add all bars in this batch
                    for bar in df_par[par].columns:
                        scat = go.Scatter(
                            x=runs, y=df_par[par][bar],
                            mode='markers' if ignore_batches else 'lines',
                            marker_size=3,
                            line_color=colors[par],
                            showlegend=False,
                            name=par,
                            visible=False,
                        )
                        showlegend_first_only[bar] = False
                        fig.add_trace(scat, **rc, secondary_y=True)
                        itraces[(ibatch, bar, par)] = itrace
                        itrace +=1
                
                if ignore_batches:
                    break

        # apply routine
        rc = dict(row=1, col=1)
        subdf = self.df.query('run < 3000')
        routine(fig, subdf, rc)

        rc = dict(row=2, col=1)
        subdf = self.df.query('run > 4000')
        routine(fig, subdf, rc)

        # create slider step
        steps = []
        bars = sorted(set([_bar for (_, _bar, _) in itraces.keys()]))
        for bar in bars:
            visibilities = [_data.visible for _data in fig.data]
            for (_, _bar, _), itrace in itraces.items():
                visibilities[itrace] = (_bar == bar)
            step = dict(
                method='update',
                args=[
                    dict(visible=visibilities),
                    dict(title=f'<i><b>NW{AB}-{bar:02d} position calibration</b></i>'),
                ],
                label=f'{bar:02d}',
            )
            steps.append(step)

        # create slider
        init_active_index = 0
        for (_, _bar, _), itrace in itraces.items():
            fig.data[itrace].visible = (_bar == bars[init_active_index])
        sliders = [dict(
            active=init_active_index,
            steps=steps,
            currentvalue=dict(prefix=f'NW{AB}-'),
            y=-0.03,
            ticklen=3,
            tickwidth=3,
        )]

        # finalizing layout
        y_width = y_range[1] - y_range[0]
        y_range = [y_range[0] - 0.05 * y_width, y_range[1] + 0.05 * y_width]
        y2_title = f'<span style="color: {colors["p0"]};"><b>p<sub>0</sub></b></span>'
        y2_title += ', '
        y2_title += f'<span style="color: {colors["p1"]};"><b>p<sub>1</sub></b></span>'
        fig.update_yaxes(
            secondary_y=True,
            range=y_range,
            title=y2_title,
        )
        fig.update_layout(
            sliders=sliders,
            title=f'<i><b>NW{AB}-{bars[init_active_index]:02d} position calibration</b></i>',
        )
        return fig

class Query:
    """A class of query methods for the database.

    Currently, all queries are made from the Elog database. Of course, one can
    always directly make queries on the :py:attr:`elog.df` dataframe. This class
    is mostly for writing down functions that are used often, or to provide some
    basic query interface for users who are not familiar with pandas.

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
    
    """
    elog = ElogQuery(load_run_batches=True)
    """Class attribute :py:class:`ElogQuery` object loaded with run batches."""

    _map_run_iloc = {run: iloc for iloc, run in enumerate(elog.df['run'].to_list())}

    @staticmethod
    def _get_run_query(run):
        """Returns sub-dataframe of :py:attr:`ElogQuery.df` in
        :py:attr:`Query.elog` for a given run.

        Parameters
        ----------
        run : int
            Run number.
        """
        return Query.elog.df.iloc[Query._map_run_iloc[run]]
    
    @staticmethod
    def get_run_info(run):
        """Returns a dictionary of properties for a given run.
        
        Parameters
        ----------
        run : int
        """
        return dict(Query._get_run_query(run))
    
    @staticmethod
    def _get_batch_query(ibatch):
        """Returns a sub-dataframe of :py:attr:`ElogQuery.df` in
        :py:attr:`Query.elog` for a given batch number.

        Parameters
        ----------
        ibatch : int
            Batch number (0-indexed).
        """
        return Query.elog.df.loc[ibatch]

    @staticmethod
    def is_good(run):
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
    def are_good(runs):
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
    def get_batch(ibatch):
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
    def get_batch_info(ibatch, include_comments=True):
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
        return result
    
    @staticmethod
    def get_n_batches():
        """Returns the number of batches in the database."""
        return len(Query.elog.df_batches)
    
    @staticmethod
    def targets():
        """Returns all distinct target names in the database."""
        return Query.elog.df['target'].unique().tolist()
    
    @staticmethod
    def beams():
        """Returns all distinct beam names in the database."""
        return Query.elog.df['beam'].unique().tolist()
    
    @staticmethod
    def beam_energies():
        """Returns all distinct beam energies in the database."""
        return Query.elog.df['beam_energy'].unique().tolist()

    @staticmethod
    def select_runs(cut, comment_cut=None, **kw_comment):
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
    def select_batches(cut):
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
