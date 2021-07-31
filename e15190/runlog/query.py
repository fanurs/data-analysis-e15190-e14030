import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = 'simple_white'

from e15190 import PROJECT_DIR

class ElogQuery:
    def __init__(self, set_breakpoints=True):
        self.path = pathlib.Path(PROJECT_DIR, 'database/runlog/elog_final.h5')
        with pd.HDFStore(self.path, 'r') as file:
            self.df = file['runs_final']
            self.df.sort_values('run', inplace=True, ignore_index=True)

        self.unified_properties = ['target', 'beam', 'shadow_bar']
        self.max_run_gap = np.timedelta64(2, 'h')

        self.set_breakpoints()
    
    def set_breakpoints(self, unified_properties=None, max_run_gap=None):
        # breakpoints by reaction systems and shadow bars (in/out)
        if unified_properties is not None:
            self.unified_properties = unified_properties
        cols = self.unified_properties # alias
        breakpoints = (self.df[cols] != self.df[cols].shift(-1)).any('columns')

        # breakpoints by maximum inter-run time gap
        if max_run_gap is not None:
            self.max_run_gap = max_run_gap
        run_gaps = (self.df['begin_time'].shift(-1) - self.df['end_time'])
        breakpoints = np.any([breakpoints, run_gaps > self.max_run_gap], axis=0)
        breakpoints = pd.Series(breakpoints)

        # finalize multi indices: ibatch, irun
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

    @staticmethod
    def append_trigger_rate(fig, df):
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
        subdf = df.query('run < 3000')
        routine(fig, subdf, rc)

        rc = dict(row=2, col=1)
        subdf = df.query('run > 4000')
        routine(fig, subdf, rc)

        fig.update_yaxes(
            secondary_y=True,
            range=[1000, 4000],
            title='<b>Trigger rate</b> (Hz)',
        )
        return fig

    @staticmethod
    def append_nwb_pos_calib_params(fig, df):
        def get_calib_params(run):
            path = pathlib.Path(
                PROJECT_DIR,
                'database/neutron_wall/position_calibration/calib_params',
                f'run-{run:04d}-nwb.dat',
            )
            if path.is_file():
                df_par = pd.read_csv(path, delim_whitespace=True, comment='#')
                df_par.set_index('nwb-bar', drop=True, inplace=True)
                return df_par
            else:
                return None

        y_range = [1e9, -1e9]
        showlegend_first_only = {bar: True for bar in range(25)}
        original_len = len(fig.data)
        itrace = original_len
        itraces = dict()
        def routine(fig, df, rc):
            nonlocal itrace
            colors = {'p0': 'green', 'p1': 'purple'}
            ibatches = sorted(df.index.get_level_values('ibatch').unique())
            for ibatch in ibatches:
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
                            mode='lines',
                            line_color=colors[par],
                            showlegend=False,
                            name=par,
                            visible=False,
                        )
                        showlegend_first_only[bar] = False
                        fig.add_trace(scat, **rc, secondary_y=True)
                        itraces[(ibatch, bar, par)] = itrace
                        itrace +=1

        # apply routine
        rc = dict(row=1, col=1)
        subdf = df.query('run < 3000')
        routine(fig, subdf, rc)

        rc = dict(row=2, col=1)
        subdf = df.query('run > 4000')
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
                    dict(title=f'<i><b>NWB-{bar:02d} position calibration</b></i>'),
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
            currentvalue=dict(prefix='NWB-'),
            y=-0.03,
            ticklen=3,
            tickwidth=3,
        )]

        # finalizing layout
        y_width = y_range[1] - y_range[0]
        y_range = [y_range[0] - 0.05 * y_width, y_range[1] + 0.05 * y_width]
        fig.update_yaxes(
            secondary_y=True,
            range=y_range,
        )
        fig.update_layout(
            sliders=sliders,
            title=f'<i><b>NWB-{bars[init_active_index]:02d} position calibration</b></i>',
            title_x=0.5,
            title_xanchor='center',
        )
        return fig