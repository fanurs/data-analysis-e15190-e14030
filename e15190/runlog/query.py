import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = 'simple_white'

import e15190

class ElogQuery:
    def __init__(self):
        self.path = pathlib.Path(e15190.PROJECT_DIR, 'database/runlog/elog_final.h5')
        with pd.HDFStore(self.path, 'r') as file:
            self.df = file['runs_final']
            self.df.sort_values('run', inplace=True, ignore_index=True)

        self.unified_properties = ['target', 'beam', 'shadow_bar']
        self.max_run_gap = np.timedelta64(2, 'h')
    
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
    
    def draw_reaction_overview(self, append_trigger_rate=True, dim=480):
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
