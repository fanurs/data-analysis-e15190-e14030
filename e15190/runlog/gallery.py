import functools
from pathlib import Path
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from e15190.runlog.query import ElogQuery

def get_run_calib_params(filename, directory, index_key, **kwargs):
    path = Path(directory) / filename
    if not path.is_file():
        return
    kw = dict(delim_whitespace=True, comment='#')
    kw.update(kwargs)
    df_par = pd.read_csv(path, **kw)
    df_par.set_index(index_key, drop=True, inplace=True)
    return df_par

class ElogQueryGallery(ElogQuery):
    DATABASE_DIR = '$DATABASE_DIR/runlog/interactive_plots/'

    def __init__(self):
        super().__init__()
        pio.templates.default = 'simple_white'

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
                # energy = int(first_entry['beam'].split()[1]) # e.g. Ca48 140 MeV/u
                energy = first_entry['beam_energy']
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
        return self.append_trigger_rate(fig) if append_trigger_rate else fig

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

    def append_nw_pos_calib_params(self, AB, fig, bars=None, ignore_batches=False):
        ab = AB.lower()
        get_calib_params = functools.partial(
            get_run_calib_params,
            directory=os.path.expandvars('$DATABASE_DIR/neutron_wall/position_calibration/calib_params'),
            index_key=f'nw{ab}-bar',
        )

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
                    _df = get_calib_params(f'run-{run:04d}-nw{ab}.dat')
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
        if bars is None:
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

    def append_nw_attenuation_lengths(self, AB, fig, bars=None, ignore_batches=False):
        ab = AB.lower()
        get_calib_params = functools.partial(
            get_run_calib_params,
            directory=os.path.expandvars('$DATABASE_DIR/neutron_wall/light_output_calibration/calib_params'),
            index_key='bar',
        )
        
        colors = {'att_length': 'green'}
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
                df_par = {'att_length': None}
                for run in subdf['run']:
                    _df = get_calib_params(f'run-{run:04d}.dat')
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
        if bars is None:
            bars = sorted(set([_bar for (_, _bar, _) in itraces.keys()]))
        for bar in bars:
            visibilities = [_data.visible for _data in fig.data]
            for (_, _bar, _), itrace in itraces.items():
                visibilities[itrace] = (_bar == bar)
            step = dict(
                method='update',
                args=[
                    dict(visible=visibilities),
                    dict(title=f'<i><b>NW{AB}-{bar:02d} attenuation length</b></i>'),
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
        y_range = [50, 200]
        y_width = y_range[1] - y_range[0]
        y_range = [y_range[0] - 0.05 * y_width, y_range[1] + 0.05 * y_width]
        y2_title = f'<span style="color: {colors["att_length"]};"><b>λ [cm]</b></span>'
        fig.update_yaxes(
            secondary_y=True,
            range=y_range,
            title=y2_title,
        )
        fig.update_layout(
            sliders=sliders,
            title=f'<i><b>NW{AB}-{bars[init_active_index]:02d} attenuation length</b></i>',
        )
        return fig

    def append_nw_gain_ratios(self, AB, fig, bars=None, ignore_batches=False):
        ab = AB.lower()
        get_calib_params = functools.partial(
            get_run_calib_params,
            directory=os.path.expandvars('$DATABASE_DIR/neutron_wall/light_output_calibration/calib_params'),
            index_key='bar',
        )
        
        colors = {'gain_ratio': 'green'}
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
                df_par = {'gain_ratio': None}
                for run in subdf['run']:
                    _df = get_calib_params(f'run-{run:04d}.dat')
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
        if bars is None:
            bars = sorted(set([_bar for (_, _bar, _) in itraces.keys()]))
        for bar in bars:
            visibilities = [_data.visible for _data in fig.data]
            for (_, _bar, _), itrace in itraces.items():
                visibilities[itrace] = (_bar == bar)
            step = dict(
                method='update',
                args=[
                    dict(visible=visibilities),
                    dict(title=f'<i><b>NW{AB}-{bar:02d} gain ratio (Right/Left)</b></i>'),
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
        y_range = [0.5, 2]
        y_width = y_range[1] - y_range[0]
        y_range = [y_range[0] - 0.05 * y_width, y_range[1] + 0.05 * y_width]
        y2_title = f'<span style="color: {colors["gain_ratio"]};"><b>g<sub>R</sub>/g<sub>L</sub></b></span>'
        fig.update_yaxes(
            secondary_y=True,
            range=y_range,
            title=y2_title,
        )
        fig.update_layout(
            sliders=sliders,
            title=f'<i><b>NW{AB}-{bars[init_active_index]:02d} gain ratio</b></i>',
        )
        return fig

    def save_html_to_fishtank(self, key, chmod=0o775):
        """Save figure as HTML file to be hosted via Fishtank server.

        Parameters
        ----------
        key : str
            "overview" or "pos-calib".
        chmod : octal, default 0o775
            The unix permission needed to make it accessible from the web.
            Fishtank specifically requires it to be 0o775, which is the default
            value.
        """
        fig = self.get_figure_reaction_overview(append_trigger_rate=False, dim=720)
        if key == 'overview':
            fig = self.append_trigger_rate(fig)
        if key == 'pos-calib':
            fig = self.append_nw_pos_calib_params('B', fig)
        if key == 'att-length':
            fig = self.append_nw_attenuation_lengths('B', fig)
        if key == 'gain-ratio':
            fig = self.append_nw_gain_ratios('B', fig)
        path = Path(os.path.expandvars(self.DATABASE_DIR)) / f'plotly_{key}.html'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        os.chmod(path, chmod)
        return fig
