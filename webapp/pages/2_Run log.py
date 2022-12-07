import itertools

import pandas as pd

from e15190.runlog.query import Query

import streamlit as st
st.set_page_config(page_title='Run log', layout='wide')
"""# Run log"""



##### Read in run log from the database
df_elog = Query().elog.df.copy()
df_elog = df_elog.reset_index()
# df_elog['begin_time'] = pd.to_datetime(df_elog['begin_time'])
# df_elog['end_time'] = pd.to_datetime(df_elog['end_time'])
df_elog['elapse'] = pd.to_timedelta(df_elog['elapse'])
df_elog['beam_energy'] = df_elog['beam_energy'].astype(int)
df_elog['trigger_rate'] = df_elog['trigger_rate'].astype(int)
df_elog['beam'] = df_elog['beam'] + ' at ' + df_elog['beam_energy'].astype(str) + ' MeV/u'
df_elog = df_elog.drop(columns=['irun', 'beam_energy', 'trigger_rate'])



"""Here are the data runs that have been selected for analysis. Use the left sidebar to specify filters on the runs."""
with st.sidebar:
    beam = st.multiselect('Beam', sorted(df_elog['beam'].unique()))
    target = st.multiselect('Target', sorted(df_elog['target'].unique()))
    shadow_bar = st.multiselect('Shadow bar', ['in', 'out'])
    run_range = st.slider('Run range', 2000, 5000, (2000, 5000))
    elapse_range = st.slider('Elapse range', 0, 90, (0, 90), help='Elapse time in minutes')
    date_range = st.date_input(
        'Date range',
        (pd.to_datetime('2018-02-01'), pd.to_datetime('2018-03-31')),
        min_value=pd.to_datetime('2018-02-01'), max_value=pd.to_datetime('2018-03-31'),
        help='Start date depends on "begin_time"; end date depends on "end_time"',
    )



##### Filter the data runs
conditions = [
    None if not beam else f'beam in {beam}',
    None if not target else f'target in {target}',
    None if not shadow_bar else f'shadow_bar in {shadow_bar}',
    f'run >= {run_range[0]} & run <= {run_range[1]}',
]
conditions_str = ' & '.join([c for c in conditions if c is not None])
df_elog_display = df_elog.query(conditions_str)
df_elog_display = df_elog_display[
    (df_elog_display['elapse'] >= pd.to_timedelta(f'00:{elapse_range[0]}:00')) &
    (df_elog_display['elapse'] <= pd.to_timedelta(f'00:{elapse_range[1]}:00')) &
    (df_elog_display['begin_time'] >= pd.to_datetime(date_range[0])) &
    (df_elog_display['end_time'] <= pd.to_datetime(date_range[1]))
]
df_elog_display['elapse'] = (pd.to_datetime('2000-01-01') + df_elog_display['elapse']).dt.strftime(r'%H:%M:%S')
df_elog_display['begin_time'] = df_elog_display['begin_time'].dt.strftime(r'%m/%d %H:%M:%S')
df_elog_display['end_time'] = df_elog_display['end_time'].dt.strftime(r'%m/%d %H:%M:%S')
st.dataframe(df_elog_display, use_container_width=True)
st.caption(f'Selected {len(df_elog_display)} runs.')



##### Summarize the runs into chunks"""
def group_into_chunks(numbers):
    chunks = []
    for _, group in itertools.groupby(enumerate(sorted(numbers)), lambda x: x[1] - x[0]):
        chunk = [ele[1] for ele in group]
        chunks.append((chunk[0], chunk[-1]))
    return chunks
run_chunks = group_into_chunks(df_elog_display['run'])

df_chunks = []
for run_chunk in run_chunks:
    sub_elog = df_elog.query(f'run >= {run_chunk[0]} & run <= {run_chunk[1]}')
    targets, beams, shadow_bars, comments = map(
        lambda x: sorted(sub_elog[x].unique()),
        ['target', 'beam', 'shadow_bar', 'comment'],
    )
    df_chunks.append([
        f'{run_chunk[0]} - {run_chunk[1]}' if run_chunk[0] != run_chunk[1] else f'{run_chunk[0]}',
        f'{run_chunk[1] - run_chunk[0] + 1}',
        sub_elog['elapse'].sum().total_seconds(),
        ', '.join(targets),
        ', '.join(beams),
        ', '.join(shadow_bars),
        '"' + '", "'.join(comments) + '"',
    ])
df_chunks = pd.DataFrame(df_chunks, columns=['run(s)', 'no. of runs', 'total elapse', 'target(s)', 'beam(s)', 'shadow bar(s)', 'comment(s)'])
df_chunks['total elapse'] = (pd.to_datetime('2000-01-01') + pd.to_timedelta(df_chunks['total elapse'], unit='s')).dt.strftime(r'%H:%M:%S')

"""Here are the selected runs grouped into chunks:"""
st.table(df_chunks)
