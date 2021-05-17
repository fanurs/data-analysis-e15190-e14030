import itertools as itr
import pathlib

import numpy as np
import pandas as pd

from .. import PROJECT_DIR
from . import MYSQL_DOWNLOAD_PATH, ELOG_DOWNLOAD_PATH
from ..utilities import tables

"""After data cleansing, all tables are being saved into TXT files and HDF files.
TXT files allow quick inspection without HDF viewer.
HDF files are suitable for data analysis as they preserve data types like datetimes, floats, etc.
"""

MYSQL_CLEANSED_TXT_DIR = pathlib.Path(PROJECT_DIR, 'database', 'runlog', 'mysql_cleansed_txt')
MYSQL_CLEANSED_TXT_DIR.mkdir(parents=True, exist_ok=True)
MYSQL_CLEANSED_PATH = pathlib.Path(PROJECT_DIR, 'database', 'runlog', 'mysql_cleansed.h5')

ELOG_CLEANSED_CSV_DIR = pathlib.Path(PROJECT_DIR, 'database', 'runlog', 'elog_cleansed_csv')
ELOG_CLEANSED_CSV_DIR.mkdir(parents=True, exist_ok=True)
ELOG_CLEANSED_PATH = pathlib.Path(PROJECT_DIR, 'database', 'runlog', 'elog_cleansed.h5')

class ElogCleanser:
    """This is a class of methods to perform data cleansing on the table downloaded from ELOG webpage at WMU.
    """
    def __init__(self):
        pass

    def cleanse(self):
        with open(ELOG_DOWNLOAD_PATH, 'r') as file:
            self.elog = pd.read_html(file)[4]

        # re-labeling
        self.elog.columns = self.elog.iloc[0] # use first row as column
        self.elog.drop(0, inplace=True)
        self.elog = self.elog.astype(str) # turn all entries into string; convert numbers and datetimes later
        self.elog.reset_index(drop=True, inplace=True)

        # split into runs and events
        is_run_mask = self.elog['RUN'].str.isdigit()
        self.events = self.elog[~is_run_mask]
        self.elog = self.elog[is_run_mask]

        # data cleansing
        self._cleanse_elog()
        self._cleanse_events()

        # save to files
        with pd.HDFStore(ELOG_CLEANSED_PATH, 'w') as file:
            file.append('elog', self.elog)
            file.append('events', self.events)

        path = pathlib.Path(ELOG_CLEANSED_CSV_DIR, 'elog.csv')
        self.elog.to_csv(path, index=False)

        path = pathlib.Path(ELOG_CLEANSED_CSV_DIR, 'events.csv')
        self.events.to_csv(path, index=False)

    def _cleanse_elog(self):
        # drop entries with type value 'junk'
        self.elog = self.elog[~self.elog['Type'].str.contains('junk', case=False)]

        # convert run into integers
        self.elog = self.elog.astype({'RUN': int})

        # convert into datetime and timedelta objects
        self.elog['Begin time'] = pd.to_datetime(self.elog['Begin time'])
        self.elog['End time'] = pd.to_datetime(self.elog['End time'])
        self.elog['Elapse'] = pd.to_timedelta(self.elog['Elapse'])

        # standardize targets
        targets = [
            ('Ni', '58'),
            ('Ni', '64'),
            ('Sn', '112'),
            ('Sn', '124'),
            ('CH', '2'),
        ]
        for symb, mass in targets:
            self.elog.replace(
                {'Target': f'^.*(?i)({symb}{mass}|{mass}{symb}).*$'}, f'{symb}{mass}',
                regex=True, inplace=True,
            )

        # take care of 'nan', i.e. convert all 'nan' in string-type columns into empty strings
        self.elog.replace(
            {col: '(?i)nan' for col in ['Type', 'Target', 'Beam', 'Shadow bars']}, '',
            regex=True, inplace=True,
        )

        # convert trigger rates into float
        self.elog.replace({'Trigger rate': '(?i)nan'}, 0.0, regex=True, inplace=True)
        self.elog = self.elog.astype({'Trigger rate': float})

        # remove trailing 'Scalers: ...' in the comments
        self.elog.replace({'Comments': 'Scalers.*$'}, '', regex=True, inplace=True)
        self.elog['Comments'] = self.elog['Comments'].str.strip()
        self.elog.rename(columns={'Comments': 'Comment'})

        # use only lowercase for column names and replace spaces with underscores
        self.elog.columns = [col.lower().replace(' ', '_') for col in self.elog.columns]

    def _cleanse_events(self):
        self.events['Entry'] = self.events['End time']
        self.events.replace({'Comments': '(?i)nan'}, 'none', regex=True, inplace=True)
        self.events = self.events[['RUN', 'Begin time', 'Entry', 'Comments']]
        self.events.columns = ['run', 'time', 'entry', 'comment']

class MySqlCleanser:
    """This is a class of methods to perform data cleansing on the tables freshly downloaded from MySQL at WMU.
    """
    def __init__(self):
        self.file = pd.HDFStore(MYSQL_DOWNLOAD_PATH, 'r')
        self.table_names = [ # tables from WMU that contain relevant information for our analysis
            'runbeam',
            'runbeamintensity',
            'runinfo',
            'runlog',
            'runscalernames',
            'runscalers',
            'runtarget',
        ]

    def cleanse(self):
        df = {
            'runbeam': self._cleanse_runbeam(),
            'runtarget': self._cleanse_runtarget(),
            'runlog': self._cleanse_runlog(),
            'runscalernames': self._cleanse_runscalernames(),
        }

        with pd.HDFStore(MYSQL_CLEANSED_PATH, 'w') as file:
            for name, _df in df.items():
                file.append(name, _df)
        
        for name, _df in df.items():
            path = pathlib.Path(MYSQL_CLEANSED_TXT_DIR, f'{name}.txt')
            tables.to_fwf(_df, path)

    def _cleanse_runbeam(self):
        print('Data cleansing runbeam... ', end='', flush=True)
        df = self.file['runbeam'].copy()
        df.rename(columns={
            'name': 'beam',
            'bid': 'id',
        }, inplace=True)
        df = df[['id', 'beam']]

        print('Done!')
        return df

    def _cleanse_runtarget(self):
        print('Data cleansing runtarget... ', end='', flush=True)

        # HARD-CODED from hiramodules/runtarget
        df = pd.DataFrame(
            columns=['id', 'target', 'thickness'],
            data=[
                [1,   'none',   0.0],
                [2, 'viewer',   0.0],
                [3,  'blank',   0.0],
                [4,   'Ni58',   5.0],
                [5,   'Ni64',   5.3],
                [6,  'Sn124',  6.47],
                [7,  'Sn112',  6.09],
                [8,    'CH2',  10.0],
            ],
        )

        print('Done!')
        return df

    def _cleanse_runlog(self):
        """Perform data cleansing on runlog.

        The `runlog` table from the WMU MySQL database contains many entries.
        A valid run in the `runlog` is composed of a pair of entries that satisfy all the following conditions:
        - Run number either between 2000 - 3000 or 4000 - 5000.
        - Exactly one entry has a state of `begin`, and another entry has a state of `end`.
        - The `end` state must happen after the `begin` state (compare `date`).
        - Both titles must be identical.

        Any invalid runs will be discarded after data cleansing, except for entries that have `event` as their host.
        All `event` entries will be kept.

        **Remark: Runs that are invalid are not all trash. But for our purpose of data analysis, we do not care about them.
        On the other hand, not all valid runs contain useful physics. There is no check on whether the detectors are working
        properly, the runs are electronics are set correctly, etc. All these are not being taken care in this stage of data
        cleansing. We leave the responsibility to analyses after this.
        """
        print('Data cleansing runlog... ', end='', flush=True)

        df = self.file['runlog'].copy()
        df = df.query('(runno >= 2000 & runno < 3000) | (runno >= 4000 & runno < 5000)')

        mask_begin = (df['state'] == 'begin')
        mask_end = (df['state'] == 'end')

        df_cleansed = None
        # for run in itr.chain(range(2000, 3000), range(4000, 5000)):
        for run in range(2000, 5000):
            # continue if no run is found
            mask_run = (df['runno'] == run)
            if np.sum(mask_run) == 0:
                continue

            # continue if not finding exactly one begin state
            df_begin = df[mask_run & mask_begin]
            df_end = df[mask_run & mask_end]
            if len(df_begin) != 1 or len(df_end) != 1:
                continue

            # continue if end-state entry does not happen after begin-state entry
            if not (df_begin.iloc[0]['date'] < df_end.iloc[0]['date']):
                continue

            # continue if the titles are different
            if df_begin.iloc[0]['title'] != df_end.iloc[0]['title']:
                continue

            # concatenate result
            df_cleansed = pd.concat([df_cleansed, df_begin, df_end], ignore_index=True)

        df_cleansed.rename(columns={'runno': 'run', 'date': 'datetime'}, inplace=True)
        df_cleansed.drop(columns=['host', 'lid'], inplace=True)
        df_cleansed.set_index(['run', 'state'], inplace=True, verify_integrity=True)

        print('Done!')
        return df_cleansed

    def _cleanse_runscalernames(self):
        print('Data cleansing runscalernames... ', end='', flush=True)

        # HARD-CODED FROM hiramodules/runscalernames
        df = pd.DataFrame(
            columns=['id', 'chn', 'name', 'description', 'wmu_name'],
            data=[
                # HiRA, MB & FA scalers
                [0,  0, 'MB1-BACK-OR',      'microball 1 back OR',                 'MB1_Back_OR'],
                [0,  1, 'MB2-BACK-OR',      'microball 2 back OR',                 'MB2_Back_OR'],
                [0,  2, 'MB1-FRONT-OR',     'microball 1 front OR',                'MB1_Front_OR'],
                [0,  3, 'MB2-FRONT-OR',     'microball 2 front OR',                'MB2_Front_OR'],
                [0,  4, 'SI-BACK-OR',       'silicon back OR',                     'Si_Back_OR'],
                [0,  5, 'SI-FRONT-OR',      'silicon front OR',                    'Si_Front_OR'],
                [0,  6, 'CSI1-OR',          'cesium iodide 1 OR',                  'CsI_1_OR'],
                [0,  7, 'CSI2-OR',          'cesium iodide 2 OR',                  'CsI_2_OR'],
                [0,  8, 'CSI3-OR',          'cesium iodide 3 OR',                  'CsI_3_OR'],
                [0,  9, 'CSI-OR-OR',        'cesium iodide OR of ORs',             'CsI_OR_of_ORs'],
                [0, 10, 'HIRA-RAW',         'hira raw trigger',                    'Raw_HiRA'],
                [0, 11, 'GLOB-MASTER',      'global master',                       'Global_Master'],
                [0, 12, 'BUSY',             'busy',                                'Busy'],
                [0, 13, 'HIRA-DSS',         'hira AND downstream scintillator',    'DSS_HiRA'],
                [0, 14, 'NW-RAW',           'neutron wall raw trigger',            'NW_Trigger_Raw'],
                [0, 15, 'MB-OR',            'microball OR',                        'microBall_OR'],
                [0, 16, 'FA01',             'forward array 1',                     'FA01'],
                [0, 17, 'FA02',             'forward array 2',                     'FA02'],
                [0, 18, 'FA03',             'forward array 3',                     'FA03'],
                [0, 19, 'FA04',             'forward array 4',                     'FA04'],
                [0, 20, 'FA05',             'forward array 5',                     'FA05'],
                [0, 21, 'FA06',             'forward array 6',                     'FA06'],
                [0, 22, 'FA07',             'forward array 7',                     'FA07'],
                [0, 23, 'FA08',             'forward array 8',                     'FA08'],
                [0, 24, 'FA09',             'forward array 9',                     'FA09'],
                [0, 25, 'FA10',             'forward array 10',                    'FA10'],
                [0, 26, 'FA11',             'forward array 11',                    'FA11'],
                [0, 27, 'FA12',             'forward array 12',                    'FA12'],
                [0, 28, 'FA13',             'forward array 13',                    'FA13'],
                [0, 29, 'FA14',             'forward array 14',                    'FA14'],
                [0, 30, 'FA15',             'forward array 15',                    'FA15'],
                [0, 31, 'FA16',             'forward array 16',                    'FA16'],
                [0, 32, 'SI-LIVE-BACK-OR',  'silicon live back OR',                'Live_Si_Back_OR'],
                [0, 33, 'SI-LIVE-FRONT-OR', 'silicon live front OR',               'Live_Si_Front_OR'],
                [0, 34, 'CSI1-LIVE-OR',     'cesium iodide 1 live OR',             'Live_CsI_1_OR'],
                [0, 35, 'CSI2-LIVE-OR',     'cesium iodide 2 live OR',             'Live_CsI_2_OR'],
                [0, 36, 'CSI3-LIVE-OR',     'cesium iodide 3 live OR',             'Live_CsI_3_OR'],
                [0, 37, 'MB-LIVE-OR',       'microball live OR',                   'Live_microBall_OR'],
                [0, 38, 'MB-LIVE-MULTI',    'microball live multiplicity trigger', 'Live_microBall_Mult_Trigger'],
                [0, 39, 'CSI-LIVE-OR-OR',   'cesium iodide OR of LIVE ORs',        'Live_CsI_OR_of_ORs'],
                [0, 44, 'MB',               'microball trigger',                   'microBallTrigger'],
                [0, 45, 'MB-DS',            'microball downscaled trigger',        'microBallTrigger_DS'],
                [0, 46, 'DSS-LIVE',         'downstream scintillator live',        'Live_DSS'],
                [0, 47, 'HIRA-DS',          'hira downscaled trigger',             'HiRA_DS'],
                [0, 48, 'DSS2-LIVE',        'downstream scintillator 2 live',      'Live_DSS2'],
                
                # NW & VW scalers
                [1,  0, 'VW-TOP-OR',        'veto wall top OR',                    'OR_T_VW'],
                [1,  1, 'VW-BOT-OR',        'veto wall bottom OR',                 'OR_B_VW'],
                [1,  2, 'VW-TOP-BOT-OR',    'veto wall top-bottom OR',             'OR_T_OR_B_VW'],
                [1,  3, 'VW-GATE',          'veto wall gate',                      'GATE_VW'],
                [1,  4, 'VW-FCLR',          'veto wall fast clear',                'FCLR_VW'],
                [1,  6, 'MASTER',           'master trigger',                      'MASTER_TRG'],
                [1,  7, 'MASTER2',          'master trigger 2',                    'MASTER_TRG'],
                [1,  8, 'VW-DELAY',         'veto delayed trigger',                'WV_TRIG_DELAYED'],
                [1,  9, 'FA-OR',            'forward array OR',                    'FART_OR'],
                [1, 10, 'DSS-OR',           'downstream scintillator OR',          'DSS_OR'],
                [1, 16, 'NW-RAW',           'neutron wall raw trigger',            'NW_Raw_Trig'],
                [1, 17, 'NW-LIVE',          'neutron wall live trigger',           'NW_Live_Trig'],
                [1, 18, 'NW-FCLR',          'neutron wall fast clear',             'NW_Fast_Clear'],
                [1, 19, 'DSS',              'downstream scintillator',             'DSS'],
                [1, 30, 'FA17',             'forward array 17',                    'FA17'],
                [1, 31, 'FA18',             'forward array 18',                    'FA18'],
            ],
        )

        print('Done!')
        return df

