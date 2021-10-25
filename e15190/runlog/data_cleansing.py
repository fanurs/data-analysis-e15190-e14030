import pathlib

import numpy as np
import pandas as pd

from e15190 import PROJECT_DIR
from e15190.runlog.downloader import MYSQL_DOWNLOAD_PATH, ELOG_DOWNLOAD_PATH
from e15190.utilities import tables

CLEANSED_DIR = PROJECT_DIR / 'database/runlog/cleansed'
CLEANSED_DIR.mkdir(parents=True, exist_ok=True)

class ElogCleanser:
    def __init__(self, elog_path=None):
        """To cleanse the ELOG downloaded from the web.

        The ELOG is hosted at
        http://neutronstar.physics.wmich.edu/runlog/index.php?op=list
        It is basically a large table that contains the following columns:
            - RUN
            - Begin time
            - End time
            - Elapse
            - DAQ
            - Type
            - Target
            - Beam
            - Shadow bars
            - Trigger rate
            - Comments
        Each row is either an experimental run or an event.
        
        The purpose of this class is to ensure all columns have been converted
        into some suitable data types. Some simple removal of entries (rows) is
        also done here.

        Parameters
        ----------
        elog_path : str, default None
            The path to the local ELOG html file. If ``None``, the default path
            is used, i.e. ``$PROJECT_DIR/database/runlog/downloads/elog.html``.
        """
        self.elog_path = ELOG_DOWNLOAD_PATH if elog_path is None else pathlib.Path(elog_path)

        self.runs = None
        """Dataframe of experimental runs"""

        self.events = None
        """Dataframe of events"""

        self.runs_final = None
        """Dataframe of experimental runs after filtering"""

        self.events_final = None
        """To do: Dataframe of events after filtering"""

    def cleanse(self):
        """Cleanse the elog.

        This function updates the ``self.runs`` and ``self.events`` dataframes.
        Most entries are being preserved, i.e. minimal filtering is done except
        for those that are labeled as "junk" or simply corrupted.
        """
        with open(self.elog_path, 'r') as file:
            self.runs = pd.read_html(file)[4]

        # re-labeling
        self.runs.columns = self.runs.iloc[0] # use first row as column
        self.runs.drop(0, inplace=True)
        self.runs = self.runs.astype(str) # turn all entries into string; convert numbers and datetimes later
        self.runs.reset_index(drop=True, inplace=True)

        # split into runs and events
        is_run_mask = self.runs['RUN'].str.isdigit()
        self.events = self.runs[~is_run_mask]
        self.runs = self.runs[is_run_mask]

        # data cleansing
        self._cleanse_runs()
        self._cleanse_events()
    
    def save_cleansed_elog(
        self,
        output_dir=None,
        df_runs=None,
        df_events=None,
        elog_runs_csv_name='elog_runs',
        elog_events_csv_name='elog_events',
        verbose=True,
    ):
        """Save experimental runs and events after data cleansing.

        Both ``self.runs`` and ``self.events`` are being saved into separate CSV
        files.

        Parameters
        ----------
        output_dir : str or pathlib.Path, default None
            The directory to save the files. If ``None``, the default path is
            used, i.e. ``$PROJECT_DIR/database/runlog/cleansed``.
        df_runs : pd.DataFrame, default None
            Experimental runs to save. If None, the default is to use
            ``self.runs``.
        df_events : pd.DataFrame, default None
            Events to save. If None, the default is to use ``self.events``.
        elog_runs_csv_name : str, default 'elog_runs'
            The name of the CSV file for experimental runs without the extension;
            the extension ``.csv`` will be appended automatically.
        elog_events_csv_name : str, default 'elog_events'
            The name of the CSV file for events without the extension; the
            extension ``.csv`` will be appended automatically.
        verbose : bool, default True
            Whether to print the progress.
        """
        output_dir = CLEANSED_DIR if output_dir is None else pathlib.Path(output_dir)
        df_runs = self.runs if df_runs is None else df_runs
        df_events = self.events if df_events is None else df_events

        path = output_dir / (elog_runs_csv_name + '.csv')
        self._save_as_csv(df_runs, path)
        if verbose:
            print(f'Cleansed runs dataframe has been saved to "{path}"')

        path = output_dir / (elog_events_csv_name + '.csv')
        self._save_as_csv(df_events, path)
        if verbose:
            print(f'Cleansed events has been saved to "{path}"')

    def _cleanse_runs(self):
        """Cleanse the entries (rows) of experimental runs.

        This function updates the ``self.runs`` dataframe and go through the
        following steps:
            - Remove entries whose "Type" are labeled as "junk".
            - Convert all runs into integers.
            - Convert all times into datetimes, this includes "Begin time" and "End time".
            - Convert elapse time into timedelta, this includes "Elapse".
            - Standardize the "Target" into one of "Ni58", "Ni64", "Sn112", "Sn124" or "CH2".
            - Turn all "nan" in "Type", "Target", "Beam" and "Shadow bars" into empty strings.
            - Convert "Trigger rate" into float.
            - Convert "Comments" into "none" if it is "nan".
            - Use singular form for all column names, e.g. "Shadow bars" -> "Shadow bar".
            - Convert all column names to lowercase, and replace spaces with underscores.
        """
        # drop entries with type value 'junk'
        self.runs = self.runs[~self.runs['Type'].str.contains('junk', case=False)]

        # convert run into integers
        self.runs = self.runs.astype({'RUN': int})

        # convert into datetime and timedelta objects
        self.runs['Begin time'] = pd.to_datetime(self.runs['Begin time'])
        self.runs['End time'] = pd.to_datetime(self.runs['End time'])
        self.runs['Elapse'] = pd.to_timedelta(self.runs['Elapse'])

        # standardize targets
        targets = [
            ('Ni', '58'),
            ('Ni', '64'),
            ('Sn', '112'),
            ('Sn', '124'),
            ('CH', '2'),
        ]
        for symb, mass in targets:
            self.runs.replace(
                {'Target': f'^.*(?i)({symb}{mass}|{mass}{symb}).*$'}, f'{symb}{mass}',
                regex=True, inplace=True,
            )

        # take care of 'nan', i.e. convert all 'nan' in string-type columns into empty strings
        self.runs.replace(
            {col: '(?i)nan' for col in ['Type', 'Target', 'Beam', 'Shadow bars']}, '',
            regex=True, inplace=True,
        )
        self.runs.rename(columns={'Shadow bars': 'Shadow bar'}, inplace=True)

        # convert trigger rates into float
        self.runs.replace({'Trigger rate': '(?i)nan'}, 0.0, regex=True, inplace=True)
        self.runs = self.runs.astype({'Trigger rate': float})

        # remove trailing 'Scalers: ...' in the comments
        self.runs.replace({'Comments': 'Scalers.*$'}, '', regex=True, inplace=True)
        self.runs['Comments'] = self.runs['Comments'].str.strip()
        self.runs.rename(columns={'Comments': 'Comment'}, inplace=True)

        # use only lowercase for column names and replace spaces with underscores
        self.runs.columns = [col.lower().replace(' ', '_') for col in self.runs.columns]

    def _cleanse_events(self):
        """Cleanse the entries (rows) of events.

        This function updates the ``self.events`` dataframe which has columns
        "run", "time", "entry" and "comment".
        """
        # "End time" is actually entry
        self.events['Entry'] = self.events['End time']

        # replace "nan" with "none"
        self.events.replace({'Comments': '(?i)nan'}, 'none', regex=True, inplace=True)

        # select columns that are relevant for events
        self.events = self.events[['RUN', 'Begin time', 'Entry', 'Comments']]

        # use only lowercase for column names and replace spaces with underscores
        self.events.columns = ['run', 'time', 'entry', 'comment']

    def filtered_runs(self):
        """Updates and returns filtered experimental runs.

        This function filters the experimental runs and keeps only the
        meaningful entries. The filters are:
            - run in [2000, 3000) or [4000, 5000)
            - elapse time > 5 minutes
            - daq == "Merged"
            - type == "data"
            - target is one of "Ni58", "Ni64", "Sn112" or "Sn124"
            - beam is one of "Ca40 56 MeV/u", "Ca40 140 MeV/u", "Ca48 56 MeV/u" or "Ca48 140 MeV/u"
            - trigger rate > 1000.0 / sec
        
        Returns
        -------
        runs_final : pd.DataFrame
            Filtered experimental runs.
        """
        # get a copy of elog to filter for valid runs
        path = pathlib.Path(CLEANSED_DIR, 'elog_runlog.h5')
        if self.runs is None:
            with pd.HDFStore(path, 'r') as file:
                self.runs_final = file['runs'].copy()
        else:
            self.runs_final = self.runs.copy()

        # select valid runs only
        conditions = [
            '(run >= 2000 & run < 3000) | (run >= 4000 & run < 5000)',
            'elapse > @pd.Timedelta(minutes=5.0)',
            'daq == "Merged"',
            'type == "data"',
            'target in ["Ni58", "Ni64", "Sn112", "Sn124"]',
            'beam in ["Ca40 56 MeV/u", "Ca40 140 MeV/u", "Ca48 56 MeV/u", "Ca48 140 MeV/u"]',
            'trigger_rate > 1000.0',
        ]
        self.runs_final = self.runs_final.query('(' + ') & ('.join(conditions) + ')', engine='python')

        # drop redundant columns
        self.runs_final.drop(columns=['daq', 'type'], inplace=True)

        return self.runs_final

    def save_filtered_runs(
        self,
        file_extension='csv',
        output_path=None,
        df=None,
        verbose=True,
    ):
        """Save filtered experimental runs to a file.

        Parameters
        ----------
        file_extension : 'csv' or 'h5', default 'csv'
            File extension of the output file. This will be overwritten if
            ``output_path`` is specified explicitly by user.
        output_path : pathlib.Path, default None
            Path to the output file. If None, the default path is
            ``$PROJECT_DIR/database/runlog/elog_runs_filtered.${EXT}``, where
            ``${EXT}`` is the file extension specified by ``file_extension``.
        df : pd.DataFrame, default None
            Experimental runs to save. If None, the default is to use
            ``self.runs_final``.
        verbose : bool, default True
            Whether to print the message.
        
        Raises
        ------
        ValueError
            If ``file_extension`` is not one of 'csv' or 'h5'.
        """
        # determine the dataframe to save
        if df is None:
            df = self.runs_final
        
        # determine output path and file extension
        if output_path is None:
            output_path = PROJECT_DIR / f'database/runlog/elog_runs_filtered.{file_extension}'
        else:
            output_path = pathlib.Path(output_path)
            file_extension = output_path.suffix[1:] # remove the leading '.'
        
        # save file according to file extension
        if file_extension == 'h5':
            self._save_as_hdf({'runs': df}, output_path)
        elif file_extension == 'csv':
            self._save_as_csv(df, output_path)
        else:
            raise ValueError(f'Unsupported file extension: {file_extension}')

        if verbose:
            print(f'Filtered elog runs have been saved to "{str(output_path)}"')

    @staticmethod
    def _save_as_csv(df, filepath):
        """Save a dataframe as a csv file.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        filepath : str or pathlib.Path
            Path to the output file.
        """
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def _save_as_hdf(df, filepath):
        """Save a dataframe as an hdf file.

        Parameters
        ----------
        df : dict of pd.DataFrame
            A dictionary of dataframes to save.
        filepath : str or pathlib.Path
            Path to the output file.
        """
        with pd.HDFStore(filepath, 'w') as file:
            for key, value in df.items():
                file.append(key, value)

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

    def cleanse(self, verbose=True):
        df = {
            'runbeam': self._cleanse_runbeam(),
            'runtarget': self._cleanse_runtarget(),
            'runlog': self._cleanse_runlog(),
            'runscalernames': self._cleanse_runscalernames(),
        }

        path = pathlib.Path(CLEANSED_DIR, 'mysql_cleansed.h5')
        with pd.HDFStore(path, 'w') as file:
            for name, _df in df.items():
                file.append(name, _df)
        if verbose:
            print(f'Cleansed MySQL database saved to "{path}"')
        
        for name, _df in df.items():
            path = pathlib.Path(CLEANSED_DIR, f'mysql_{name}.txt')
            tables.to_fwf(_df, path)
            if verbose:
                print(f'Cleansed {name} saved to "{path}"')

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


if __name__ == '__main__':
    print('Data cleansing the ELOG...')
    elog_cleanser = ElogCleanser()
    elog_cleanser.cleanse()
    elog_cleanser.filtered_runs()
    elog_cleanser.save_cleansed_elog()
    elog_cleanser.save_filtered_runs()