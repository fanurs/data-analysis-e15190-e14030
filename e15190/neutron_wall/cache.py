import concurrent.futures
import os
from pathlib import Path
from typing import Union
import warnings

import duckdb as dk
import pandas as pd
import sqlite3
import uproot

class RunCache:
    def __init__(self, src_path_fmt, cache_path_fmt, max_workers=8):
        """
        Parameters
        ----------
        src_path_fmt : str with 'run' specifier
            The format of the path to the ROOT file, e.g.
            ``'/home/user/data/CalibratedData_{run:04d}.root'``.
        cache_path_fmt : str with 'run' specifier
            The format of the path to the sqlite3 database file, e.g.
            ``'/mnt/analysis/data/cache/run-{run:04d}.db'``.
        max_workers : int, default 8
            The maximum number of thread pool executor workers to use for
            decompression and interpretation of ROOT files by Uproot.
        """
        self.SRC_PATH_FMT = src_path_fmt
        self.CACHE_PATH_FMT = cache_path_fmt
        self.max_workers = max_workers
    
    @staticmethod
    def infer_tree_name(uproot_file) -> str:
        names = list(set(key.split(';')[0] for key in uproot_file.keys()))
        candidate = None
        n_candidates = 0
        for name in names:
            if 'TTree' in str(type(uproot_file[name])):
                n_candidates += 1
                candidate = name
            if n_candidates > 1:
                break
        if n_candidates == 1:
            return candidate
        raise ValueError('Could not infer tree name from file')
    
    def read_run_from_root(self, run, branches, tree_name=None) -> pd.DataFrame:
        """Read in one run (one file) from Daniele's ROOT files.

        Parameters
        ----------
        run : int
            The run number.
        branches : dict of str:str or list of str
            If a dict, old_br_name -> new_br_name. The keys are the branch names
            at original path, and the values are the new names given to the
            dataframe. If a list, the branch names at path are used as the new
            names.
        tree_name : str, default None
            The name of the tree to read. If None, the tree name is inferred
            from the ROOT file. Infer is guaranteed only if there exists exactly
            one TTree in the ROOT file.
        """
        path = Path(os.path.expandvars(self.SRC_PATH_FMT.format(run=run)))
        if not isinstance(branches, dict):
            branches = {branch: branch for branch in branches}

        if tree_name is None:
            with uproot.open(str(path)) as file:
                tree_name = self.infer_tree_name(file)
        
        with (
            uproot.open(str(path) + ':' + tree_name) as tree,
            concurrent.futures.ThreadPoolExecutor(self.max_workers) as decompression_executor,
            concurrent.futures.ThreadPoolExecutor(self.max_workers) as interpretation_executor,
        ):
            df = tree.arrays(
                list(branches.keys()),
                library='pd',
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
            )

        if isinstance(df, (tuple, list)):
            raise ValueError(f'The provided branches cannot be deduced into a single dataframe.\n{branches}')
        df.columns = list(branches.values())
        return df
    
    @staticmethod
    def _get_table_name(run) -> str:
        return f'run{run:04d}'

    def save_run_to_sqlite(self, run, df : pd.DataFrame) -> Path:
        """Save the run as a dataframe in sqlite database.

        This is primarily used as cache for future use.

        Parameters
        ----------
        run : int
            The run number.
        df : pandas.DataFrame
            The dataframe to save.
        
        Returns
        -------
        path : pathlib.Path
            The path to the sqlite database file.
        """
        path = Path(os.path.expandvars(self.CACHE_PATH_FMT.format(run=run)))
        path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(path)) as conn:
            df.to_sql(self._get_table_name(run), con=conn)
        return path
    
    def read_run_from_sqlite(self, run, sql_cmd='') -> Union[pd.DataFrame, None]:
        path = Path(os.path.expandvars(self.CACHE_PATH_FMT.format(run=run)))
        if not path.is_file():
            return
        with sqlite3.connect(str(path)) as conn:
            df = pd.read_sql(f'SELECT * FROM {self._get_table_name(run)} {sql_cmd}', con=conn)
        return df.set_index(['entry', 'subentry'], drop=True)

    def read_run(
        self,
        run,
        branches,
        sql_cmd='',
        tree_name=None,
        from_cache=True,
        save_cache=True,
    ) -> pd.DataFrame:
        """Read in one run (one file) from Daniele's ROOT files.

        By default, the run is read from the SQLite cache. If the cache does not
        exist, the run is read from the ROOT file and saved to the cache.

        Parameters
        ----------
        run : int
            The run number.
        branches : dict of str:str or list of str
            If dict, old_br_name -> new_br_name. If list, branch names remain
            the same. If the function is reading from cache, this option is ignored;
            if the function is reading from ROOT, this option is required.
        sql_cmd : str, default ''
            The SQL command to be appended after 'SELECT * FROM table_name'. A
            space is automatically added between the table name and the command.
        tree_name : str, default None
            The name of the tree to read. If None, the tree name is inferred.
            Infer is guaranteed only if there exists exactly one TTree in the
            ROOT file.
        from_cache : bool, default True
            If True, the function always attempts to read from the cache, unless
            the cache does not exist. If False, cache will be ignored, and the
            run will be read from the ROOT file.
        save_cache : bool, default True
            If True, the run is saved to the cache after reading from the ROOT
            file. If False, the run will not be saved to the cache. This option
            is ignored if the function is already reading from the cache.
        
        Returns
        -------
        df : pandas.DataFrame
            The dataframe of the run.
        """
        df = None
        if from_cache:
            df = self.read_run_from_sqlite(run, sql_cmd=sql_cmd)
        if df is None:
            df = self.read_run_from_root(run, branches, tree_name=tree_name)
            if save_cache:
                self.save_run_to_sqlite(run, df)
            
            # duckdb query does not preserve the index (yet, hopefully)
            # so we set indices as regular columns before query, and then drop
            # them after query
            index_names = df.index.names
            for i, iname in enumerate(index_names):
                df[iname] = df.index.get_level_values(i)
            df = dk.query(f'SELECT * FROM df {sql_cmd}').df()
            df = df.set_index(index_names, drop=True)
        return df
    
    def read(
        self,
        runs,
        branches,
        sql_cmd='',
        drop_columns=None,
        tree_name=None,
        from_cache=True,
        save_cache=True,
        reset_index=False,
        insert_run_index=True,
        verbose=False,
    ) -> pd.DataFrame:
        """Read in multiple runs (multiple files) from Daniele's ROOT files.

        It is user's responsibility to ensure that the resultant dataframe fits
        into memory. Use ``sql_cmd`` to select only the rows of interest. This
        filter is applied to the single-run dataframes before being concatenated
        into the final dataframe, keeping the memory usage minimal.

        The entries and columns that are being cached are not affected by
        ``sql_cmd`` and ``drop_columns``. These options only take place when
        constructing the final dataframe and loading to memory.

        Parameters
        ----------
        runs : list of int or int
            The run numbers.
        branches : dict of str:str or list of str
            If dict, old_br_name -> new_br_name. If list, branch names remain
            the same. If the function is reading from cache, this option is ignored.
        sql_cmd : str, default ''
            The SQL command to be appended after 'SELECT * FROM table_name'. This
            option is ignored when reading from ROOT files.
        drop_columns : list of str, default None
            The columns to be dropped from the dataframe for each run. This is
            executed after ``pandas_query``. This has no effect on the columns
            that are being cached.
        tree_name : str, default None
            The name of the tree to read. If None, the tree name is inferred.
        from_cache : bool, default True
            If True, the function attempts to read from cache. If False, cache
            will be ignored, and the runs will be read from ROOT files.
        save_cache : bool, default True
            If True, the runs are saved to the cache after reading from ROOT
            files.
        reset_index : bool, default False
            If False, the original index is kept. The original index comes from
            uproot, where there is a two-level index of (entry, subentry). Entry
            enumerates the triggered events and is identical to the entry number
            in the ROOT file. Subentry enumerates the event multiplicity of a
            single event/entry in a detector, e.g. microBall multiplicity. If
            True, new single-level index starting from 0 is used.
        insert_run_index : bool, default True
            If True, 'run' is inserted as the first level index, above entry and
            subentry.
        verbose : bool, default False
            If True, print the progress of the function.

        Returns
        -------
        df : pandas.DataFrame
            The concatenated dataframe of all runs.
        """
        if isinstance(runs, int):
            runs = [runs]
        df = None
        for run in runs:
            if verbose:
                print(f'Reading run {run}...')
            df_run = self.read_run(
                run,
                branches,
                sql_cmd=sql_cmd,
                tree_name=tree_name,
                from_cache=from_cache,
                save_cache=save_cache,
            )
            if len(df_run) == 0:
                warnings.warn(f'Run {run} is empty or not found.')
            if insert_run_index:
                df_run.insert(0, 'run', run)
                df_run = df_run.set_index(['run'], append=True, drop=True)
                df_run = df_run.reorder_levels(['run', 'entry', 'subentry'])
            if drop_columns is not None:
                df_run = df_run.drop(drop_columns, axis=1)
            df = pd.concat([df, df_run], axis=0)
        if reset_index:
            df.reset_index(drop=True, inplace=True)
        return df
