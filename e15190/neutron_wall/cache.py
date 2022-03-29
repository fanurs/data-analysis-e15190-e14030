import concurrent.futures
from pathlib import Path
import os

import pandas as pd
import sqlite3
import uproot

class RunCache:
    def __init__(self, AB, src_path_fmt, cache_path_fmt, max_workers=8):
        """
        Parameters
        ----------
        AB : 'A' or 'B'
            Neutron wall A or B.
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
        self.AB = AB.upper()
        self.ab = AB.lower()
        self.SRC_PATH_FMT = src_path_fmt
        self.CACHE_PATH_FMT = cache_path_fmt
        self.max_workers = max_workers
        self.decompression_executor = None
        self.interpretation_executor = None
    
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
        
        if self.decompression_executor is None:
            self.decompression_executor = concurrent.futures.ThreadPoolExecutor(self.max_workers)
        if self.interpretation_executor is None:
            self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(self.max_workers)
        
        with uproot.open(str(path) + ':' + tree_name) as tree:
            df = tree.arrays(
                list(branches.keys()),
                library='pd',
                decompression_executor=self.decompression_executor,
                interpretation_executor=self.interpretation_executor,
            )
        if isinstance(df, (tuple, list)):
            raise ValueError(f'The provided branches cannot be deduced into a single dataframe.\n{branches}')
        df.columns = list(branches.values())
        return df
    
    @staticmethod
    def _get_table_name(run):
        return f'run{run:04d}'

    def save_run_to_sqlite(self, run, df) -> Path:
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
        with sqlite3.connect(str(path)) as conn:
            df.to_sql(self._get_table_name(run), con=conn, index=False)
        return path
    
    def read_run_from_sqlite(self, run, sql_cmd=''):
        path = Path(os.path.expandvars(self.CACHE_PATH_FMT.format(run=run)))
        if not path.exists():
            return False
        with sqlite3.connect(str(path)) as conn:
            return pd.read_sql(f'SELECT * FROM {self._get_table_name(run)} {sql_cmd}', con=conn)

    def read_single_run(
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
        if df is None or df is False:
            df = self.read_run_from_root(run, branches, tree_name=tree_name)
            if save_cache:
                self.save_run_to_sqlite(run, df)
        return df
    