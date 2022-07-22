#!/usr/bin/env python3
import argparse
import concurrent.futures
import inspect
import os
from pathlib import Path
import sqlite3

import uproot

from e15190.utilities import root6 as rt6

class RootToSQLite:
    def __init__(self, src_path_fmt=None, dst_path_fmt=None):
        self.tables = [
            'tdc',
            'mb',
            'fa',
            'vw',
            'nwb',
        ]

        if src_path_fmt is None:
            src_path_fmt = os.path.expandvars('$DATABASE_DIR/root_files/run-{run:04d}.root')
        self.src_path_fmt = src_path_fmt
        if dst_path_fmt is None:
            dst_path_fmt = os.path.expandvars('$DATABASE_DIR/sqlite_files/run-{run:04d}.db')
        self.dst_path_fmt = dst_path_fmt

        self.decompression_executor = None
        self.interpretation_executor = None
    
    def init_executors(self, force=False, n_workers=4):
        if self.decompression_executor is None or force:
            self.decompression_executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        if self.interpretation_executor is None or force:
            self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        
    @staticmethod
    def get_branches(path, tree, prefix):
        all_branches = rt6.get_all_branches(path, tree)
        return [br for br in all_branches if br.startswith(prefix)]

    def _get_subdataframe(self, path, tree_name, prefix, entry_start, n_entries=10_000):
        branches = self.get_branches(path, tree_name, prefix)
        with uproot.open((path) + ':' + tree_name) as tree:
            return tree.arrays(
                branches,
                library='pd',
                decompression_executor=self.decompression_executor,
                interpretation_executor=self.interpretation_executor,
                entry_start=entry_start,
                entry_stop=entry_start + n_entries,
            )
    
    @staticmethod
    def _standardize_index_names(df):
        std_names = ['entry', 'subentry', 'subsubentry']
        names = std_names[:len(df.index.names)]
        if len(names) == 1:
            names = names[0]
        df.index = df.index.rename(names)
        return df

    def get_dataframe(self, run, prefix, tree_name=None, n_entries_per_chunk=200_000, verbose=True):
        self.init_executors()
        path = self.src_path_fmt.format(run=run)
        if tree_name is None:
            tree_name = rt6.infer_tree_name(path)
        n_entries = rt6.get_n_entries(path, tree_name)
        for entry_start in range(0, n_entries, n_entries_per_chunk):
            if verbose:
                print(f'\r> entry = {entry_start:,}', end='')
            df = self._get_subdataframe(path, tree_name, prefix, entry_start, n_entries_per_chunk)
            yield self._standardize_index_names(df)
        if verbose:
            print()
    
    def convert(self, run, new_file=False, tables=None, verbose=True):
        """Convert a root file to an sqlite3 file.

        Parameters
        ----------
        run : int
            Experimental run number (HiRA)
        new_file : bool, default False
            If True, the old sqlite3 file, if exists, will be deleted before
            conversion.
        tables : list of str, default None
            Tables to convert. If None, all tables in :py:attr:`self.tables`
            will be converted.
        verbose : bool, default True
            If True, print progress.
        """
        in_path = self.src_path_fmt.format(run=run)
        if not Path(in_path).is_file():
            if verbose:
                print(f'{in_path} does not exist... skipping')
            return

        if verbose:
            print(f'Converting run-{run:04d}...')
        out_path = Path(self.dst_path_fmt.format(run=run))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if new_file:
            out_path.unlink(missing_ok=True)
        if tables is None:
            tables = self.tables
        with sqlite3.connect(out_path) as conn:
            for table in tables:
                if verbose:
                    print(f'Converting "{table}"...')
                for i, df in enumerate(self.get_dataframe(run, table.upper(), verbose=verbose)):
                    df.to_sql(table, conn, if_exists='append' if i > 0 else 'replace')
        if verbose:
            print(f'Done converting run-"{run:04d}"')

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Convert ROOT files to SQLite files.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'runs',
        nargs='+',
        help=inspect.cleandoc('''
            Runs to convert.

            Consecutive runs can be specified in ranges separated by the
            character "-". For example,
                > python root_to_sqlite.py 8-10 11 20 2-5
            converts runs 8, 9, 10, 11, 20, 2, 3, 4, 5.
        '''),
    )
    parser.add_argument(
        '-n', '--nworkers',
        default=1,
        type=int,
        help=inspect.cleandoc('''
            Number of workers for multiprocessing.
        '''),
    )
    args = parser.parse_args()

    # process the runs
    runs = []
    for run_str in args.runs:
        run_range = [int(run) for run in run_str.split('-')]
        if len(run_range) == 1:
            runs.append(run_range[0])
        elif len(run_range) == 2:
            runs.extend(range(run_range[0], run_range[1] + 1))
        else:
            raise ValueError(f'Invalid run range: {run_str}')
    args.runs = runs

    return args
    
def single_job(run, verbose=True):
    try:
        RootToSQLite().convert(run, verbose=verbose)
    except Exception as exc:
        print(f'Error converting run-{run:04d}:\n{exc}')
    return run
    
if __name__ == '__main__':
    args = get_arguments()

    if args.nworkers <= 1:
        for run in args.runs:
            single_job(run, verbose=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.nworkers) as executor:
            results = [executor.submit(single_job, run, verbose=False) for run in args.runs]
            for future in concurrent.futures.as_completed(results):
                print(f'> run-{future.result():04d} is completed')
