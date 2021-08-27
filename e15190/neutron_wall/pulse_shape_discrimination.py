import concurrent.futures

import numpy as np
import pandas as pd
import uproot 

from e15190 import PROJECT_DIR

_database_dir = PROJECT_DIR / 'database/neutron_wall/pulse_shape_discrimination'
_root_files_dir = PROJECT_DIR / 'database/root_files'

class PulseShapeDiscriminator:
    light_GM_range = [1.0, 200.0] # MeVee
    pos_range = [-120.0, 120.0] # cm
    adc_range = [0, 4096]

    def __init__(self, AB, max_workers=12):
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.decompression_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    @classmethod
    def _cut_for_root_file_data(cls, AB):
        cuts = [
            f'NW{AB}_light_GM > {cls.light_GM_range[0]}',
            f'NW{AB}_light_GM < {cls.light_GM_range[1]}',
            f'NW{AB}_pos > {cls.pos_range[0]}',
            f'NW{AB}_pos < {cls.pos_range[1]}',
            f'NW{AB}_total_L >= {cls.adc_range[0]}',
            f'NW{AB}_total_L <= {cls.adc_range[1]}',
            f'NW{AB}_total_R >= {cls.adc_range[0]}',
            f'NW{AB}_total_R <= {cls.adc_range[1]}',
            f'NW{AB}_fast_L >= {cls.adc_range[0]}',
            f'NW{AB}_fast_L <= {cls.adc_range[1]}',
            f'NW{AB}_fast_R >= {cls.adc_range[0]}',
            f'NW{AB}_fast_R <= {cls.adc_range[1]}',
        ]
        return ' & '.join([f'({c.strip()})' for c in cuts])

    def read_run_from_root_file(self, run, tree_name=None, apply_cut=True):
        path = _root_files_dir / f'run-{run:04d}.root'

        # determine the tree_name
        if tree_name is None:
            with uproot.open(str(path)) as file:
                objects = list(set(key.split(';')[0] for key in file.keys()))
            if len(objects) == 1:
                tree_name = objects[0]
            else:
                raise Exception(f'Multiple objects found in {path}')

        # load in the data
        branch_names = [
            'bar',
            'total_L',
            'total_R',
            'fast_L',
            'fast_R',
            'pos',
            'light_GM',
        ]
        branch_names = [f'NW{self.AB}_{name}' for name in branch_names]
        df = uproot.concatenate(
            f'{str(path)}:{tree_name}',
            branch_names,
            library='pd',
            decompression_executor=self.decompression_executor,
            interpretation_executor=self.interpretation_executor,
        )

        if apply_cut:
            df = df.query(self._cut_for_root_file_data(self.AB))
        return df
    
    def cache_run(self, run, tree_name=None):
        """Read in the data from ROOT file and save relevant branches to an HDF5 file.

        The data will be categorized according to bar number, because future
        retrieval by this class will most likely analyze only one bar at a time.
        """
        path = _database_dir / f'cache/run-{run:04d}.h5'
        df = self.read_run_from_root_file(run, tree_name=tree_name)

        # convert all float64 columns into float32
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        # write cache to HDF5 files bar by bar
        columns = [col for col in df.columns if col != f'NW{self.AB}_bar']
        for bar, subdf in df.groupby(f'NW{self.AB}_bar'):
            subdf.reset_index(drop=True, inplace=True)
            with pd.HDFStore(path, mode='a') as file:
                file.put(f'nw{self.ab}{bar:02d}', subdf[columns], format='fixed')
    
    def _read_single_run(self, run, bar, cache=True):
        path = _database_dir / f'cache/run-{run:04d}.h5'
        if not cache or not path.exists():
            self.cache_run(run)
        
        with pd.HDFStore(path, mode='r') as file:
            df = file.get(f'nw{self.ab}{bar:02d}')
        return df
    
    def read(self, run, bar, cache=True):
        if isinstance(run, int):
            runs = [run]
        else:
            runs = run
        
        df = None
        for run in runs:
            df_run = self._read_single_run(run, bar, cache=cache)
            df_run.insert(0, 'run', run)
            if df is None:
                df = df_run.copy()
            else:
                df = pd.concat([df, df_run], ignore_index=True)
        return df
    
    @staticmethod
    def randomize_integers(df, seed=None, inplace=True):
        if not inplace:
            df = df.copy()

        rng = np.random.default_rng(seed=seed)
        for name, column in df.iteritems():
            if name == 'run':
                continue
            if np.issubdtype(column.dtype, np.integer):
                df[name] += rng.uniform(low=-0.5, high=+0.5, size=len(column))
        return df
    