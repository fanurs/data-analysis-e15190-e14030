import concurrent.futures

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import uproot 

from e15190 import PROJECT_DIR

class PulseShapeDiscriminator:
    database_dir = PROJECT_DIR / 'database/neutron_wall/pulse_shape_discrimination'
    root_files_dir = PROJECT_DIR / 'database/root_files'
    light_GM_range = [1.0, 200.0] # MeVee
    pos_range = [-120.0, 120.0] # cm
    adc_range = [0, 4000] # the upper limit is 4096, but those would definitely be saturated.

    def __init__(self, AB, max_workers=12):
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.decompression_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.features = [
            f'NW{self.AB}_pos',
            f'NW{self.AB}_total_L',
            f'NW{self.AB}_total_R',
            f'NW{self.AB}_fast_L',
            f'NW{self.AB}_fast_R',
            f'NW{self.AB}_light_GM',
        ]
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.root_files_dir.mkdir(parents=True, exist_ok=True)
    
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
        path = self.root_files_dir / f'run-{run:04d}.root'

        # determine the tree_name
        if tree_name is None:
            with uproot.open(str(path)) as file:
                objects = list(set(key.split(';')[0] for key in file.keys()))
            if len(objects) == 1:
                tree_name = objects[0]
            else:
                raise Exception(f'Multiple objects found in {path}')

        # load in the data
        nw_branches = [
            'bar',
            'total_L',
            'total_R',
            'fast_L',
            'fast_R',
            'pos',
            'light_GM',
        ]
        branch_names = [f'NW{self.AB}_{name}' for name in nw_branches]
        branch_names.append('VW_multi')
        with uproot.open(str(path) + ':' + tree_name) as tree:
            df = tree.arrays(
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
        path = self.database_dir / f'cache/run-{run:04d}.h5'
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
    
    def _read_single_run(self, run, bar, from_cache=True):
        path = self.database_dir / f'cache/run-{run:04d}.h5'
        if not from_cache or not path.exists():
            self.cache_run(run)
        
        with pd.HDFStore(path, mode='r') as file:
            df = file.get(f'nw{self.ab}{bar:02d}')
        return df
    
    def read(self, run, bar, from_cache=True, verbose=False):
        if isinstance(run, int):
            runs = [run]
        else:
            runs = run
        
        df = None
        for run in runs:
            if verbose:
                print(f'\rReading run-{run:04d}', end='', flush=True)
            df_run = self._read_single_run(run, bar, from_cache=from_cache)
            df_run.insert(0, 'run', run)
            if df is None:
                df = df_run.copy()
            else:
                df = pd.concat([df, df_run], ignore_index=True)
        if verbose:
            print('\r', flush=True)
        self.df = df
        return df
    
    def randomize_integer_features(self, seed=None):
        rng = np.random.default_rng(seed=seed)
        for name, column in self.df.iteritems():
            if name not in self.features:
                continue
            if np.issubdtype(column.dtype, np.integer):
                self.df[name] += rng.uniform(low=-0.5, high=+0.5, size=len(column))
                self.df[name] = self.df[name].astype(np.float32)
    
    def normalize_features(self):
        self.feature_scaler = StandardScaler().fit(self.df[self.features])
        self.df[self.features] = self.feature_scaler.transform(self.df[self.features])
    
    def denormalize_features(self):
        self.df[self.features] = self.feature_scaler.inverse_transform(self.df[self.features])
    
    def remove_vetowall_coincidences(self):
        self.df = self.df.query('VW_multi == 0')
    
    def discrimination_using_pca(self):
        self.remove_vetowall_coincidences()
        self.randomize_integer_features()
        self.normalize_features()

        self.pca = decomposition.PCA(n_components=len(self.features))
        self.pca.fit(self.df[self.features])
        self.psd_params_from_pca = None
        for component in self.pca.components_:
            total_L = component[self.features.index(f'NW{self.AB}_total_L')]
            total_R = component[self.features.index(f'NW{self.AB}_total_R')]
            if np.sign(total_L) != np.sign(total_R):
                continue
            fast_L = component[self.features.index(f'NW{self.AB}_fast_L')]
            fast_R = component[self.features.index(f'NW{self.AB}_fast_R')]
            slope_L = fast_L / total_L
            slope_R = fast_R / total_R
            if np.isclose(slope_L, -1.0, atol=0.2) and np.isclose(slope_R, -1.0, atol=0.2):
                self.psd_params_from_pca = component.copy()
                self.psd_params_from_pca *= np.sign(total_L)
                break
        if self.psd_params_from_pca is not None:
            self.df['pca_psd'] = np.dot(self.df[self.features], self.psd_params_from_pca)
            return True
        else:
            return False