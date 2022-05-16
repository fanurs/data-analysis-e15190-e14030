import hashlib
import inspect
import textwrap

import numpy as np

def convert_64_to_32(df):
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df

def randomize_columns(df, columns, seed=None):
    if isinstance(columns, str):
        columns = [columns]
    rand = np.random.RandomState(seed)
    df[columns] += rand.uniform(-0.5, 0.5, size=df[columns].shape)
    return df

def runs_hash(runs):
    """Returns a string that uniquely identifies the runs.
    """
    runs = sorted(set(runs))
    runs_hash = hashlib.sha256(','.join([str(run) for run in runs]).encode()).hexdigest()
    return f'run-{min(runs):04d}-{max(runs):04d}-h{runs_hash[-5:]}'

class MainUtilities:
    @staticmethod
    def wrap(multi_indented_line, **kwargs):
        line = multi_indented_line
        return textwrap.fill(inspect.cleandoc(line), **kwargs)
    
    @staticmethod
    def _parse_ranges(string_list):
        numbers = []
        for s in string_list:
            num_range = [int(num) for num in s.split('-')]
            if len(num_range) == 1:
                numbers.append(num_range[0])
            elif len(num_range) == 2:
                numbers.extend(range(num_range[0], num_range[1] + 1))
            else:
                raise ValueError(f'Unrecognized input: {s}')
        return numbers
    
    @staticmethod
    def parse_runs(args_runs, are_good=None, verbose=False):
        runs = MainUtilities._parse_ranges(args_runs)
        if are_good is None:
            return runs

        good_mask = are_good(runs)
        good_runs = [run for run, is_good in zip(runs, good_mask) if is_good]
        bad_runs = sorted(set(runs) - set(good_runs))
        if len(bad_runs) > 0:
            bad_runs_str = ', '.join([f'{run:04d}' for run in bad_runs])
            if verbose:
                print(MainUtilities.wrap(f'''
                    The following runs will be skipped because they are "bad":
                    {bad_runs_str}
                    '''
                ), flush=True)
        return list(good_runs)

    @staticmethod
    def parse_bars(args_bars):
        return MainUtilities._parse_ranges(args_bars)