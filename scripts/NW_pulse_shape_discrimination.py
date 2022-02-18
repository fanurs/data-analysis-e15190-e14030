#!/usr/bin/env python
import argparse
import inspect

from e15190 import PROJECT_DIR
from e15190.neutron_wall import pulse_shape_discrimination

def main():
    args = get_args()

    psd = pulse_shape_discrimination.PulseShapeDiscriminator(args.AB)
    read_verbose = not args.silence
    for bar in range(1, 24 + 1):
        psd.read(run=args.runs, bar=bar, from_cache=not args.no_cache, verbose=read_verbose)
        read_verbose = False # only show read status once

        if not args.silence:
            print(f'\rCalibrating NW{args.AB}-bar{bar:02d}...', end='', flush=True)

        psd.fit()
        psd.save_parameters()
        psd.save_to_gallery()

    if not args.silence:
        print()

def get_args():
    parser = argparse.ArgumentParser(
        description='Pulse shape discrimination for neutron wall',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'AB',
        type=str,
        help='"A" or "B", this selects NWA or NWB'
    )
    parser.add_argument(
        'runs',
        nargs='+',
        help=inspect.cleandoc('''
            Runs to calibrate.

            Usually the script needs at least five runs to give reliable
            calibrations, otherwise there is not enough statistics. Consecutive
            runs can be specified in ranges separated by the character "-". Here
            is an example:
                > ./NW_pulse_shape_discrimination.py B 8-10 11 20 2-5
            This will calibrate the runs 8, 9, 10, 11, 20, 2, 3, 4, 5, on NWB.
        '''),
    )
    parser.add_argument(
        '-c', '--no-cache',
        help=inspect.cleandoc('''
            When this option is given, the script will ignore the HDF5 cache
            files. All data will be read from the ROOT files. New cache files
            will then be created. By default, the script will use the cache.
        '''),
        action='store_true',
    )
    parser.add_argument(
        '-s', '--silence',
        help='To silent all status messages.',
        action='store_true',
    )
    args = parser.parse_args()

    # process the wall type
    args.AB = args.AB.upper()
    if args.AB not in ['A', 'B']:
        raise ValueError(f'Invalid wall type: "{args.AB}"')
    
    # process the runs
    runs = []
    for run_str in args.runs:
        run_range = [int(run) for run in run_str.split('-')]
        if len(run_range) == 1:
            runs.append(run_range[0])
        elif len(run_range) == 2:
            runs.extend(range(run_range[0], run_range[1] + 1))
        else:
            raise ValueError(f'Unrecognized input: {run_str}')
    args.runs = runs

    return args

if __name__ == "__main__":
    main()