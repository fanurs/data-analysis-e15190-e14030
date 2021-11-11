#!/usr/bin/env python
import argparse
import inspect

from e15190 import PROJECT_DIR
from e15190.neutron_wall import shadow_bar


def main():
    args = get_args()
    shade = shadow_bar.ShadowBar(args.AB)
    read_verbose = not args.silence
    bar_of_interest = [8, 16]
    bg_pos = ['F', 'B']
    for bar in bar_of_interest:
        for bg_position in bg_pos:
            print(f'Analyzing Bar {bar} Background position {bg_position}')
            shade.read(run=args.runs, bar=bar,
                       from_cache=not args.no_cache, verbose=read_verbose)
            read_verbose = False  # only show read status once
            shade.fit(bg_position)


def get_args():
    parser = argparse.ArgumentParser(
        description='Shadow Bar Analysis for Neutron Wall',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'AB',
        type=str,
        help='"A" or "B", this selects NWA or NWB'
    )
    # parser.add_argument(
    #     'bar',
    #     type=int,
    #     help='select the bar number to analyse'
    # )
    # parser.add_argument(
    #     'FB',
    #     type=str,
    #     help='"F" or "B", this selects forward or backward angle'
    # )
    parser.add_argument(
        'runs',
        nargs='+',
        help=inspect.cleandoc('''
            Runs to analyse.

            Usually the script needs at least five runs to give reliable
            calibrations, otherwise there is not enough statistics. Consecutive
            runs can be specified in ranges separated by the character "-". Here
            is an example:
                > ./shadow.py 8-10 11 20 2-5
            This will look into runs 8, 9, 10, 11, 20, 2, 3, 4, 5.
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

    # # process the bar number

    # if args.bar not in [8, 16]:
    #     raise ValueError(f'Invalid bar number: "{args.bar}"')

    # # process the position information
    # args.FB = args.FB.upper()
    # if args.FB not in ['F', 'B']:
    #     raise ValueError(f'Invalid position: "{args.FB}"')

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
