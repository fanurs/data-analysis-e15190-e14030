#!/usr/bin/env python3
import sys
if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    raise Exception('Requires at least Python 3.7 to use this script.')

import argparse
import concurrent.futures
import inspect
import os
import pathlib
import subprocess
import textwrap
import time

try:
    from e15190 import PROJECT_DIR
except ImportError:
    PROJECT_DIR = pathlib.Path(os.environ['PROJECT_DIR'])

EXECUTABLE = 'calibrate.exe'
OUTPUT_DIR = 'database/root_files/' # relative to $PROJECT_DIR

def single_job(run, output_directory, shared=None):
    if shared is None:
        shared = ''
    outroot_path = output_directory / f'run-{run:04d}.root'
    log_path = f'./logs/run-{run:04d}.log'
    cmd = f'./{EXECUTABLE} -r {run} -o {outroot_path} {shared} |& tee {log_path}'
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
        executable='/bin/bash',
    )
    if result.stderr != '':
        print(f'Error in run-{run:04d}: {result.stderr}', flush=True)
    return f'run-{run:04d}.root: ' + result.stdout.split('\n')[-2] # last line

def main():
    args = get_arguments()
    n_runs = len(args.runs)
    n_runs_done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.cores) as executor:
        jobs = []
        for run in args.runs:
            print(textwrap.fill(f'Submitting job for run-{run}'))
            jobs.append(executor.submit(single_job, run, args.outdir, args.shared))
            time.sleep(2.0)

        for job in concurrent.futures.as_completed(jobs):
            n_runs_done += 1
            print(f' [n_jobs: {n_runs_done}/{n_runs}] ' + job.result())

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Running calibrate.exe in parallel',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'runs',
        nargs='+',
        help=inspect.cleandoc(f'''
            Runs to calibrate.

            Calibrate the runs in parallel. Consecutive runs can be specified in
            ranges separated by the character "-". Here is an example:
                > ./batch_calibrate.py 8-10 11 20 2-5
            This will calibrate the runs 8, 9, 10, 11, 20, 2, 3, 4, 5.
        '''),
    )
    parser.add_argument(
        '-o', '--outdir',
        default=PROJECT_DIR / OUTPUT_DIR,
        help=inspect.cleandoc(f'''
            Output directory for the calibrated root files.

            By default, the output directory is $PROJECT_DIR/{str(OUTPUT_DIR)}.
        '''),
    )
    parser.add_argument(
        '-c', '--cores',
        default=4,
        type=int,
        help=inspect.cleandoc(f'''
            Number of cores to use. Please do not use more than 8 cores, as this
            might cause issues for other users.

            By default, the number of cores is 4.
        '''),
    )
    _outdir_short = pathlib.Path('$PROJECT_DIR') / OUTPUT_DIR
    parser.add_argument(
        '-s', '--shared',
        default='',
        type=str,
        help=inspect.cleandoc(f'''
            Shared argument, as a string, to pass to the "{EXECUTABLE}". For
            example, to use "-n" and "-i", we can do:
                > ./batch_calibrate.py 8-10 --cores 4 --shared "-n 1200 -i 55"
            This is equivalently to running the following jobs in parallel:
                > ./{EXECUTABLE} -r 8 -n 1200 -i 55 -o {_outdir_short}/run-0008.root
                > ./{EXECUTABLE} -r 9 -n 1200 -i 55 -o {_outdir_short}/run-0009.root
                > ./{EXECUTABLE} -r 10 -n 1200 -i 55 -o {_outdir_short}/run-0010.root
            
            Notice even though the flag "-o" is indeed a shared argument, we do
            not pass it to the "--shared" argument because the output directory
            has a default value that is not known to "{EXECUTABLE}", which is
            the default database. This is true even when using custom output
            directory, e.g. when debugging, it is common to output ROOT files to
            somewhere else, then we would do:
                > ./batch_calibrate.py 8-10 --cores 4 --outdir ./debug --shared "-n 1200 -i 55"
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
            raise ValueError(f'Unrecognized input: {run_str}')
    args.runs = runs

    # process the output directory
    args.outdir = pathlib.Path(args.outdir).resolve()
    if not args.outdir.exists():
        response = input(f'Output directory "{args.outdir}" does not exist.\nCreate it? [Y/n] ')
        if response != 'Y':
            exit(1)
        args.outdir.mkdir(parents=True, exist_ok=True)
        print(f'Created output directory "{args.outdir}"')

    # warn if cores too many
    if args.cores > 8:
        print('WARNING: Using too many cores might cause issues for other users.')
        response = input('Are you sure you want to continue? [Y/n]')
        if response != 'Y':
            exit(1)

    # print out message
    print(f'Running calibrate.exe in parallel ({args.cores} cores) on runs:')
    print(f'\t{args.runs}')
    print(f'Output directory:')
    print(f'\t"{str(args.outdir)}"')

    return args

if __name__ == '__main__':
    main()
