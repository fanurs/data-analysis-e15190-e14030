#!/usr/bin/env python
import argparse
import concurrent.futures
import inspect
import pathlib
import subprocess
import textwrap
import time

from e15190 import PROJECT_DIR

EXECUTABLE = 'calibrate.exe'
OUTPUT_DIR = 'database/root_files' # relative to $PROJECT_DIR

def single_job(run, output_directory):
    outroot_path = output_directory / f'run-{run:04d}.root'
    log_path = f'./logs/run-{run:04d}.log'
    cmd = f'./{EXECUTABLE} -r {run} -o {outroot_path} |& tee {log_path}'
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
            jobs.append(executor.submit(single_job, run, args.outdir))
            time.sleep(2.0)

        for job in concurrent.futures.as_completed(jobs):
            n_runs_done += 1
            print(job.result(), f' [n_jobs: {n_runs_done}/{n_runs}]')

def get_arguments():
    parser = argparse.ArgumentParser(description='Running calibrate.exe in parallel')
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

    # warn if cores too many
    if args.cores > 8:
        print('WARNING: Using too many cores might cause issues for other users.')
        response = input('Are you sure you want to continue? [y/n]')
        if response != 'y':
            exit(1)

    # print out message
    print(f'Running calibrate.exe in parallel ({args.cores} cores) on runs:')
    print(f'{args.runs}')
    print(f'Output directory:')
    print(f'{str(args.outdir)}')

    return args

if __name__ == '__main__':
    main()
