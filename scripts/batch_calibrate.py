#!/usr/bin/env python
import argparse
import concurrent.futures
import functools
import inspect
import pathlib
import subprocess
import time

from e15190 import PROJECT_DIR

EXECUTABLE = 'calibrate.exe'
OUTPUT_DIR = PROJECT_DIR / 'database/root_files'
MAX_WORKERS = 4
subprocess.run = functools.partial(subprocess.run, capture_output=True, shell=True, text=True)

def single_job(run, output_directory=OUTPUT_DIR):
    outroot_path = output_directory / f'run-{run:04d}.root'
    log_path = f'./logs/run-{run}.log'
    cmd = f'./{EXECUTABLE} -r {run} -o {outroot_path} 2>&1 tee {log_path}'
    subprocess.run(cmd)
    return f'run-{run:04d} Done'

def main():
    args = get_arguments()
    n_runs = len(args.runs)
    n_runs_done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        jobs = []
        for run in args.runs:
            print(f'Submitting job for run-{run}')
            jobs.append(executor.submit(single_job, run, args.outdir))
            time.sleep(2.0)

        for job in concurrent.futures.as_completed(jobs):
            n_runs_done += 1
            print(job.result(), f' [{n_runs_done}/{n_runs}]')

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
        default=OUTPUT_DIR,
        help=inspect.cleandoc(f'''
            Output directory for the calibrated root files.

            By default, the output directory is {str(OUTPUT_DIR)}.
        '''),
    )
    parser.add_argument(
        '-c', '--cores',
        default=MAX_WORKERS,
        help=inspect.cleandoc(f'''
            Number of cores to use.

            By default, the number of cores is {MAX_WORKERS}.
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

    # print out message
    print(f'Running calibrate.exe in parallel ({args.cores} cores) on runs:')
    print(f'{args.runs}')
    print(f'Output directory:')
    print(f'{str(args.outdir)}')

    return args

if __name__ == '__main__':
    main()
