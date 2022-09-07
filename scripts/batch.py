#!/usr/bin/env python3
import sys
if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    raise Exception('Requires at least Python 3.7 to use this script.')

import argparse
import concurrent.futures
import inspect
import subprocess
import time

CMD = None

def single_job(run, cmd):
    global CMD
    log_path = f'./logs/run-{run:04d}.log'
    cmd = cmd.replace('RUN', f'{run:04d}')
    cmd += f' |& tee {log_path}'
    print(cmd, flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
        executable='/bin/bash',
    )
    if result.stderr != '':
        print(f'Error in run-{run:04d}: {result.stderr}', flush=True)
    return f'run-{run:04d}: ' + result.stdout.split('\n')[-2] # last line

def main():
    args = get_arguments()
    n_runs = len(args.runs)
    n_runs_done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.cores) as executor:
        jobs = []
        for run in args.runs:
            print('Submitting job: ', end='', flush=True)
            jobs.append(executor.submit(single_job, run, args.cmd))
            time.sleep(2.0)
        
        print('Done all submissions. Waiting for jobs to finish...', flush=True)

        for job in concurrent.futures.as_completed(jobs):
            n_runs_done += 1
            print(f' [n_jobs: {n_runs_done}/{n_runs}] ' + job.result())

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Running executable in parallel.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'cmd',
        type=str,
        help=inspect.cleandoc(f'''
            The full command to execute in parallel. Use "RUN" (case-sensitive)
            as a placeholder for the run number. The run number will be replaced
            using the format of "%%04d". For example,
                > ./batch.py "calibrate.exe -r RUN -o ./out_dir/run-RUN.root -n 1200" 8-10
            Logging will automatically be exported to "./logs/run-RUN.log".
        '''),
    )
    parser.add_argument(
        'runs',
        nargs='+',
        help=inspect.cleandoc(f'''
            Runs to execute in parallel.

            Consecutive runs can be specified in ranges separated by the
            character "-". Here is an example:
                > ./batch.py "calibrate.exe -r $RUN" 8-10 11 20 2-5
            This will calibrate the runs 8, 9, 10, 11, 20, 2, 3, 4, 5.
        '''),
    )
    parser.add_argument(
        '-c', '--cores',
        default=4,
        type=int,
        help=inspect.cleandoc(f'''
            Number of cores to use. Please do not use more than 8 cores, as this
            might cause issues for other users. By default, the number of cores
            is 4.
        '''),
    )
    parser.add_argument(
        '--good-runs-only',
        action='store_true',
        help=inspect.cleandoc(f'''
            If set, only runs that are marked as "good" in the run database will
            run. This option can be only be run when e15190 conda environment is
            active.
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
    if args.good_runs_only:
        from e15190.runlog.query import Query
        mask = Query().are_good(runs)
        runs = [run for run, is_good in zip(runs, mask) if is_good]
    args.runs = runs

    # warn if cores too many
    if args.cores > 8:
        print('WARNING: Using too many cores might cause issues for other users.')
        response = input('Are you sure you want to continue? [Y/n]')
        if response != 'Y':
            exit(1)

    # print out message
    print(f'Running calibrate.exe in parallel ({args.cores} cores) on runs:')
    print(f'\t{args.runs}')

    return args

if __name__ == '__main__':
    main()
