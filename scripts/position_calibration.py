import concurrent.futures
import pathlib
import sys

from pandas._libs import missing

sys.path.append('../')
from e15190.neutron_wall import position_calibration as pc
from e15190.utilities import timer

def main():
    runs = list(range(2000, 3000)) + list(range(4000, 4663))
    njobs_done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        jobs = [executor.submit(single_job, run) for run in runs]
        
        for job in concurrent.futures.as_completed(jobs):
            njobs_done += 1
            print(f'>> run-{job.result():04d} Done! ({njobs_done}/{len(runs)})', flush=True)

def single_job(run):
    timer.start()

    try:
        calib = pc.NWBPositionCalibrator(
            max_workers=4,
            stdout_path=f'./logs/run-{run:04d}.log',
            verbose=True,
        )

        if not calib.read_run(run):
            pass
        elif calib.calibrate():
            calib.save_parameters()
            calib.save_to_gallery(show_plot=False)
    except:
        print('Caught some errors/exceptions.', flush=True)

    # remove log file of short elapse job
    elapse = timer.stop(prefix='\n' + '-' * 30 + f'\n> Elapse run-{run:04d}: ', suffix=' sec.\n')
    if elapse < 1.0:
        path = pathlib.Path(f'./logs/run-{run:04d}.log')
        path.unlink(missing_ok=True)

    return run

if __name__ == '__main__':
    main()