# Primary scripts for processing ROOT files

## The main script - [`calibrate.cpp`](calibrate.cpp)
This is the script that is used to calibrate or re-calibrate all the data in "Daniele's ROOT files". Basically, it takes in, for example, `CalibratedData_4083.root`, and produces `database/root_files/run-4083.root`.

> **:warning: For NSCL/FRIB Fishtank users**<br>
> Starting from March 10, 2022, the ROOT installation under conda envrionment has failed to compile any C++ scripts (issue #12), even though both PyROOT and [Cling](https://github.com/root-project/cling) are still working fine. If you are *not* on Fishtank, your conda environment may still work, so you can skip to the [usage section](#usage).
> 
> The current solution is that when compiling or running `calibrate.cpp`, please turn off conda environment by `conda deactivate` and use the system-wide installation of `root/gnu/6.24.02`:
> ```console
> (env_e15190) user@server $ cd scripts/
> (env_e15190) user@server $ conda deactivate
> (base) user@server $ conda deactivate            # if you have installed g++ or ROOT here
> user@server $ source config.sh
> Loading root/gnu/6.24.02
>   Loading requirement: gnu/gcc/9.3
> user@server $ root -l
> root [0] 
> ```
> If you are using the terminal within VS Code, you may see a slightly different message:
> ```console
> user@server $ source config.sh
> The command "module load" is not available.
> Using hard-coded paths instead.
> ```
> This shall *not* affect the compilation and running of the script. It's just that you cannot easily "unload" the ROOT, other than restarting a new terminal session.


### Usage
To compile:
```console
cd scripts/
make
```
The old method of using custom `groot` is deprecated and will be removed soon.

Upon successful compilation, an executable file will be created with the name `calibrate.exe`. To calibrate, say, run 4083, type in:
```console
./calibrate.exe -r 4083 -o demo-4083.root
```
To inspect all the options, enter `./calibrate.exe -h`.


### Running [`calibrate.cpp`](calibrate.cpp) in parallel (*non-SLURM solution*)
To do this, we invoke the script [`batch_calibrate.py`](batch_calibrate.py), which basically uses the [`concurrent.futures`](https://docs.python.org/3.8/library/concurrent.futures.html) standard library in Python. This script can be run with *any* python 3.7 or above (not necesarily a conda one), as long as you are in an environment where `./calibrate.exe` can still be executed correctly (usually the environment you used for compilation) and the environment variable `$PROJECT_DIR` has been set.

As a demonstration, we calibrate runs 4085 to 4090 and 4095 to 4100 in parallel as follows:
```console
user@server $ python3 batch_calibrate.py 4085-4090 4095-4100 -c 8 -o ./demo -s "-n 1000000"
Output directory "./demo" does not exist.
Create it? [Y/n] Y
Created output directory "./demo"
Running calibrate.exe in parallel (8 cores) on runs:
    [4085, 4086, 4087, 4088, 4089, 4090, 4095, 4096, 4097, 4098, 4099, 4100]
Output directory:
    "./demo"
Submitting job for run-4085
Submitting job for run-4086
...
Submitting job for run-4099
Submitting job for run-4100
 [n_jobs: 1/12] run-4090.root: > 100.00%        i_evt: 999,999/999,999      (total_nevts: 4,516,454)
 [n_jobs: 2/12] run-4089.root: > 100.00%        i_evt: 999,999/999,999      (total_nevts: 4,440,795)
 ...
 [n_jobs: 11/12] run-4099.root: > 100.00%        i_evt: 999,999/999,999      (total_nevts: 3,542,890)
 [n_jobs: 12/12] run-4100.root: > 100.00%        i_evt: 999,999/999,999      (total_nevts: 4,371,005)
```

For Fishtank users, please do not use too many cores to occupy the majority of the *shared* CPU resources. The [`batch_calibrate.py`](batch_calibrate.py) is only a quick parallel solution that is good for speeding up the process by less than 10 times, or a little more than that if you are running things after hours. If you want to attain more parallelization, e.g. 100 times or above, please use a SLURM solution, e.g. NSCL/FRIB's ember cluster or MSU's HPCC.