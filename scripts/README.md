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
