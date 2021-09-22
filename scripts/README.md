**Table of contents**
- [`calibrate.cpp`](#calibratecpp)
- [`NW_pulse_shape_discrimination.py`](#nw_pulse_shape_discriminationpy)

## [`calibrate.cpp`](calibrate.cpp)
This is the C++ script that is used to calibrate or re-calibrate all the data in "Daniele's ROOT files".

To compile, simply run:
```console
groot calibrate.cpp
```
Upon successful compilation, an executable file will be created with the name `calibrate.exe`. Type
```console
./calibrate.exe -h
```
to see the usage and options.

## [`NW_pulse_shape_discrimination.py`](NW_pulse_shape_discrimination.py)
This is the Python script that is used to obtain the pulse shape discrimination parameters.

To run the script, make sure that you have activated the conda environment for this repository.
Then make this script executable by doing
```console
chmod +x NW_pulse_shape_discrimination.py
```

Type
```console
./NW_pulse_shape_discrimination.py -h
```
to inspect the usage and options.
