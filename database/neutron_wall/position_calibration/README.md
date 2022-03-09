# Neutron wall position calibration

Currently, only NWB is calibrated. We have switched to using the (anti-)shadows of VetoWall bars rather than just the neutron wall edges for calibration.

To quickly inspect the result of position calibration, visit [infographics](https://groups.nscl.msu.edu/hira/15190-14030/fanurs/index.html?view=pos-calib).

## Structure of [`calib_params.json`](database/neutron_wall/position_calibration/calib_params.json)
This is the file that always gives the latest position calibration parameters. The first-level keys are bar numbers, ranging from 1 to 24 for NWB. Each entry (each bar) contains multiple run ranges as well as the corresponding calibration parameters. A _hypothetical example_ that contains only bar-26 and bar-27 is given below:
```json
{
    "26": [
        {"run_range": [100, 250],      "parameters": [40.5, 7.2]},
        {"run_range": [251, 306],      "parameters": [-25.2, 7.1]},
        {"run_range": [307, 330],      "parameters": [2.8, 7.3]}
    ],
    "27": [
        {"run_range": [100, 288],      "parameters": [-22.5, 6.9]},
        {"run_range": [289, 330],      "parameters": [-12.9, 7.0]}
    ]
}
```
Several remarks:
1. Different bars can have different sets of run ranges and, of course, different calibration parameters.
2. The numbers in `run_range` are inclusive, e.g. `[100, 104]` includes runs `100, 101, 102, 103, 104`.
3. When parsing in the file, it is _safe_ to assume the run ranges are ordered.
4. For runs that do not belong to any run ranges, that implies no position calibration was done for them. Users may choose to extrapolate from the closest run range.
5. Similarly, not all runs in a run range are good for physics, or even exist. The boundaries of these run ranges are defined at where calibration parameters have undergone some statistically significant changes. Even if there were some junk runs within a particular run range, it does not necesssarily split the run range into two. An example would be `run_range = [100, 120]`, and assume that `run = 109` does not exist. But as long as the calibration parameters in `[100, 108]` are similar to those in `[110, 120]`, we would still group both sets of runs into one, sharing the same parameters (averages).
6. The calibration was done assuming a simple linear relation with the time difference of PMTs:
    ```python
    time_difference = time_left - time_right # nanosecond
    position = parameters[0] + parameters[1] * time_difference # centimeter
    ```

Finally, [calib_params.dat](database/neutron_wall/position_calibration/calib_params.dat) offers the same information but with a different presentation.