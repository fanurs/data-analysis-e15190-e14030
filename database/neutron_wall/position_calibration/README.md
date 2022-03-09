# Neutron wall position calibration

Currently, only Neutron Wall (NW) B is calibrated. To quickly inspect the overall result of position calibration, visit [infographics](https://groups.nscl.msu.edu/hira/15190-14030/fanurs/index.html?view=pos-calib).


## How to use the calibration parameters?

The file [`calib_params.json`](database/neutron_wall/position_calibration/calib_params.json) always gives the latest position calibration parameters. The first-level keys are bar numbers, ranging from 1 to 24 for NWB. Each entry (each bar) contains multiple run ranges as well as the corresponding calibration parameters. A _hypothetical example_ that contains only bar-26 and bar-27 is given below:
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

Finally, [`calib_params.dat`](database/neutron_wall/position_calibration/calib_params.dat) offers the same information but with a different presentation.

## Methodology
We have switched to using the (anti-)shadows of Veto Wall (VW) bars rather than just the neutron wall edges for calibration. This new approach is much more robust as there are many more VW bars (> 20), whereas the NW edges have only two. Also, NW edges, by definition, always suffer from some fuzziness that make them difficult to be pinpointed accurately. Below is a figure demonstrating how the VW bar (anti-)shadows are used for position calibration:<br>
<img width="800" alt="image" src="https://user-images.githubusercontent.com/21100851/157473846-5f6ae07b-977c-4fc5-893a-b941ee159572.png"><br>
When extracting the positions of VW bar shadows, we do it in two iterations - one on the even number VW bars, another on the odd number VW bars. This is to leave some gaps between the shadows, so that an algorithm can easily pick up the shadow peaks.

Lastly, we present the geometric coverage we simulated with 3D modeling for reference:<br>
<img height="400" alt="image" src="https://user-images.githubusercontent.com/21100851/157476510-8f98edf5-a7da-41b6-ba56-af19e5b3404e.png">
<img height="400" alt="image" src="https://user-images.githubusercontent.com/21100851/157476153-a8e3cd58-d1ae-4540-9e3e-fd59cfdef753.png">

