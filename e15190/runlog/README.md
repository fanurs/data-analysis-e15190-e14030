# The `e15190.runlog` module

> Currently, only the Elog is being processed fully.


## First-timer must-do's

To enable to runlog module, some files have to be created locally to [`database/`](database) directory. These files are not being tracked by git for various reasons, e.g. confidentiality, huge binary files, etc. Fortunately, all the files can easily be reproduced by following the instructions below. Make sure you have activated the conda environment.

1. Go to this directory:
    ```console
    $ cd $PROJECT_DIR/e15190/runlog/
    ```
1. Download a local copy of the Elog to `database/runlog/downloads/elog.html`:
    ```console
    $ python downloader.py
    Attempting to download web content from
    "http://neutronstar.physics.wmich.edu/runlog/index.php?op=list"... 
    Done!
    ```
1. Data cleansing to produce several files:
    ```console
    $ python data_cleansing.py
    Data cleansing the ELOG...
    Cleansed runs and events have been saved to "$PROJECT_DIR/database/runlog/cleansed/elog.h5"
    Cleansed runs dataframe has been saved to "$PROJECT_DIR/database/runlog/cleansed/elog_runs.csv"
    Cleansed events has been saved to "$PROJECT_DIRdatabase/runlog/cleansed/elog_events.csv"
    Filtered elog runs have been saved to "$PROJECT_DIR/database/runlog/elog_runs_filtered.h5"
    Filtered elog runs have been saved to "$PROJECT_DIR/database/runlog/elog_runs_filtered.csv"
    ```

That's it! To check, you should now be able to run the following python script:
```python
from e15190.runlog.query import Query
Query.get_run_info(4083)
# {'run': 4083,
#  'begin_time': Timestamp('2018-03-10 17:25:50'),
#  'end_time': Timestamp('2018-03-10 17:56:09'),
#  'elapse': Timedelta('0 days 00:30:19'),
#  'target': 'Ni64',
#  'beam': 'Ca48',
#  'beam_energy': 140.0,
#  'shadow_bar': 'in',
#  'trigger_rate': 2608.0,
#  'comment': '140MeV 64Ni, coincidence trigger, uB DS, RF trig'
# }
```