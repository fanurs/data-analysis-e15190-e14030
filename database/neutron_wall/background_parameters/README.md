The [`background.dat`](background.dat) file contains the parameter of the exponential decay background function
```
bg = amplitude * exp(-K / decay)
```
where `K` is the relativistic kinetic energy of neutrons in MeV, and `bg` is the background portion (so `1 - bg` is the "actual neutrons") in **percentage**, ranging from 0% to 100%. Both `amplitude` and `decay` are parameters that can be read from the file.
