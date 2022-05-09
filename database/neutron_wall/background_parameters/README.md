The [`background.dat`](background.dat) file contains the parameter of the exponential decay background function
```
bg = amplitude * exp(-E / decay)
```
where `E` is the relativistic kinetic energy of neutrons in MeV, and `bg` is the background portion (so `1 - bg` is the "actual neutrons") in percentage, ranging from 0% to 100%.
