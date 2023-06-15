```python
#!/usr/bin/env python3
from __future__ import annotations
from copy import deepcopy
import itertools

from e15190.hira import spectra as hira_spectra
from e15190.utilities import dataframe_histogram as dfh

reactions = {40: 'ca40ni58e140', 48: 'ca48ni64e140'}
particles = ['p', 'd', 't', 'He3', 'He4']

# read in hira spectra from ROOT files
df = {40: dict(), 48: dict()}
for beamA, particle in itertools.product(df.keys(), particles):
    df[beamA][particle] = (hira_spectra
        .LabPtransverseRapidity(reaction=reactions[beamA], particle=particle)
        .get_ptA_spectrum(rapidity_range=(0.4, 0.6), correct_coverage=True, range=(100, 600), bins=500 // 20, drop_outliers=-1)
    )
df_original = deepcopy(df)

# construct compound spectra and ratios
for beamA in [40, 48]:
    df[beamA]['n'] = hira_spectra.PseudoNeutron(df_original[beamA]).get_spectrum(fit_range=(200, 350), switch_point=350)
    df[beamA]['CI-n'] = hira_spectra.CoalescenseInvariant(df_original[beamA]).get_neutron_spectrum(fit_range=(200, 350), switch_point=350)
    df[beamA]['CI-p'] = hira_spectra.CoalescenseInvariant(df_original[beamA]).get_proton_spectrum()

    df[beamA]['CI-np'] = dfh.div(df[beamA]['CI-n'], df[beamA]['CI-p'])
double_ratio = dfh.div(df[48]['CI-np'], df[40]['CI-np'])

# save to csv
for beamA in [40, 48]:
    df[beamA]['n'].to_csv(f'{reactions[beamA]}-pseudo-n.csv', index=False)
    df[beamA]['CI-n'].to_csv(f'{reactions[beamA]}-CI-n.csv', index=False)
    df[beamA]['CI-p'].to_csv(f'{reactions[beamA]}-CI-p.csv', index=False)
    df[beamA]['CI-np'].to_csv(f'{reactions[beamA]}-CI-np.csv', index=False)
double_ratio.to_csv(f'CI-double-np.csv', index=False)
```