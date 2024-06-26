from __future__ import annotations
from inspect import cleandoc
from pathlib import Path
from os.path import expandvars
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.interpolate import UnivariateSpline # type: ignore

from e15190.utilities import tables

class geNt4:
    PATH = '$PROJECT_DIR/database/neutron_wall/efficiency/intrinsic-efficiency-geNt4.dat'

    @classmethod
    def get_intrinsic_efficiency_data(
        cls, path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Returns the intrinsic efficiency data points calculated with geNt4.

        Parameters
        ----------
        path : str or Path, optional
            The path to the file containing the intrinsic efficiency data points.
            If None, the path specified by :py:attr:`PATH` is used.
        
        Returns
        -------
        data : pandas.DataFrame
            The intrinsic efficiency data points, with columns 'energy' and
            'efficiency'.
        """
        if path is None:
            path = expandvars(cls.PATH)
        path = Path(path)
        return pd.read_csv(path, delim_whitespace=True, comment='#')

    @classmethod
    def get_intrinsic_efficiency(
        cls, path: Optional[str | Path] = None,
    ) -> Callable[[ArrayLike], ArrayLike]:
        """Returns the intrinsic efficiency curve calculated with geNt4.

        A spline is used to interpolate the efficiency data points. Moreover, a
        boundary condition of zero is imposed at zero energy, extending the
        efficiency curve below the lowest energy data point all the way down to
        zero. For energy values below zero or above the highest energy data
        point, the efficiency is set to zero.

        Parameters
        ----------
        path : str or Path, optional
            The path to the file containing the intrinsic efficiency curve. If
            None, the path specified by :py:attr:`PATH` is used.
        
        Returns
        -------
        intrinsic_efficiency : callable
            The intrinsic efficiency curve.
        """
        df = cls.get_intrinsic_efficiency_data(path)

        # insert (0, 0) to the beginning of the dataframe
        df = pd.concat([pd.DataFrame([[0, 0]], columns=df.columns), df], ignore_index=True)

        return UnivariateSpline(df.energy, df.efficiency, s=0, ext='zeros')

class ScinfulQmd:
    PATH_FMT = '/mnt/simulations/LANA_simulations/old/berdosa/outputs/E15190-att25-{mode}/{energy:03d}/res00.out'
    RESULT_PATH = '$DATABASE_DIR/neutron_wall/efficiency/scinful-qmd-eff-curve.txt'

    @staticmethod
    def get_efficiency_curve(from_raw_output=False, save_txt=False) -> pd.DataFrame:
        """Returns the efficiency curve.

        Parameters
        ----------
        from_raw_output : bool, default False
            If True, the efficiency curve is calculated from the "res00.out" files. Otherwise, the
            efficiency curve is read from the file specified by :py:attr:`RESULT_PATH`.
        save_txt : bool, default False
            If True, the efficiency curve is saved to the file specified by :py:attr:`RESULT_PATH`.
        
        Returns
        -------
        efficiency_curve : pandas.DataFrame
            The efficiency curve, with columns 'energy', 'efficiency', and 'mode'.
        """
        result_path = Path(expandvars(ScinfulQmd.RESULT_PATH))
        if from_raw_output:
            scinful_eff_curve = ScinfulQmd.read_efficiency_curve('scinful', (1, 300))
            qmd_eff_curve = ScinfulQmd.read_efficiency_curve('qmd', (131, 300))

            joint_curve = pd.concat([scinful_eff_curve.query('energy <= 80'), qmd_eff_curve])
            spl_joint_curve = UnivariateSpline(joint_curve.energy, joint_curve.efficiency, s=0)
            joint_curve = pd.DataFrame(
                [[energy, np.round(spl_joint_curve(energy), 8)] for energy in range(1, 300 + 1)],
                columns=['energy', 'efficiency']
            )
        else:
            joint_curve = pd.read_csv(result_path, delim_whitespace=True, comment='#')
        joint_curve['mode'] = 'interpolated'
        joint_curve.loc[joint_curve.energy <= 80, 'mode'] = 'scinful'
        joint_curve.loc[joint_curve.energy >= 131, 'mode'] = 'qmd'

        if save_txt:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            tables.to_fwf(
                joint_curve, result_path,
                comment=cleandoc('''
                # Input.data:
                #       None
                #    1000000, -213.0000,    0.0400
                #       None,    0.0000,    0.0000,    0.0010,       300
                #       None
                #    0.0000,    0.0000, -450.0000
                #    4.3000,    6.3500,    4.3000
                #         2,         1,         2,         3
                '''),
            )
        return joint_curve

    @staticmethod
    def read_response(energy: int, mode: Literal['scinful', 'qmd']) -> pd.DataFrame:
        """Read in the light response table from "res00.out" file.

        Parameters
        ----------
        energy : int
            The energy of the incident neutron in MeV.
        mode : {'scinful', 'qmd'}
            The mode of the simulation. Scinful is used for energies below 80
            MeV, and QMD is used for energies above 80 MeV.
        
        Returns
        -------
        light_response : pd.DataFrame
            The light response table, with columns 'light', 'resp', and 'err'.
        """
        path = ScinfulQmd.PATH_FMT.format(
            mode='scin' if mode == 'scinful' else 'qmd',
            energy=energy,
        )
        df = pd.read_csv(path, delim_whitespace=True, skiprows=8, header=None)
        df.columns = ['low', 'upp', 'resp', 'err']
        light = 0.5 * (df.low + df.upp)
        df.drop(columns=['low', 'upp'], inplace=True)
        df.insert(0, 'light', light)
        return df

    @staticmethod
    def calculate_efficiency(light_response: pd.DataFrame, bias: float = 3.0) -> float:
        """Calculate the efficiency from the light response table.

        Parameters
        ----------
        light_response : pd.DataFrame
            The light response table, containing at least columns 'light' and 'resp'.
        bias : float, default 3.0
            The bias light output in MeVee. This serves as the lower bound of the
            integral.
        
        Returns
        -------
        efficiency : float
            The detection efficiency.
        """
        spl = UnivariateSpline(light_response.light, light_response.resp, s=0)
        return spl.integral(bias, np.array(light_response.light)[-1])

    @staticmethod
    def read_efficiency_curve(mode: Literal['scinful', 'qmd'], energy_range: Tuple[int, int]) -> pd.DataFrame:
        """Read in the efficiency curve from the "res00.out" files.

        Parameters
        ----------
        mode : {'scinful', 'qmd'}
            The mode of the simulation.
        energy_range : tuple of int
            The inclusive energy range of the incident neutron in MeV. Scinful
            mode offers a range from 1 to 300 MeV, and QMD mode offers a range
            from 131 to 300 MeV.
        
        Returns
        -------
        efficiency_curve : pd.DataFrame
            The efficiency curve, with columns 'energy' and 'efficiency'.
        """
        return pd.DataFrame([
            [energy, ScinfulQmd.calculate_efficiency(ScinfulQmd.read_response(energy, mode))]
            for energy in range(energy_range[0], energy_range[1] + 1)
        ], columns=['energy', 'efficiency'])
