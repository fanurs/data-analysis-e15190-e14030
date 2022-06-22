from copy import copy
import os
from pathlib import Path
import re
from typing import Literal

from astropy import constants
import pandas as pd
import ROOT
from scipy.interpolate import UnivariateSpline

from e15190.runlog.query import Query
from e15190.neutron_wall import geometry as nwgeo
from e15190.utilities import (
    atomic_mass_evaluation as ame,
    root6 as rt6,
    physics,
)

def enable_implicit_multithreading():
    ROOT.EnableImplicitMT()

class Cuts:
    """A class of default cuts for neutron wall spectra.

    This class only offers the typical cuts that are being applied. If you want
    to change the cuts by a lot, it might be easier to simply write your own
    strings.
    """
    @staticmethod
    def edge(left=nwgeo.Bar.edges_x[0], right=nwgeo.Bar.edges_x[1], on='NWB'):
        """Cut for the two edges of a neutron wall bar.

        This cut is used to remove the ambiguity near the edges of the bar.

        Parameters
        ----------
        left : float, default :py:attr:`e15190.neutron_wall.geometry.Bar.edges_x[0]`
            The left edge of the bar in cm.
        right : float, default :py:attr:`e15190.neutron_wall.geometry.Bar.edges_x[1]`
            The right edge of the bar in cm.
        """
        x = f'{on}_pos'
        return f'{x} > {left} && {x} < {right}'

    @staticmethod
    def shadow_bars(on='NWB'):
        """Cut for the regions on the neutron wall that are shadowed.

        We remove the segments `-50 < x < -15` and `15 < x < 50` for NWB bars 7,
        8, 9, 15, 16, 17.
        """
        bar = f'{on}_bar'
        x = f'{on}_pos'
        return ' && '.join([
            f'!(({bar} == 7 || {bar} == 8 || {bar} == 9) && {x} > -50 && {x} < -15)',
            f'!(({bar} == 7 || {bar} == 8 || {bar} == 9) && {x} > 15 && {x} < 50)',
            f'!(({bar} == 15 || {bar} == 16 || {bar} == 17) && {x} > -50 && {x} < -15)',
            f'!(({bar} == 15 || {bar} == 16 || {bar} == 17) && {x} > 15 && {x} < 50)',
        ])
    
    @staticmethod
    def psd(psd_low=0.5, psd_upp=4.0, perp_low=-0.7, perp_upp=0.7, on='NWB'):
        """Cut for pulse shape discrimination.

        We used two-dimensional pulse shape discrimination (PSD). The first
        dimension or axis accounts for most of the separation, while the second
        dimension is perpendicular to the first one.

        This cut only makes sense after vetoing the charged particles.

        Parameters
        ----------
        psd_low : float, default 0.5
            The lower bound of the cut. By default, it is 0.5 - data above 0.5
            are accepted as neutrons, whereas data below 0.5 are rejected as
            gammas.
        psd_upp : float, default 4.0
            The upper bound of the cut, to make sure no extreme outliers are
            included.
        perp_low : float, default -0.7
            The lower bound of the perpendicular axis.
        perp_upp : float, default 0.7
            The upper bound of the perpendicular axis.
        """
        psd = f'{on}_psd'
        ppsd = f'{on}_psd_perp'
        return ' && '.join([
            f'{psd} > {psd_low}',
            f'{psd} < {psd_upp}',
            f'{ppsd} > {perp_low}',
            f'{ppsd} < {perp_upp}',
        ])
    
class Spectrum:
    """A class for constructing neutron related spectra.

    Examples
    --------
    We first import the libraries needed.

    .. code-block:: python

        import os
        import ROOT
        from e15190.neutron_wall import spectra as nw_spec
        nw_spec.enable_implicit_multithreading() # optional
    
    To construct the :py:class:`Spectrum` object, we need to specify the paths
    to the ROOT files.

    .. code-block:: python

        paths = [
            os.path.expandvars(f'$DATABASE_DIR/root_files/run-{run:04d}.root')
            for run in range(4100, 4110 + 1)
        ]
        spec = nw_spec.Spectrum('B', paths)

    We can now look at, for example, the time-of-flight spectra:

    .. code-block:: python

        # apply cuts
        spec.filter_forward_array()
        spec.filter_charged_particles()
        spec.define_time_of_flight() # 'tof' added to rdf
        spec.define_nw_cut() # 'nw_cut' added to rdf
        spec.rdf = spec.rdf.Define('tof_cut', 'tof[nw_cut]')

        # plot
        canv = ROOT.TCanvas()
        h = spec.rdf.Histo1D(('h', 'No NW cut', 300, 0, 300), 'tof').GetValue()
        h_n = spec.rdf.Histo1D(('h_n', 'With NW cut', 300, 0, 300), 'tof_cut').GetValue()
        h.Draw('HIST')
        h_n.Draw('HIST SAME')
        canv.Draw()

    We can also look at the neutron wall hit pattern:

    .. code-block:: python

        spec.rdf = (spec.rdf # this is how you can write a long statement in multiple lines
            .Define('theta_cut', 'NWB_theta[nw_cut]')
            .Define('phi_cut', 'NWB_phi[nw_cut]')
        )
        canv = ROOT.TCanvas()
        h2d = spec.rdf.Histo1D(
            ('h2d', 'Hit pattern', 200, 25, 55, 200, -30, 30),
            'theta_cut', 'phi_cut',
        ).GetValue()
        h2d.Draw('COLZ')
        canv.Draw()

    """
    def __init__(self, AB : Literal['A', 'B'], paths, runs='infer', check_runlog=True, init_rdf=True):
        """A class for constructing neutron related spectra.

        Currently, only NWB is supported.

        Parameters
        ----------
        AB : 'A' or 'B'
            Neutron wall. Behavior for 'A' has not been developed yet.
        paths : list of str or list of Path
            Paths to the ROOT files.
        runs : 'infer' or list of int
            Run numbers. If 'infer', the run numbers are inferred from the
            paths. If list of int, the given run numbers are directly used
            without checking.
        check_runlog : bool
            To check the consistency of the runs. See
            :py:func:`check_runlog_consistency` for more details. When this is
            set to ``False``, any function that automatically modifies its
            behavior based on the runlog will not be working.
        init_rdf : bool
            To initialize the RDataFrame object.
        """
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.paths = list(map(str, paths))
        self.runs = self.infer_runs_from_paths() if runs == 'infer' else runs
        if check_runlog:
            self.check_runlog_consistency()
        if init_rdf:
            self.get_rdf()
    
    def infer_runs_from_paths(self, inplace=True):
        """Infer the run number for each path.

        Parameters
        ----------
        inplace : bool
            If ``True``, :py:attr:`runs` is updated inplace.
        
        Returns
        -------
        runs : list of int
            The inferred run numbers.
        """
        regex = re.compile(r'\d+')
        extract_run = lambda path: int(max(regex.findall(Path(path).stem), key=len))
        runs = [extract_run(path) for path in self.paths]
        if inplace:
            self.runs = runs
        return self.runs
    
    def check_runlog_consistency(self):
        """To check the consistencies of runs.

        This function queries the run info for each run in :py:attr:`runs` and
        check if they all share the same targets, beams, beam energies and
        shadow bar status. If not, an assertion error is raised. If the runs are
        consistent, then attributes like :py:attr:`target`, :py:attr:`beam`,
        :py:attr:`beam_energy` and :py:attr:`shadow_bar` are defined and added
        to the object.
        """
        first_run_info = Query.get_run_info(self.runs[0])
        for run in self.runs[1:]:
            run_info = Query.get_run_info(run)
            assert run_info['target'] == first_run_info['target']
            assert run_info['beam'] == first_run_info['beam']
            assert run_info['beam_energy'] == first_run_info['beam_energy']
            assert run_info['shadow_bar'] == first_run_info['shadow_bar']
        self.target = first_run_info['target']
        self.beam = first_run_info['beam']
        self.beam_energy = first_run_info['beam_energy']
        self.shadow_bar = first_run_info['shadow_bar']

    def get_rdf(self, tree_name=None):
        """Construct the RDataFrame object.

        The following attributes will be added to the object:
            - :py:attr:`tree_name`, the name of the tree.
            - :py:attr:`rdf_raw`, the RDataFrame object, but it will not be modified.
            - :py:attr:`rdf`, the RDataFrame object that will be modified.

        Parameters
        ----------
        tree_name : str or None, default None
            The name of the tree to be used. If ``None``, the function attempts
            to infer the tree name from the ROOT files.
        
        Returns
        -------
        rdf : RDataFrame
            The RDataFrame object.
        """
        if tree_name is None:
            tree_name = rt6.infer_tree_name(self.paths)
        self.tree_name = tree_name
        self.rdf_raw = ROOT.RDataFrame(self.tree_name, self.paths)
        self.rdf = self.rdf_raw
        return self.rdf
    
    @property
    def columns(self):
        return [str(col) for col in self.rdf.GetColumnNames()]

    def _inplace_rdf_result(self, rdf_result, inplace):
        if inplace:
            self.rdf = rdf_result
        return rdf_result

    def filter_forward_array(self, inplace=True):
        """Select events with forward array multiplicity greater than 0.
        """
        rdf_result = self.rdf.Filter('FA_multi > 0')
        return self._inplace_rdf_result(rdf_result, inplace)
    
    def filter_charged_particles(self, inplace=True):
        """Select events with charged particle multiplicity equals to 0."""
        rdf_result = self.rdf.Filter('VW_multi == 0')
        return self._inplace_rdf_result(rdf_result, inplace)
    
    def define_nw_cut(
        self,
        light_threshold=3.0, # MeVee
        psd_range=(0.5, 4.0),
        shadow_bar : Literal['auto', True, False] ='auto',
        psd_perp_range=(-0.7, 0.7),
        extra_cuts=None,
        inplace=True,
    ):
        """Cuts that are applied to the Neutron Wall.

        The cut will be named ``'nw_cut'`` in :py:attr:`rdf`.

        Implementation detail: ``Filter()`` cannot be used because the neutron
        wall observables are zigzag arrays. Instead, we define arrays of zigzag
        booleans, and use them to mask the arrays. This mask is named
        ``'nw_cut'``.  Users can use ``'nw_cut'`` to select hits directly on
        :py:attr:`rdf`.

        Parameters
        ----------
        light_threshold : float, default 3.0
            The light output threshold in MeVee. Only hits with light output
            above this threshold are kept.
        psd_range : (float, float), default (0.5, 4.0)
            The range of the PSD. The gamma-neutron separation line is at ``0.5``.
        shadow_bar : 'auto' or bool, default 'auto'
            Whether to cut out the shadowed regions due to shadow bars. If
            ``'auto'``, shadow bar cut is applied whenever the shadow bar is
            present for the runs. If ``True``, shadow bar cut is applied always.
            If ``False``, shadow bar cut is never applied.
        psd_perp_range : (float, float), default (-0.7, 0.7)
            The range of the second axis of the two-dimensional PSD.
        extra_cuts : list of str or None, default None
            Additional cuts to be applied. The cuts will be joined with the
            operator AND, i.e. ``&&`` in ROOT. Any syntax that is supported by
            RDataFrame should work here.
        inplace : bool, default True
            If ``True``, :py:attr:`rdf` is updated inplace.
        
        Returns
        -------
        rdf : RDataFrame
            The modified RDataFrame object.
        """
        conditions = [
            Cuts.edge(),
            f'NW{self.AB}_light_GM > {light_threshold}',
            Cuts.psd(
                psd_low=psd_range[0], psd_high=psd_range[1],
                psd_perp_low=psd_perp_range[0], psd_perp_high=psd_perp_range[1],
            ),
        ]
        if shadow_bar == 'auto':
            shadow_bar = (self.shadow_bar == 'in')
        if shadow_bar:
            conditions.append(Cuts.shadow_bars())
        if extra_cuts is not None:
            conditions.extend(extra_cuts)
        rdf_result = self.rdf.Define('nw_cut', ' && '.join(conditions))
        return self._inplace_rdf_result(rdf_result, inplace)

    def define_time_of_flight(self, inplace=True):
        """Define the time-of-flight in nanosecond.

        The time-of-flight will be named ``'tof'`` in :py:attr:`rdf`.
        """
        rdf_result = self.rdf.Define('tof', f'NW{self.AB}_time - FA_time_min')
        return self._inplace_rdf_result(rdf_result, inplace)
    
    def define_kinergy(self, inplace=True):
        """Calculate the kinetic energies of neutrons.

        The following quantities will be added to :py:attr:`rdf`:
            - ``'beta'``, the speed divided by the speed of light.
            - ``'energy'``, the relativistic energy (including rest mass) in MeV.
            - ``'kinergy'``, the relativistic kinetic energy (excluding rest mass) in MeV.
        
        Calculation of energy requires the knowledge of particle mass. Since we
        are interested in neutrons, we use the mass of the neutron for all hits.
        In other words, if you did not remove gammas, then you will get an
        enormously large energy, which is incorrect.
        """
        if 'tof' not in self.columns:
            self.define_time_of_flight()
        rdf_result = (self.rdf
            .Define('beta', f'NW{self.AB}_distance / tof / {constants.c.to("cm/ns").value}')
            .Define('energy', f'{ame.mass("n")} / sqrt(1 - beta * beta)')
            .Define('kinergy', f'energy - {ame.mass("n")}')
        )
        return self._inplace_rdf_result(rdf_result, inplace)

    def define_transverse_momentum(self, inplace=True):
        """Define the transverse momentum in MeV/c.

        The following quantities will be added to :py:attr:`rdf`:
            - ``'momentum_mag'``, the magnitude of the momentum in MeV/c.
            - ``'momentum_transv'``, the transverse momentum :math:`p_\mathrm{T}` in MeV/c.
        """
        if 'kinergy' not in self.columns:
            self.define_kinergy()
        rdf_result = (self.rdf
            .Define('momentum_mag', f'sqrt(kinergy * (kinergy + 2 * {ame.mass("n")}))')
            .Define('momentum_transv', f'momentum_mag * sin(TMath::DegToRad() * NW{self.AB}_theta)')
        )
        return self._inplace_rdf_result(rdf_result, inplace)
    
    def define_lab_rapidity(self, norm=True, inplace=True):
        """Define the rapidity in the lab frame.

        The following quantities will be added to :py:attr:`rdf`:
            - ``'momentum_z'``, the z component of the momentum in MeV/c.
            - ``'lab_rapidity'``, the rapidity in the lab frame.
        
        Parameters
        ----------
        norm : bool, default True
            Whether to normalize the momentum to the beam rapidity in the lab frame.
        inplace : bool, default True
            If ``True``, :py:attr:`rdf` is updated inplace.
        """
        if 'energy' not in self.columns:
            self.define_kinergy()
        if 'momentum_mag' not in self.columns:
            self.define_transverse_momentum()

        lab_rapidity_expr = f'0.5 * log((energy + momentum_z) / (energy - momentum_z))'
        if norm:
            normalization = physics.BeamTargetReaction(self.beam, self.target, self.beam_energy).beam_lab_rapidity
            lab_rapidity_expr += f' / {normalization}'

        rdf_result = (self.rdf
            .Define('momentum_z', f'momentum_mag * cos(TMath::DegToRad() * NW{self.AB}_theta)')
            .Define('lab_rapidity', lab_rapidity_expr)
        )
        return self._inplace_rdf_result(rdf_result, inplace)

    def geometry_efficiency(
        self,
        shadowed_bars='auto',
        cut_edges_bars='all',
        skip_bars=(0, ),
        custom_cuts=None,
        norm=True,
    ):
        """Returns a function that describes the geometry efficiency.

        This is a function of :math:`\theta` in radian, the polar angle with
        respect to the beam direction in lab frame. The function returns
        :math:`\delta\phi` or :math:`\delta\phi/2\pi` (when normalized), i.e.
        the azimuthal coverage of the detector at :math:`\theta`.

        Parameters
        ----------
        shadowed_bars : 'auto' or bool or list of int
            If 'auto', the shadow bar cut is automatically applied if the shadow
            bars were present in the run. If ``True``, the shadow bar cut is
            applied to NWB bars 7, 8, 9, 15, 16, 17; if ``False``, no shadow bar
            cut is applied. If a list of int, the shadow bar cut is applied to
            the specified bars.
        cut_edges_bars : 'all' or list of int
            If 'all', the cut edges cut is applied to all bars. If a list of int,
            the edge cut will only be applied to the specified bars.
        skip_bars : tuple of int, default (0, )
            The bars to skip in the geometry efficiency. In the experiment, NWB-bar00
            is the bottommost bar that was blocked by the ground.
        custom_cuts : dict of int or list of int, default None
            When custom cuts are applied, all the other cuts are ignored, so
            users should make sure the cuts are complete, e.g. edge cut, if
            desired, has to be specified manually. The only cut variable being
            supported is ``'x'``.
        norm : bool, default True
            Whether to normalize the efficiency by :math:`2\pi`.
        """
        nw = nwgeo.Wall(self.AB)
        kw = copy(locals())
        if kw['shadowed_bars'] == 'auto':
            kw['shadowed_bars'] = self.shadow_bar
        return nw.geometry_efficiency(**kw)
    
    def detection_efficiency(self, path=None):
        """Returns the intrinsic or detection efficiency curve.

        This was previously given by SCINFUL-QMD, but now it is given by Geant4
        simulation or SCINFUL-PHITS.

        The efficiency curve is constructed using spline interpolation of the
        two-column data stored at ``path``.
    
        Parameters
        ----------
        path : str or Path, default None
            The path to the efficiency data. The file should be a two-column
            ASCII file, with headers ``'kinergy'`` and ``'efficiency'``. Columns
            are separated by spaces. The ``'kinergy'`` range should at least
            cover from 20 MeV to 200 MeV.
        
        Returns
        -------
        eff_curve : UnivariateSpline object
            A spline interpolation of the efficiency curve.
        """
        if path is None:
            path = '$DATABASE_DIR/neutron_wall/efficiency/efficiency.txt'
        path = os.path.expandvars(path)

        df = pd.read_csv(path, delim_whitespace=True)
        return UnivariateSpline(df.kinergy, df.efficiency, s=0)
