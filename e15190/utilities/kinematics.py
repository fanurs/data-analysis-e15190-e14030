from astropy import constants as const
from astropy import units as u
import numpy as np

from . import atomic_mass_evaluation as ame

ene_unit = 'GeV'

class AdditionalConstants:
    _mass_n = (const.m_n * const.c**2).to(ene_unit).value
    _mass_p = (const.m_p * const.c**2).to(ene_unit).value
    _amu = (const.u * const.c**2).to(ene_unit).value

    def __init__(self, module):
        attr_keys = [attr for attr in dir(self) if not attr.startswith('__')]
        for key in attr_keys:
            if not hasattr(module, key):
                setattr(module, key, getattr(self, key))
AdditionalConstants(const)

class BeamTargetReaction:
    def __init__(self, beam, targ, beam_kinergy, **kwargs):
        self.beam_A, self.beam_Z = ame.get_A_Z(beam, simple_tuple=True)
        self.targ_A, self.targ_Z = ame.get_A_Z(targ, simple_tuple=True)
        if isinstance(beam_kinergy, u.Quantity):
            self.beam_kinergy = beam_kinergy.to(ene_unit).value
        else:
            self.beam_kinergy = beam_kinergy # assume to be GeV

    @property
    def com_beta(self):
        """Returns the beta (v/c) of C.O.M. of beam-target w.r.t. the lab.
        """
        beam_mass = ame.mass((self.beam_A, self.beam_Z))
        targ_mass = ame.mass((self.targ_A, self.targ_Z))

        beam_ene = self.beam_kinergy * (beam_mass / const._amu) + beam_mass
        targ_ene = targ_mass # target is at rest

        beamtarg_ene = beam_ene + targ_ene
        beamtarg_mom = (beam_ene**2 - beam_mass**2)**0.5
        return beamtarg_mom / beamtarg_ene
    
    @property
    def beam_com_rapidity(self):
        """Returns the (experimentally modified) rapidity of beam w.r.t. the C.O.M.
        """
        beam_mass = ame.mass((self.beam_A, self.beam_Z))

        # in lab frame
        beam_ene = self.beam_kinergy * (beam_mass / const._amu) + beam_mass
        beam_mom = (beam_ene**2 - beam_mass**2)**0.5

        # in com frame; apply Lorentz transform
        beta = self.com_beta
        gamma = 1.0 / (1 - beta**2)**0.5
        beam_ene, beam_mom = (
            gamma * (beam_ene - beta * beam_mom),
            gamma * (-beta * beam_ene + beam_mom)
        )
        return 0.5 * np.log((beam_ene + beam_mom) / (beam_ene - beam_mom))

    @property
    def beam_lab_rapidity(self):
        """Returns the (experimentally modified) rapidity of beam w.r.t. the lab.
        """
        beam_mass = ame.mass((self.beam_A, self.beam_Z))
        beam_ene = self.beam_kinergy * (beam_mass / const._amu) + beam_mass
        beam_mom = (beam_ene**2 - beam_mass**2)**0.5
        return 0.5 * np.log((beam_ene + beam_mom) / (beam_ene - beam_mom))