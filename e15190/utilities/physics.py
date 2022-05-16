from astropy import units
import numpy as np
import pandas as pd

from e15190.utilities import atomic_mass_evaluation as ame

def gamma(beta):
    return 1 / np.sqrt(1 - beta**2)

def energy(mass, beta):
    return gamma(beta) * mass

def kinergy(mass, beta):
    return (gamma(beta) - 1) * mass

def momentum(mass, beta):
    return gamma(beta) * mass * beta

def kinergy_to_momentum(mass, kinergy):
    return np.sqrt((kinergy + mass)**2 - mass**2)

def energy_to_momentum(mass, energy):
    return np.sqrt(energy**2 - mass**2)

def rapidity(energy, momentum_z):
    return 0.5 * np.log((energy + momentum_z) / (energy - momentum_z))

class LorentzVector:
    def __init__(self, x, y, z, t):
        def convert_scalar(val):
            if isinstance(val, (int, float)):
                val = np.array([val])
            return val
        self.x, self.y, self.z, self.t = map(convert_scalar, (x, y, z, t))

    @property
    def df(self):
        return pd.DataFrame({
            'x': self.x,
            'y': self.y,
            'z': self.z,
            't': self.t,
        })
    
    def boost(self, bx, by, bz):
        b2 = bx**2 + by**2 + bz**2
        gamma_ = 1 / np.sqrt(1 - b2)
        bp = bx * self.x + by * self.y + bz * self.z
        gamma2 = (gamma_ - 1) / b2 if b2 > 0 else 0

        x = self.x + gamma2 * bp * bx + gamma_ * bx * self.t
        y = self.y + gamma2 * bp * by + gamma_ * by * self.t
        z = self.z + gamma2 * bp * bz + gamma_ * bz * self.t
        t = gamma_ * (self.t + bp)

        return LorentzVector(x, y, z, t)

class BeamTargetReaction:
    ene_unit = 'MeV'

    def __init__(self, beam, targ, beam_kinergy, **kwargs):
        self.beam_A, self.beam_Z = ame.get_A_Z(beam, simple_tuple=True)
        self.targ_A, self.targ_Z = ame.get_A_Z(targ, simple_tuple=True)
        if isinstance(beam_kinergy, units.Quantity):
            self.beam_kinergy = beam_kinergy.to(self.ene_unit).value
        else:
            self.beam_kinergy = beam_kinergy # assume to be the correct unit already

    @property
    def com_beta(self):
        """Returns the beta (v/c) of C.O.M. of beam-target w.r.t. the lab.
        """
        beam_mass = ame.mass((self.beam_A, self.beam_Z))
        targ_mass = ame.mass((self.targ_A, self.targ_Z))

        beam_ene = self.beam_kinergy * (beam_mass / ame.amu.to(self.ene_unit).value) + beam_mass
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
        beam_ene = self.beam_kinergy * (beam_mass / ame.amu.to(self.ene_unit).value) + beam_mass
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
        beam_ene = self.beam_kinergy * (beam_mass / ame.amu.to(self.ene_unit).value) + beam_mass
        beam_mom = (beam_ene**2 - beam_mass**2)**0.5
        return 0.5 * np.log((beam_ene + beam_mom) / (beam_ene - beam_mom))
