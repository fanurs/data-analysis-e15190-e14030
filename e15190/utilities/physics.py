import numpy as np
import pandas as pd

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

def rapidity(energy, momentum):
    return 0.5 * np.log((energy + momentum) / (energy - momentum))

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
