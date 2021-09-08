import numpy as np
import pandas as pd
import scipy.interpolate

from e15190 import PROJECT_DIR

class MicroballImpactParameterReader:
    database_dir = PROJECT_DIR / 'database/microball/impact_parameter'
    database = dict()
    b2multi_interpolation = dict()
    bhat2multi_interpolation = dict()

    @staticmethod
    def key(proj, targ, beam_energy):
        return f'{proj.capitalize()}{targ.capitalize()}E{beam_energy:.0f}'

    @classmethod
    def load(cls, proj, targ, beam_energy, force_reread=False):
        key = cls.key(proj, targ, beam_energy)
        if key not in cls.database or force_reread:
            filepath = cls.database_dir / f'{key}.dat'
            cls.database[key] = pd.read_csv(filepath, delim_whitespace=True)
        return cls.database[key]

    @classmethod
    def interpolate_b2multi(cls, proj, targ, beam_energy, force_redo=False):
        key = cls.key(proj, targ, beam_energy)
        if key not in cls.b2multi_interpolation or force_redo:
            df = cls.load(proj, targ, beam_energy)
            df = df.iloc[::-1]
            cls.b2multi_interpolation[key] = scipy.interpolate.PchipInterpolator(
                df['b'], df['multiplicity'], extrapolate=True,
            )
        return cls.b2multi_interpolation[key]
    
    @classmethod
    def interpolate_bhat2multi(cls, proj, targ, beam_energy, force_redo=False):
        key = cls.key(proj, targ, beam_energy)
        if key not in cls.bhat2multi_interpolation or force_redo:
            df = cls.load(proj, targ, beam_energy)
            df = df.iloc[::-1]
            cls.bhat2multi_interpolation[key] = scipy.interpolate.PchipInterpolator(
                df['bhat'], df['multiplicity'], extrapolate=True,
            )
        return cls.bhat2multi_interpolation[key]

_reader = MicroballImpactParameterReader()

def get_database(proj, targ, beam_energy):
    global _reader
    return _reader.load(proj, targ, beam_energy)

def get_impact_parameter(proj, targ, beam_energy, multiplicity):
    global _reader
    if not np.issubdtype(type(multiplicity), np.integer):
        raise ValueError(f'multiplicity must be an integer, not {type(multiplicity)}')
    df = _reader.load(proj, targ, beam_energy)
    subdf = df.query(f'multiplicity == {multiplicity}')
    return tuple(subdf[['b', 'b_err']].iloc[0])

def get_b(proj, targ, beam_energy, multiplicity):
    return get_impact_parameter(proj, targ, beam_energy, multiplicity)

def get_normalized_impact_parameter(proj, targ, beam_energy, multiplicity):
    global _reader
    if not np.issubdtype(type(multiplicity), np.integer):
        raise ValueError(f'multiplicity must be an integer, not {type(multiplicity)}')
    df = _reader.load(proj, targ, beam_energy)
    subdf = df.query(f'multiplicity == {multiplicity}')
    return tuple(subdf[['bhat', 'bhat_err']].iloc[0])

def get_bhat(proj, targ, beam_energy, multiplicity):
    return get_normalized_impact_parameter(proj, targ, beam_energy, multiplicity)

def get_multiplicity_range(proj, targ, beam_energy, b_range=None, bhat_range=None):
    global _reader
    df = _reader.load(proj, targ, beam_energy)
    if b_range is not None:
        interpolation = _reader.interpolate_b2multi(proj, targ, beam_energy)
        in_range = np.clip(b_range, df.b.min(), df.b.max())
        result = interpolation(in_range)
    else:
        interpolation = _reader.interpolate_bhat2multi(proj, targ, beam_energy)
        in_range = np.clip(bhat_range, df.bhat.min(), df.bhat.max())
        result = interpolation(in_range)
    return result
