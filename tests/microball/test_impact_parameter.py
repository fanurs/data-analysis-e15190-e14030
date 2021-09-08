import pytest

import itertools

import numpy as np
import pandas as pd

from e15190.microball import impact_parameter

@pytest.fixture
def reaction_systems():
    projectiles = ['Ca40', 'Ca48']
    targets = ['Ni58', 'Ni64', 'Sn112', 'Sn124']
    beam_energies = [56, 140]
    result = []
    for proj, targ, beam_energy in itertools.product(projectiles, targets, beam_energies):
        result.append(dict(proj=proj, targ=targ, beam_energy=beam_energy))
    return result

def test_get_database(reaction_systems):
    for reac_sys in reaction_systems:
        df = impact_parameter.get_database(**reac_sys)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 5

def test_get_impact_parameter(reaction_systems):
    multiplicities = [8, 10, 12]
    for reac_sys in reaction_systems:
        b_values = {
            multi: impact_parameter.get_impact_parameter(**reac_sys, multiplicity=multi)
            for multi in multiplicities
        }
        assert b_values[8][0] > b_values[10][0] > b_values[12][0]

def test_get_normalized_impact_parameter(reaction_systems):
    multiplicities = [8, 10, 12]
    for reac_sys in reaction_systems:
        bhat_values = {
            multi: impact_parameter.get_normalized_impact_parameter(**reac_sys, multiplicity=multi)
            for multi in multiplicities
        }
        assert bhat_values[8][0] > bhat_values[10][0] > bhat_values[12][0]

def test_get_multiplicity_range(reaction_systems):
    m_pair = np.array([9, 11])
    for reac_sys in reaction_systems:
        # interpolation on impact parameter b
        b_m0 = impact_parameter.get_impact_parameter(**reac_sys, multiplicity=m_pair[0])[0]
        b_m1 = impact_parameter.get_impact_parameter(**reac_sys, multiplicity=m_pair[1])[0]
        m_range = impact_parameter.get_multiplicity_range(**reac_sys, b_range=[b_m0, b_m1])
        assert np.allclose(m_range, m_pair, atol=1e-6)

        # interpolation on normalized impact parameter bhat
        bhat_m0 = impact_parameter.get_normalized_impact_parameter(**reac_sys, multiplicity=m_pair[0])[0]
        bhat_m1 = impact_parameter.get_normalized_impact_parameter(**reac_sys, multiplicity=m_pair[1])[0]
        m_range = impact_parameter.get_multiplicity_range(**reac_sys, bhat_range=[bhat_m0, bhat_m1])
        assert np.allclose(m_range, m_pair, atol=1e-6)

        # check numpy.clip
        m_range = impact_parameter.get_multiplicity_range(**reac_sys, b_range=[100.0, 200.0])
        assert m_range[0] == pytest.approx(m_range[1])
        m_range = impact_parameter.get_multiplicity_range(**reac_sys, b_range=[-2, -1])
        assert m_range[0] == pytest.approx(m_range[1])
