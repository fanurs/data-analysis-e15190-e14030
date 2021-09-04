import pytest

from astropy import constants

from e15190.utilities import atomic_mass_evaluation as ame

class TestGlobalFunctions:
    amu = (constants.u * constants.c**2).to('MeV').value

    def test_get_A_Z(self):
        # "Symb00"
        assert ame.get_A_Z('Ca40') == (40, 20)
        assert ame.get_A_Z('Ca48') == (48, 20)
        assert ame.get_A_Z('Ni58') == (58, 28)
        assert ame.get_A_Z('Ni64') == (64, 28)
        assert ame.get_A_Z('Sn112') == (112, 50)
        assert ame.get_A_Z('Sn124') == (124, 50)

        # "symb00"
        assert ame.get_A_Z('ca40') == (40, 20)
        assert ame.get_A_Z('ca48') == (48, 20)
        assert ame.get_A_Z('ni58') == (58, 28)
        assert ame.get_A_Z('ni64') == (64, 28)
        assert ame.get_A_Z('sn112') == (112, 50)
        assert ame.get_A_Z('sn124') == (124, 50)

        # "00Symb"
        assert ame.get_A_Z('40Ca') == (40, 20)
        assert ame.get_A_Z('48Ca') == (48, 20)
        assert ame.get_A_Z('58Ni') == (58, 28)
        assert ame.get_A_Z('64Ni') == (64, 28)
        assert ame.get_A_Z('112Sn') == (112, 50)
        assert ame.get_A_Z('124Sn') == (124, 50)

        # "00symb"
        assert ame.get_A_Z('40ca') == (40, 20)
        assert ame.get_A_Z('48ca') == (48, 20)
        assert ame.get_A_Z('58ni') == (58, 28)
        assert ame.get_A_Z('64ni') == (64, 28)
        assert ame.get_A_Z('112sn') == (112, 50)
        assert ame.get_A_Z('124sn') == (124, 50)

    def test_mass(self):
        assert ame.mass('Ca40') / self.amu == pytest.approx(39.962590866)
        assert ame.mass('Ca48') / self.amu == pytest.approx(47.95252290)
        assert ame.mass('Ni58') / self.amu == pytest.approx(57.9353429)
        assert ame.mass('Ni64') / self.amu == pytest.approx(63.9279660)
        assert ame.mass('Sn112') / self.amu == pytest.approx(111.904818)
        assert ame.mass('Sn124') / self.amu == pytest.approx(123.9052739)
