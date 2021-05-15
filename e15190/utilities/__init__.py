import pathlib

from . import atomic_mass_evaluation as ame

ame.MODULE_DIR = pathlib.Path(__file__).parent.resolve()
ame.PROJECT_DIR = ame.MODULE_DIR.parent.parent.resolve()
ame._data_manager = ame.DataManager()