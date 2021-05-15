"""This module interacts with the atomic mass data.
By the time of writing this module, the latest available data is
Atomic Mass Evaluation in 2016 (AME2016). When future evaluation
is released, manual modification needs to be made by updating
`DataManager.ame_url`, and possibly `DataManager.read_in_data`,
if the format of the table has been changed.
"""
import collections
import pathlib
import re
import stat
import urllib.request
import warnings

from astropy import constants as const
from astropy import units as u
import pandas as pd

# attributes to be initialized in __init__.py
MODULE_DIR = None
PROJECT_DIR = None

class DataManager:
    """A class to interacts with the atomic masses.
    Many kinematic calculations require the knowledge of masses of atomic
    isotopes. This class allows user to automatically download the data
    sheet of Atomic Mass Evaluation to local machine, and perform some
    simple query tasks on the table.
    """
    def __init__(self, force_download=False, auto_read_in=True):
        """This creates a `DataManager` object.
        A local copy has to be present before any query task can be made.
        Hence this initializer downloads the data if local copy is not
        present yet, or the user has decided to turn on the `force_download`
        option.
        Parameters:
            force_download : bool *optional*
                If `True`, a new local copy will always be downloaded from
                the website (see class variable `ame_url`). If `False`, the
                program only downloads the data when no local copy is found.
                Default is `False`.
        """
        self.ame_url = 'https://www-nds.iaea.org/amdc/ame2016/mass16.txt'
        self.ame_loc_path = pathlib.Path(f'{PROJECT_DIR}/database/utilities/mass16.txt')

        # download from web if local copy is not found
        file_exists = self.ame_loc_path.is_file()
        if force_download or not file_exists:
            if file_exists: self.ame_loc_path.unlink()
            self.download()

        # other class attributes to be updated
        self.df = None # a `pandas.DataFrame for storing AME data
        self.units = None # units for every column in self.df
        self.Z_to_symb = None
        self.symb_to_Z = None

        if auto_read_in:
            self.read_in_data()

    def download(self, filepath=None, url=None):
        """Download AME data from the web.
        Parameters:
            filepath : str *optional*
                The local filepath that store the downloaded data. Default is
                `None`, which will then be set into class variable `AME_LOCAL_PATH`.
            url : str *optional*
                The url to the AME data. Default is `None`, which will then be
                set into class variable `ame_url`.
        """
        if filepath is None: filepath = self.ame_loc_path
        if url is None: url = self.ame_url

        # to pretend as a web browser; for some reason, DataManager.ame_url cannot be accessed otherwise
        headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as webpage:
            content = str(webpage.read(), 'utf-8')

        with open(filepath, 'w') as f:
            f.write(content)
        
        # set file to be read only for protection
        filepath.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    def read_in_data(self, filepath=None):
        """Updates class attribute `self.df` into formatted AME data.
        This function only reads from a local copy of AME data. Make sure the data
        has been downloaded from the web. Not all columns from AME data will be kept.
        The columns that will be returned are `Z`, `A`, `symb`, `mass_excess` and
        `mass_excess_err`. Other columns including isospin, binding energy (and its
        error) and mass in a.m.u (and its error) are discarded as they can all be
        calculated directly from mass excess.
        No unit conversion has been made. The AME provides mass excess in $\mathrm{keV/c^2}$.
        Nonetheless, units have been stored as class attribute `self.units` as python
        dictionary using `astropy.units`.
        Parameters:
            filepath : str *optional*
                The local filepath that store the downloaded data. Default is
                `None`, which will then be set into class variable `AME_LOCAL_PATH`.
        """
        if filepath is None: filepath = self.ame_loc_path

        # check if file can be opened
        try:
            with open(filepath, 'r') as f:
                content = f.readlines()
        except IOError:
            print(f'Fail to read in {filepath}')

        # crop to select only the table data in content
        first_line_of_data = -1
        start_token = '0'
        start_token_count = 0
        for line_i, line in enumerate(content):
            if line[0] == start_token:
                start_token_count += 1
            if start_token_count >= 3:
                first_line_of_data = line_i
                break
        content = content[line_i:]

        content = self.auto_column_splitter(content)
        
        # A lot of HARD-CODED codes from now on. They are written
        # according to the format of the table. The goal is to transform
        # `content` into a `pandas.DataFrame` with appropriate format.

        # construct `pandas.DataFrame` for columns of interest
        column_dict = {3: 'Z', 4: 'A', 5: 'symb',
                       7: 'mass_excess', 8: 'mass_excess_err'}
        content = [[row[i] for i in column_dict.keys()] for row in content]
        df = pd.DataFrame(content, columns=column_dict.values())

        # cast each column into appropriate variable type
        df['Z'] = df['Z'].astype(int)
        df['A'] = df['A'].astype(int)
        df['symb'] = df['symb'].str.extract(r'([A-Za-z]+)')
        df['mass_excess'] = df['mass_excess'].str.extract(r'([0-9.]+)').astype(float)
        df['mass_excess_err'] = df['mass_excess_err'].str.extract(r'([0-9.]+)').astype(float)

        # construct mapping from `Z` to `symb`
        Z_to_symb = dict(zip(df['Z'], df['symb']))

        # IUPAC announces official names for new elements in 2016
        # but AME2016 had not adopted the new names back then
        new_symb = {113: 'Nh', 115: 'Mc', 117: 'Ts', 118: 'Og'}
        old_to_new_symb = {Z_to_symb[k]: new_symb[k] for k in new_symb.keys()}
        df.replace({'symb': old_to_new_symb}, inplace=True)
        for k, v in new_symb.items():
            Z_to_symb[k] = v

        # construct mapping from `symb` to `Z`
        symb_to_Z = {v: k for k, v in Z_to_symb.items()}

        # update to class attributes
        df.set_index(['A', 'Z'], drop=False, inplace=True, verify_integrity=True)
        self.df = df
        self.units = {'Z': None, 'A': None, 'symb': None,
                      'mass_excess': u.keV,
                      'mass_excess_err': u.keV}
        self.Z_to_symb = Z_to_symb
        self.symb_to_Z = symb_to_Z

    @staticmethod
    def auto_column_splitter(content):
        """This function automatically separates the columns of `.txt` table.
    
        This function can separate the columns of `.txt` tables that use space
        characters as their delimiters.
        Parameters:
            content : list of str
                A list of strings that correspond to the content of a `.txt` table.
                Each element in the list, which is a string, corresponds to each row
                of the table. These strings can be either ended with the newline
                character or not. The row of column names or headers should not
                be included.
        Returns:
            splitted_content : a two-dimensional list of str
        Examples:
        ----------
        >>> from isotope_mass import DataManager
        >>> dm = DataManager()
        >>> content = ["Amy  168.5cm", "Bob  181.9cm", "Cici 157.3cm"]
        >>> dm.auto_column_splitter(content)
        [['Amy  ', '168.5cm'], ['Bob  ', '181.9cm'], ['Cici ', '157.3cm']]
        """
        min_line_length = min([len(line) for line in content])
        split_pos = []
        for c in range(1, min_line_length):
            is_col_splitter = True
            is_all_space = True
            for line in content:
                if line[c] != ' ':
                    is_all_space = False
                if line[c] != ' ' and line[c-1] != ' ':
                    is_col_splitter = False
                if not is_all_space and not is_col_splitter:
                    break
            if is_col_splitter and not is_all_space:
                split_pos.append(c)
        
        splitted_content = []
        split_pos = [0] + split_pos
        for line in content:
            line = line.strip('\n')
            splitted_line = []
            for c0, c1 in zip(split_pos[:-1], split_pos[1:]):
                splitted_line.append(line[c0:c1])
            splitted_line.append(line[c1:])
            splitted_content.append(splitted_line)
        
        return splitted_content



# attributes to be initialized in __init__.py
_data_manager = None

def get_A_Z(notation, simple_tuple=False):
    """Converts mass-number annotated isotope expression into A and Z.
    Examples:
    ----------
    >>> import isotope_mass as isom
    >>> isom.get_A_Z('ca40')
    (40, 20)
    """
    global _data_manager
    symb_to_Z = _data_manager.symb_to_Z

    common_shorthands = {'n': (1, 0),
                         'p': (1, 1),
                         'd': (2, 1),
                         't': (3, 1),
                        }
    if notation in common_shorthands:
        A, Z = common_shorthands[notation]
    else:
        expected_regex = re.compile(r'([A-Za-z][a-z]?\d{1,3}|\d{1,3}[A-Za-z][a-z]?)')
        matches = expected_regex.findall(notation)
        if (len(matches) != 1):
            raise ValueError(f'notation "{notation}" has unexpected format')
        match = matches[0]
        digit_part = ''.join(filter(str.isdigit, match))
        alpha_part = ''.join(filter(str.isalpha, match))

        A = int(digit_part)
        symb = alpha_part.capitalize()
        if symb not in symb_to_Z.keys():
            raise ValueError(f'chemical symbol "{symb}" is unidentified')
        Z = symb_to_Z[symb]

    if simple_tuple:
        return (A, Z)
    else:
        isotope = collections.namedtuple('Isotope', ['A', 'Z'])
        return isotope(A=A, Z=Z)

def mass(argv, unitless=True, not_found_okay=False, not_found_warning=True):
    """Get mass of isotope.
    """
    global _data_manager
    df = _data_manager.df
    df_units = _data_manager.units

    if isinstance(argv, str):
        tmp = get_A_Z(argv)
        A, Z = tmp.A, tmp.Z
    elif isinstance(argv, tuple) and len(argv) == 2:
        A, Z = argv
    elif isinstance(argv, dict) and set(argv.keys()) == set(['A', 'Z']):
        A, Z = argv['A'], argv['Z']

    found = True
    if (A, Z) not in df.index:
        found = False
        if not_found_okay:
            if not_found_warning:
                warnings.warn(f'(A, Z) = ({A}, {Z}) not found. Assuming zero mass excess.')
        else:
            raise ValueError(f'"(A, Z) = ({A}, {Z})" not found')

    amu = const.u * const.c**2
    if found:
        mass_excess = u.Quantity(df.loc[(A, Z)]['mass_excess'], df_units['mass_excess'])
    else:
        mass_excess = 0.0
    mass = (A * amu + mass_excess).to('MeV')
    return mass.value if unitless else mass
