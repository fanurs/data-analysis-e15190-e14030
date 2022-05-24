"""A list of convenient functions to interact with PyROOT6.

A version of ROOT 6.20 or above is assumed.
"""
import distutils.version
import pathlib
import string
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    import ROOT
except ImportError:
    print('- Could not detect ROOT package installed in current Python environment.')
    print('- Trying to check if $ROOTPATH is defined in shell...')
    root_path = subprocess.check_output('which root', shell=True)
    root_path = str(root_path, encoding='utf-8').strip()
    root_path = pathlib.Path(root_path).parent.parent
    root_path = pathlib.Path(root_path, 'lib')
    sys.path.append(str(root_path))
    import ROOT

"""
ROOT.RDataFrame is heavily used in this project for data analysis. It allows the implementation of
multi-threading that can speed up the analysis significantly. RDataFrame was first introduced in
ROOT-v6.14.00 (formerly known as ROOT::Experimental::TDataFrame). Many new improved features had
been added since then, and the suggested minimum version for running this script is ROOT-v6.18.00.
"""
ROOT_MIN_VERSION = '6.18'
Version = distutils.version.StrictVersion
if Version(ROOT.__version__.replace('/', '.')) < Version(ROOT_MIN_VERSION):
    raise Exception(f'- Detects ROOT-v{ROOT.__version__}, requires at least {ROOT_MIN_VERSION}.')

class RandomName:
    def __init__(self):

        # every instance in a ROOT session must a unique name
        # so we need to check for used names to avoid repetition
        self.used_names = []

        # a list of characters used to generate random names
        self.characters = string.ascii_letters # English alphabets, lowercase and uppercase
        self.characters += string.digits
        self.characters = list(self.characters)

        # no. of characters used to create a random name
        self.n_char = 5

    def __call__(self):
        found = False
        n_trials = int(1e3)
        for _ in range(n_trials):
            # first character should be an English alphabet
            name = np.random.choice(list(string.ascii_letters))

            # the rest can be anything
            name += ''.join(np.random.choice(self.characters, size=self.n_char-1))

            if name not in self.used_names:
                self.used_names.append(name)
                found = True
                break

        # just in case super unlucky
        if not found:
            raise Exception(f'Fail to find a new unique name after {n_trials}.')

        return name

    def clear(self):
        self.used_names.clear()
random_name = RandomName()

class HistogramConversion:
    def __init__(self):
        pass

    def __call__(self, histo, *args, **kwargs):
        return self.histo_to_dframe(histo, *args, **kwargs)

    def histo_to_dframe(self, histo, *args, **kwargs):
        if isinstance(histo, ROOT.TH2):
            func = self._histo2d_to_dframe
        elif isinstance(histo, ROOT.TH1):
            func = self._histo1d_to_dframe
        else:
            raise TypeError(f'histo must be either ROOT.TH1 or ROOT.TH2')

        return func(histo, *args, **kwargs)

    def _histo1d_to_dframe(self, histo, xname='x', yname='y'):
        df = dict()
        columns = [xname, yname, f'{yname}err']
        getters = [histo.GetXaxis().GetBinCenter, histo.GetBinContent, histo.GetBinError]
        for col, getter in zip(columns, getters):
            df[col] = [getter(b) for b in range(1, histo.GetNbinsX() + 1)]
        df = pd.DataFrame(df)

        # calculate fractional errors
        mask = (df[yname] != 0.0)
        df[f'{yname}ferr'] = np.where(
            mask,
            np.abs(df[f'{yname}err'] / df[yname]),
            0.0
        )

        return df
    
    def _histo2d_to_dframe(self, histo, xname='x', yname='y', zname='z', keep_zeros=True):
        x = np.array([histo.GetXaxis().GetBinCenter(b) for b in range(1, histo.GetNbinsX() + 1)])
        y = np.array([histo.GetYaxis().GetBinCenter(b) for b in range(1, histo.GetNbinsY() + 1)])
        
        content = np.array(histo)
        content = content.reshape(len(x) + 2, len(y) + 2, order='F')
        content = content[1:-1, 1:-1]
        
        error = np.array([histo.GetBinError(b) for b in range((len(x) + 2) * (len(y) + 2))])
        error = error.reshape(len(x) + 2, len(y) + 2, order='F')
        error = error[1:-1, 1:-1]
        
        xx, yy = np.meshgrid(x, y, indexing='ij')
        df = pd.DataFrame({
            xname: xx.flatten(),
            yname: yy.flatten(),
            zname: content.flatten(),
            f'{zname}err': error.flatten(),
        })
        mask = (df[zname] != 0.0)
        df[f'{zname}ferr'] = np.where(
            mask,
            np.abs(df[f'{zname}err'] / df[zname]),
            0.0
        )
        return df if keep_zeros else df.query(f'{zname} != 0.0').reset_index(drop=True)
histo_conversion = HistogramConversion()

class TFile:
    def __init__(self, path, mode='READ'):
        self.path = pathlib.Path(path)
        self.mode = mode.upper()

    def __enter__(self):
        self.file = ROOT.TFile(str(self.path), self.mode)
        return self.file

    def __exit__(self, *args):
        self.file.Close()

class TChain(ROOT.TChain):
    def __init__(self, paths, tree_name):
        self.paths = paths
        self.tree_name = tree_name
        super().__init__(tree_name)
        for path in self.paths:
            super().Add(path)

def get_all_branches(path, tree):
    with TFile(path) as file:
        tr = file.Get(tree)
        def get_branches(obj):
            result = []
            for br in obj.GetListOfBranches():
                if isinstance(br, ROOT.TBranchElement):
                    result.extend(get_branches(br))
                result.append(br.GetName())
            return result
        return get_branches(tr)

def infer_tree_name(path):
    """
    Parameters
    ----------
    path : str or pathlib.Path or iterables
        Path(s) to the ROOT file(s) of interest.
    """
    if not isinstance(path, (str, pathlib.Path)):
        names = list(set([infer_tree_name(p) for p in path]))
        if len(names) == 1:
            return names[0]
        if len(names) == 0:
            raise ValueError(f'No tree name found in "{path}".')
        raise ValueError(f'Paths contain different tree names: {names}')

    tree_names = []
    with TFile(path) as file:
        names = set([k.GetName() for k in file.GetListOfKeys()])
        for name in names:
            obj = file.Get(name)
            if isinstance(obj, ROOT.TTree):
                tree_names.append(obj.GetName())
    if len(tree_names) == 1:
        return tree_names[0]
    raise ValueError(f'Cannot infer tree name from\n"{path}".')
