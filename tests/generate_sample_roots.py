#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import awkward as ak
import uproot

import e15190 # don't remove

class Sample1:
    JSON_PATH = '_samples/json_files/sample1.json'
    ROOT_PATH = '_samples/root_files/sample1.root'

    @classmethod
    def get_json_path(cls, json_path=None):
        if json_path is None:
            json_path = Path(__file__).parent / cls.JSON_PATH
        return Path(json_path).resolve()

    @classmethod
    def get_root_path(cls, root_path=None):
        if root_path is None:
            root_path = Path(__file__).parent / cls.ROOT_PATH
        return Path(root_path).resolve()

    @classmethod
    def json_exists(cls, json_path=None):
        return cls.get_json_path(json_path).is_file()

    @classmethod
    def root_exists(cls, root_path=None):
        return cls.get_root_path(root_path).is_file()

    @classmethod
    def generate(cls, json_path=None, root_path=None):
        json_path = cls.get_json_path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'r') as file:
            events = json.load(file)['events']
        
        root_path = cls.get_root_path(root_path)
        root_path.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(str(root_path)) as file:
            file.mktree(
                'tree',
                {
                    'i_evt': 'int32',
                    'multi_0': 'int32',
                    'x_0': 'var * float64',
                    'multi_1': 'int32',
                    'x_1': 'var * float64',
                    'y_1': 'var * float64',
                }
            )
            for event in events:
                file['tree'].extend({
                    'i_evt': np.array([event['i_evt']]),
                    'multi_0': np.array([event['multi_0']]),
                    'x_0': ak.Array([event['x_0']]),
                    'multi_1': np.array([event['multi_1']]),
                    'x_1': ak.Array([event['x_1']]),
                    'y_1': ak.Array([event['y_1']]),
                })

if __name__ == '__main__':
    Sample1.generate()