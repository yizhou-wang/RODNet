import os
import sys

from importlib import import_module


def load_configs_from_file(config_path):
    module_name = os.path.basename(config_path)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = os.path.dirname(config_path)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    return cfg_dict
