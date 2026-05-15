#!/usr/bin/env python3
"""
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
"""

import time
import os

import numpy as np
import yaml

from abacusnbody.hod.flamingo_hod import FlamingoHOD
from abacusnbody.hod.NFW import nfw_draw
from abacusnbody.hod.fitting.hod_fitting_with_paircounting_emcee import fit_HOD
from abacusnbody.hod.fitting.paircounting import get_paircounts

DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'


def main(path_config_filename):
    print("Doing HOD fitting...", flush=True)#########################################################################
    max_like_params = fit_HOD(path_config_filename=path_config_filename, save_chains=True)

if __name__ == '__main__':
    main('config/test_profiling.yaml')
