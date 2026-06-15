#!/usr/bin/env python3
"""
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
"""

import argparse
import time
import os

import numpy as np
import yaml

from abacusnbody.hod.flamingo_hod import FlamingoHOD
from abacusnbody.hod.NFW import nfw_draw
from abacusutils.abacusnbody.hod.fitting.hod_fitting_with_paircounting_stoch_sample import fit_HOD
from abacusnbody.hod.fitting.paircounting import get_paircounts

DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'


def main(path_config_filename):
    # load the yaml parameters
    config = yaml.safe_load(open(path_config_filename))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    Paths = config["Paths"]
    Labels = config["Labels"]
    sim_label = Labels["sim_label"]
    Params = config["Params"]
    Misc = config["Misc"]
    seed = Misc["random_seed"]
    fitting_params = config["fitting_params"]
    paircount_path = fitting_params["paircounts_save_path"]

    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(
        bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1
    )
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']

    print("Making new FlamingoHOD object", flush=True)###############################################################
    # create a new FlamingoHOD object
    newBall = FlamingoHOD(path_config_filename)

    paircount_labels = ["cencen", "censat", "satsat", "satsat_onehalo", "cencen_ELGauto", "censat_ELGauto", "satsat_ELGauto", "satsat_onehalo_ELGauto",
                        "cencen_ELGcross", "censat_ELGcross", "satsat_ELGcross", "satsat_onehalo_ELGcross"]
    all_paircounts_exist = True
    for label in paircount_labels:
        if not os.path.exists(paircount_path + sim_label + f"/{label}.npy"):
            all_paircounts_exist = False
    if not all_paircounts_exist:
        print("Paircounts missing; computing them now", flush=True)
        print("Making tracer mock...", flush=True)
        max_nfw = 40
        NFW_draw = nfw_draw(10000, max_nfw, seed)
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=16, verbose=True, tabulation_mock=True
        )
        print("Paircounting...")
        get_paircounts(path_config_filename=path_config_filename, tracer_mock = mock_dict, Nthread=16, save=True, verbose=True)

    print("Doing HOD fitting...", flush=True)#########################################################################
    max_like_params = fit_HOD(path_config_filename=path_config_filename, save_chains=True)

    print("Done HOD fitting!", flush=True)
    newBall.update_HOD_params(max_like_params)

    print("Getting NFW draw for satellites", flush=True)#############################################################
    max_nfw = 40
    NFW_draw = nfw_draw(10000, max_nfw, seed)

    print("Making final mock...", flush=True)##########################################################################
    mock_dict = newBall.run_hod(
        newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=16, verbose=True
    )
    print("wp of final mock:", newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size), flush=True)


class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgParseFormatter
    )
    parser.add_argument(
        '--path_config_filename', help='Path to the config file', default=DEFAULTS['path_config_filename']
    )
    args = vars(parser.parse_args())
    main(**args)
