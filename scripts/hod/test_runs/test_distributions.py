import argparse
import time
import os
import math
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from abacusnbody.hod.flamingo_hod import FlamingoHOD
from abacusnbody.hod.NFW import nfw_draw

def cut_til_one_sat_tracer(array, Ncen):
    array_no_cen = array[Ncen:]
    return array_no_cen[::3]
def mass_cut(array, M_h, Mmin, Mmax):
    below_cut = M_h > Mmin
    above_cut = M_h < Mmax
    mask = np.logical_and(below_cut, above_cut)
    return array[mask]

def main(path_config_filename):
    # load the yaml parameters
    config = yaml.safe_load(open(path_config_filename))
    # run_params = yaml.safe_load(open(config["Paths"]["params_path"]))
    # sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    tracer_list=["LRG", "ELG", "QSO"]
    Misc = config["Misc"]
    want_rsd = HOD_params['want_rsd']
    seed = Misc["random_seed"]
    clustering_params = config['clustering_params']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(
        bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1
    )
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']

    print("Making new FlamingoHOD object", flush=True)###############################################################
    # create a new FlamingoHOD object
    newBall = FlamingoHOD(path_config_filename)

    print("Getting NFW draw for satellites", flush=True)#############################################################
    max_nfw = 40
    NFW_draw = nfw_draw(10000, max_nfw, seed)

    print("Making tabulation mock...", flush=True)###################################################################
    max_nfw = 40
    NFW_draw = nfw_draw(10000, max_nfw, seed)
    tab_mock_dict = newBall.run_hod(
        newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=True, seed=seed
    )

    mass_bin_edges = 10**10 * np.logspace(0,6,31)
    mass_lower = mass_bin_edges[9]
    mass_upper = mass_bin_edges[10]
    for tracer in tracer_list:
        M_h = tab_mock_dict[tracer]["mass"]
        Ncen = tab_mock_dict[tracer]["Ncent"]
        for field in tab_mock_dict[tracer].keys():
            tab_mock_dict[tracer][field] = cut_til_one_sat_tracer(tab_mock_dict[tracer][field], Ncen)
            tab_mock_dict[tracer][field] = mass_cut(tab_mock_dict[tracer][field], M_h, mass_lower, mass_upper)

    print("Making 'true' mock...", flush=True)##########################################################################
    true_mock_dict = newBall.run_hod(
        newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=True, seed=seed
    )
    print("Sanity check: Number of LRGs and ELGs")
    print("Number of LRGs in the tab mock:", len(tab_mock_dict["LRG"]["mass"]))
    print("Number of ELGs in the tab mock:", len(tab_mock_dict["ELG"]["mass"]))
    print("Number of LRGs in the true mock:", len(true_mock_dict["LRG"]["mass"]))
    print("Number of ELGs in the true mock:", len(true_mock_dict["ELG"]["mass"]))

    print("Getting wps...", flush=True)################################################################################
    true_wp_dict = newBall.compute_xirppi(true_mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
    tab_wp_dict = newBall.compute_xirppi(tab_mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)

    print("Plotting...", flush=True)#####################################################################################

    rpcent = np.sqrt(rpbins[1:] * rpbins[:-1])

    plt.loglog(rpcent, true_wp_dict["LRG_LRG"], label="True LRGa")
    plt.loglog(rpcent, true_wp_dict["LRG_ELG"], label="True LEx")
    plt.loglog(rpcent, true_wp_dict["ELG_ELG"], label="True ELGa")
    plt.loglog(rpcent, tab_wp_dict["LRG_LRG"], label="Tab LRGa")
    plt.loglog(rpcent, tab_wp_dict["LRG_ELG"], label="Tab LEx")
    plt.loglog(rpcent, tab_wp_dict["ELG_ELG"], label="Tab ELGa")


DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'

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
