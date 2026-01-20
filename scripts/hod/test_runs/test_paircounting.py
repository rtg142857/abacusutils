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
import math
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

import h5py

from abacusnbody.hod.flamingo_hod import FlamingoHOD
from abacusnbody.hod.NFW import nfw_draw
from abacusnbody.hod.fitting.setup_paircounting_fitting import *
from abacusnbody.hod.fitting.hod_fitting_with_paircounting_stochopy import plot_HODs
from abacusnbody.hod.fitting.wp_paircounting import get_wp_given_tracer
import abacusnbody.hod.fitting.paircounting as paircounting


DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'

# Now change Cen_HOD definition

# def Cen_HOD(params,mass_bins):
#     """
#     takes params: [M_cut, sigma_logm, something, something, something]
#     """
#     Mmin, sigma_logm = params[:2]
#     result = cumulative_spline_kernel(np.log10(mass_bins), mean = Mmin, sig=sigma_logm/np.sqrt(2))
#     return(result)

# def Sat_HOD(params,cen_hod,mass_bins):
#     """
#     takes params: [Mmin, sigma_logm, logM0, logM1, alpha]
#     """
#     M0, M1, alpha = params[2:].copy()
#     M0 = 10**M0
#     M1 = 10**M1
#     result = cen_hod * (((mass_bins-M0)/M1)**alpha)
#     return(result)

def main(path_config_filename):
    # load the yaml parameters
    config = yaml.safe_load(open(path_config_filename))
    run_params = yaml.safe_load(open(config["Paths"]["params_path"]))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    Lp = HOD_params["LRG_params"]
    Ep = HOD_params["ELG_params"]
    Qp = HOD_params["QSO_params"]
    HOD_params_list = [Lp["logM_cut"], Lp["logM1"], Lp["sigma"], Lp["alpha"], Lp["kappa"],
                  Ep["p_max"], Ep["Q"], Ep["logM_cut"], Ep["kappa"], Ep["sigma"], Ep["logM1"], Ep["alpha"], Ep["gamma"],
                  Qp["logM_cut"], Qp["logM1"], Qp["sigma"], Qp["alpha"], Qp["kappa"]]
    clustering_params = config['clustering_params']
    Paths = config["Paths"]
    Labels = config["Labels"]
    Params = config["Params"]
    Misc = config["Misc"]
    seed = Misc["random_seed"]

    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(
        bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1
    )
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    subsample_dir = sim_params["subsample_dir"]
    fitting_params = config["fitting_params"]
    paircount_path = fitting_params["paircounts_save_path"]
    sim_label = Labels["sim_label"]

    temp_stuff = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/temp_stuff/"
    mock_wp_exists = os.path.isfile(temp_stuff + "mock_wp_LL.npy") and os.path.isfile(temp_stuff + "mock_wp_LE.npy") and os.path.isfile(temp_stuff + "mock_wp_EE.npy")
    pair_wp_exists = os.path.isfile(temp_stuff + "pair_wp.npy")

    # paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/Debugging_fitting/"
    # paircounts_exist = os.path.isfile(paircount_path + "cencen.npy")

    print("Making new FlamingoHOD object", flush=True)###############################################################
    # create a new FlamingoHOD object
    newBall = FlamingoHOD(path_config_filename)

    print("Getting NFW draw for satellites", flush=True)#############################################################
    max_nfw = 40
    NFW_draw = nfw_draw(10000, max_nfw, seed)

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
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=True
        )
        print("Paircounting...")
        paircounting.get_paircounts(path_config_filename=path_config_filename, tracer_mock = mock_dict, Nthread=32, save=True, verbose=True)

    #if not pair_wp_exists:

    print("Loading paircounts from the tabulation mock", flush=True)###############################################
    paircount_path = config["fitting_params"]["paircounts_save_path"] + sim_label + "/"
    paircounts = {}
    for pair in ["cencen", "censat", "satsat", "satsat_onehalo"]:
        for pair_type in ["", "_ELGauto", "_ELGcross"]:
            filename = paircount_path + pair + pair_type + ".npy"
            paircounts[pair+pair_type] = np.load(filename)
    other_stuff_dict_here = make_other_stuff_dict(boxsize=boxsize, num_sat_parts=3, subsample_dir=subsample_dir, sim_label=sim_label)

    npart = get_npart(HOD_params_list, tracer_list=["LRG", "ELG", "QSO"], other_stuff_dict_here=other_stuff_dict_here)
    print("LRG npart:", npart["LRG"])
    print("ELG npart:", npart["ELG"])
    print("QSO npart:", npart["QSO"])
    wp_pair_LRGLRG = get_wp_given_tracer(HOD_params_list, "LRG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="LRG", verbose=True)
    wp_pair_ELGLRG = get_wp_given_tracer(HOD_params_list, "LRG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="ELG", verbose=True)
    wp_pair_ELGELG = get_wp_given_tracer(HOD_params_list, "ELG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="ELG", verbose=True)

        # print("WP from paircounting:", wp_pair)
        # np.save(temp_stuff + "pair_wp.npy", wp_pair)
    #else:
    #    pass
        # print("pair_wp already saved, skipping")
        # wp_pair = np.load(temp_stuff + "pair_wp.npy")


    if not mock_wp_exists:
        print("Getting true mock to compare wp against", flush=True)#############################################################
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=False
        )
        np.save(temp_stuff + "internal_hmf.npy", newBall.halo_mass_func)
        print("Calculating number of galaxies from the in-built function now", flush=True)
        ngal_dict = newBall.compute_ngal(Nthread=1)
        print("Number of LRGs according to compute_ngal:", ngal_dict)

        print("Getting wp from the true mock", flush=True)#############################################################
        wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
        # wp_mock = wp_dict["LRG_ELG"]
        wp_mock_LRGLRG = wp_dict["LRG_LRG"]
        wp_mock_ELGLRG = wp_dict["LRG_ELG"]
        wp_mock_ELGELG = wp_dict["ELG_ELG"]
        np.save(temp_stuff + "mock_wp_LL.npy", wp_mock_LRGLRG)
        np.save(temp_stuff + "mock_wp_LE.npy", wp_mock_ELGLRG)
        np.save(temp_stuff + "mock_wp_EE.npy", wp_mock_ELGELG)
        # print("Getting xi from the true mock", flush=True)##############################################################
        # xi_dict = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
        # xi_mock = xi_dict["LRG_LRG"]
        # np.save(temp_stuff + "mock_xi.npy", xi_mock)
    else:
        pass
        # print("mock_wp already saved, skipping")
        # wp_mock = np.load(temp_stuff + "mock_wp.npy")

    print("Plotting", flush=True)###########################################################################

    rpcent = np.sqrt(rpbins[1:] * rpbins[:-1])

    plt.loglog(rpcent, wp_mock_LRGLRG, label="True LRGa")
    plt.loglog(rpcent, wp_mock_ELGLRG, label="True LEx")
    plt.loglog(rpcent, wp_mock_ELGELG, label="True ELGa")
    plt.loglog(rpcent, wp_pair_LRGLRG, label="Pair LRGa")
    plt.loglog(rpcent, wp_pair_ELGLRG, label="Pair LEx")
    plt.loglog(rpcent, wp_pair_ELGELG, label="Pair ELGa")
    plt.legend()
    plt.title("wp(rp)")
    plt.savefig("fig_wprp")
    plt.show()
    plt.clf()

    print("Plotting HODs...")
    M_h = np.logspace(10, 16, 90)
    tracer_list = ["LRG", "ELG", "QSO"]
    hod_values = get_hod_values_given_parameters(M_h, HOD_params_list, tracer_list, other_stuff_dict_here)
    plot_HODs("HODs.png", M_h, hod_values, tracer_list)

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
