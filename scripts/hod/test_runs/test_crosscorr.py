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
from abacusnbody.hod.fitting.wp_paircounting import get_wp_given_tracer
import abacusnbody.hod.fitting.wp_paircounting as wp_paircounting
import abacusnbody.hod.fitting.paircounting as paircounting


DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'

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
    mock_wp_exists = os.path.isfile(temp_stuff + "mock_wp.npy")
    pair_wp_exists = os.path.isfile(temp_stuff + "pair_wp.npy")

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
        NFW_draw = nfw_draw(10000, max_nfw, seed=1)
        mock_dict_1 = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=True,
            reseed=1, seed=1
        )
        NFW_draw = nfw_draw(10000, max_nfw, seed=2)
        mock_dict_2 = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=True,
            reseed=2, seed=2
        )
        mock_dict = {}
        mock_dict["LRG"] = mock_dict_1["LRG"]
        mock_dict["ELG"] = mock_dict_2["LRG"]
        print("Paircounting...")
        paircounting.get_paircounts(path_config_filename=path_config_filename, tracer_mock = mock_dict, Nthread=32, save=True, verbose=True)

    print("All paircounts saved!", flush=True)
    # TODO: Find the "crosscorr" of the LRGs with the LRGs using these paircounts
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

    num_mass_bins_big = other_stuff_dict_here["num_mass_bins_big"]
    mass_bin_centres_big = other_stuff_dict_here["mass_bin_centres_big"]
    mass_bin_edges = other_stuff_dict_here["mass_bin_edges"]
    hmf_big = other_stuff_dict_here["hmf_big"]
    hod_cen1, hod_sat1 = wp_paircounting.get_accurate_tracer_HOD(HOD_params_list, "LRG", mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big)
    hod_cen2, hod_sat2 = hod_cen1, hod_sat1

    ccp = paircounts["cencen_ELGcross"]
    csp = paircounts["censat_ELGcross"]
    ssp = paircounts["satsat_ELGcross"]
    ss1p = paircounts["satsat_onehalo_ELGcross"]
    num_sat_parts = 3
    CC = wp_paircounting.create_weighting_factor(ccp,hod_cen1,hod_cen2)
    CS = wp_paircounting.create_weighting_factor(csp,hod_cen1,hod_sat2) / num_sat_parts
    SS = wp_paircounting.create_weighting_factor(ssp,hod_sat1,hod_sat2) / (num_sat_parts**2) 
    SS1 = wp_paircounting.create_weighting_factor(ss1p,hod_sat1,hod_sat2) / (num_sat_parts**2)
    
    GG = CC + CS + SS + SS1

    r_bin_edges = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pi_max = clustering_params['pimax']
    # getting randoms
    RR_out = np.zeros((len(r_bin_edges)-1)*pi_max)
    for p in range(pi_max):
        # do volume calculations
        v = 2*np.pi*r_bin_edges**2 # Volume of cylinders
        dv = np.diff(v)  # difference between r volumes
        global_volume = boxsize**3  # volume of simulation
        # calculate the random-random pairs using density * volume
        # crosscorr style
        rhor = (npart["LRG"]*npart["LRG"]) /global_volume
        RR = (dv*rhor)
        #print(RR)
        RR_out[p::pi_max] = RR
    #vprint(f"Sum of RR for {tracer1}, {tracer2}:", np.sum(RR_out), verbose)
    RR_out

    wp_rands = np.reshape(RR_out,newshape=(len(rpbins)-1,pimax))

    xi = np.divide(GG, wp_rands) - 1
    wp_pair = wp_paircounting.xi_to_wps(xi,rpbins,pimax)

    # wp_mock_ELGELG = []
    # for i in range(10):
    print(f"Getting true mock to compare wp against", flush=True)#############################################################
    mock_dict = newBall.run_hod(
        newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=32, verbose=True, tabulation_mock=False
    )
    print("Getting wp from the true mock", flush=True)#############################################################
    wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
    # wp_mock = wp_dict["LRG_ELG"]
    wp_mock_LRGLRG = wp_dict["LRG_LRG"]

    # np.save(temp_stuff + "internal_hmf.npy", newBall.halo_mass_func)
    # print("Calculating number of galaxies from the in-built function now", flush=True)
    # ngal_dict = newBall.compute_ngal(Nthread=1)
    # print("Number of LRGs according to compute_ngal:", ngal_dict)

    #     print("Getting wp from the true mock", flush=True)#############################################################
    #     wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
    #     # wp_mock = wp_dict["LRG_ELG"]
    #     # wp_mock_LRGLRG = wp_dict["LRG_LRG"]
    #     # wp_mock_ELGLRG = wp_dict["LRG_ELG"]
    #     wp_mock_ELGELG.append(wp_dict["ELG_ELG"])


    print("Plotting", flush=True)###########################################################################

    rpcent = np.sqrt(rpbins[1:] * rpbins[:-1])

    plt.loglog(rpcent, wp_pair, label="Pair LRGa via xcorr")
    plt.loglog(rpcent, wp_mock_LRGLRG, label=f"True LRGa {i}")
    plt.legend()
    plt.title("wp(rp)")
    plt.savefig("fig_wprp")
    plt.show()

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
