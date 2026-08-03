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

from Corrfunc.theory.DDrppi import DDrppi

def convert_ddrppi(output):
    n_pairs = np.zeros((24, 80))
    for i in range(24):
        for j in range(80):
            idx = i * 80 + j
            n_pairs[i,j] = output[idx][4]
    return n_pairs


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
                  Ep["p_max"], Ep["logM_cut"], Ep["kappa"], Ep["sigma"], Ep["logM1"], Ep["logM1_EE"], Ep["alpha"], Ep["gamma"],
                  Qp["logM_cut"], Qp["logM1"], Qp["sigma"], Qp["alpha"], Qp["kappa"], Qp["p_max"]]
    clustering_params = config['clustering_params']
    Paths = config["Paths"]
    Labels = config["Labels"]
    #Params = config["Params"]
    Misc = config["Misc"]
    seed = Misc["random_seed"]
    tracer_list = ["LRG", "ELG", "QSO"]

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
    # mock_wp_exists = os.path.isfile(temp_stuff + "mock_wp_LL.npy") and os.path.isfile(temp_stuff + "mock_wp_LE.npy") and os.path.isfile(temp_stuff + "mock_wp_EE.npy")
    # pair_wp_exists = os.path.isfile(temp_stuff + "pair_wp.npy")

    # paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/Debugging_fitting/"
    # paircounts_exist = os.path.isfile(paircount_path + "cencen.npy")

    print("Making new FlamingoHOD object", flush=True)###############################################################
    # create a new FlamingoHOD object
    newBall = FlamingoHOD(path_config_filename)

    print("Getting NFW draw for satellites", flush=True)#############################################################
    max_nfw = 40
    NFW_draw = nfw_draw(10000, max_nfw, seed)

    # paircount_labels = ["cencen", "censat", "satsat", "satsat_onehalo", "cencen_ELGauto", "censat_ELGauto", "satsat_ELGauto", "satsat_onehalo_ELGauto",
    #                     "cencen_ELGcross", "censat_ELGcross", "satsat_ELGcross", "satsat_onehalo_ELGcross"]
    pairs_list = ["cencen", "censat_full", "censat_1halo", "satsat_2halo", "satsat_1halo"]
    category_list = ["", "_ELGauto", "_ELGcross"]
    all_paircounts_exist = True
    for pair_label in pairs_list:
        for category in category_list:
            if not os.path.exists(paircount_path + sim_label + f"/{pair_label}{category}.npy"):
                all_paircounts_exist = False
    if not all_paircounts_exist:
        print("Paircounts missing; computing them now", flush=True)
        print("Making tracer mock...", flush=True)
        max_nfw = 40
        NFW_draw = nfw_draw(10000, max_nfw, seed)
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=True, seed=seed
        )
        print("Paircounting...")
        paircounting.get_paircounts(path_config_filename=path_config_filename, tracer_mock = mock_dict, Nthread=32, save=True, verbose=True)

    #if not pair_wp_exists:

    print("Loading paircounts from the tabulation mock", flush=True)###############################################
    paircount_path = config["fitting_params"]["paircounts_save_path"] + sim_label + "/"
    paircounts = {}
    for pair in ["cencen", "censat_full", "censat_1halo", "satsat_2halo", "satsat_1halo"]:
        for pair_type in ["", "_ELGauto", "_ELGcross"]:
            filename = paircount_path + pair + pair_type + ".npy"
            paircounts[pair+pair_type] = np.load(filename)
    other_stuff_dict_here = make_other_stuff_dict(boxsize=boxsize, num_sat_parts=3, subsample_dir=subsample_dir, sim_label=sim_label)

    param_set = Params(tracer_list=tracer_list)

    npart = get_npart(HOD_params_list, param_set, other_stuff_dict_here)
    # npart = get_npart(HOD_params_list, tracer_list=["LRG", "ELG", "QSO"], other_stuff_dict_here=other_stuff_dict_here)
    print("LRG npart:", npart["LRG"])
    print("ELG npart:", npart["ELG"])
    print("QSO npart:", npart["QSO"])
    wp_dict_pair = get_wp(HOD_params_list, paircounts, param_set, npart, other_stuff_dict_here, clustering_params, verbose=True)
    # wp_pair_LRGLRG = get_wp_given_tracer(HOD_params_list, "LRG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="LRG", verbose=True)
    # wp_pair_ELGLRG = get_wp_given_tracer(HOD_params_list, "LRG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="ELG", verbose=True)
    # wp_pair_ELGELG = get_wp_given_tracer(HOD_params_list, "ELG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="ELG", verbose=True)
    # wp_pair_LRGQSO = get_wp_given_tracer(HOD_params_list, "LRG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="QSO", verbose=True)
    # wp_pair_ELGQSO = get_wp_given_tracer(HOD_params_list, "ELG", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="QSO", verbose=True)
    # wp_pair_QSOQSO = get_wp_given_tracer(HOD_params_list, "QSO", paircounts, npart, other_stuff_dict_here, clustering_params, tracer2="QSO", verbose=True)
    np.save(temp_stuff + "pair_wp_LRG_LRG.npy", wp_dict_pair["LRG_LRG"])
    np.save(temp_stuff + "pair_wp_LRG_ELG.npy", wp_dict_pair["LRG_ELG"])
    np.save(temp_stuff + "pair_wp_ELG_ELG.npy", wp_dict_pair["ELG_ELG"])
    np.save(temp_stuff + "pair_wp_LRG_QSO.npy", wp_dict_pair["LRG_QSO"])
    np.save(temp_stuff + "pair_wp_ELG_QSO.npy", wp_dict_pair["ELG_QSO"])
    np.save(temp_stuff + "pair_wp_QSO_QSO.npy", wp_dict_pair["QSO_QSO"])

        # print("WP from paircounting:", wp_pair)
        # np.save(temp_stuff + "pair_wp.npy", wp_pair)
    #else:
    #    pass
        # print("pair_wp already saved, skipping")
        # wp_pair = np.load(temp_stuff + "pair_wp.npy")


    if True: #not mock_wp_exists:
        print("Getting true mock to compare wp against", flush=True)#############################################################
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=32, verbose=True, tabulation_mock=False, seed=seed
        )
        np.save(temp_stuff + "internal_hmf.npy", newBall.halo_mass_func)
        print("Calculating number of galaxies from the in-built function now", flush=True)
        ngal_dict = newBall.compute_ngal(Nthread=1)
        print("Number of galaxies according to compute_ngal:", ngal_dict)

        print("Getting wp from the true mock", flush=True)#############################################################
        # mock_dict_sat = {}
        # for tracer in tracer_list:
        #     Ncent = mock_dict[tracer]["Ncent"]
        #     mock_dict_sat[tracer] = {}
        #     for field in ["x", "y", "z", "vx", "vy", "vz", "mass", "id"]:
        #         mock_dict_sat[tracer][field] = mock_dict[tracer][field][Ncent:]
        wp_dict_true = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32) # TODO: Revert
        print("wp_dict with cens: ",wp_dict_true)
        # print("Getting ddrppi of the mock:")
        # lrgs = mock_dict["LRG"]
        # elgs = mock_dict["ELG"]
        # mock_ddrppi = DDrppi(autocorr=0, nthreads=32, pimax=pimax, #npibins=(pi_max//d_pi),
        #                  binfile=rpbins,
        #                  X1=lrgs["x"],Y1=lrgs["y"],Z1=lrgs["z"], weights1=np.ones(len(lrgs["x"])), X2=elgs["x"],
        #                  Y2=elgs["x"],Z2 = elgs["x"],weights2=np.ones(len(elgs["x"])),periodic=True,verbose=False, boxsize=boxsize, weight_type="pair_product")
        # print("Mock ddrppi:", mock_ddrppi)
        # np.save(temp_stuff + "mock_ddrppi", mock_ddrppi)

        # print("Now for the satelllites only:")
        # wp_dict_sat = newBall.compute_wp(mock_dict_sat, rpbins, pimax, pi_bin_size, Nthread=32)
        # print("wp_dict without cens: ",wp_dict_sat)
        # wp_mock = wp_dict["LRG_ELG"]

        # np.save(temp_stuff + "mock_wp_LRG_ELG", wp_dict["LRG_ELG"])
        for i in ["LRG_LRG", "LRG_ELG", "LRG_QSO", "ELG_ELG", "ELG_QSO", "QSO_QSO"]:
            np.save(temp_stuff + f"mock_wp_{i}", wp_dict_true[i])
        # wp_mock_LRGLRG = wp_dict["LRG_LRG"]
        # wp_mock_ELGLRG = wp_dict["LRG_ELG"]
        # wp_mock_ELGELG = wp_dict["ELG_ELG"]
        # wp_mock_QSOQSO = wp_dict["QSO_QSO"]
        # wp_mock_LRGQSO = wp_dict["LRG_QSO"]
        # wp_mock_ELGQSO = wp_dict["ELG_QSO"]
        # for i in [""]
        # np.save(temp_stuff + "mock_wp_LL.npy", wp_mock_LRGLRG)
        # np.save(temp_stuff + "mock_wp_LE.npy", wp_mock_ELGLRG)
        # np.save(temp_stuff + "mock_wp_EE.npy", wp_mock_ELGELG)
        # print("Getting xi from the true mock", flush=True)##############################################################
        # xi_dict = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
        # xi_mock = xi_dict["LRG_LRG"]
        # np.save(temp_stuff + "mock_xi.npy", xi_mock)
    else:
        print("mock_wp already saved, skipping")
        wp_mock_LRGLRG = np.load(temp_stuff + "mock_wp_LL.npy")
        wp_mock_ELGLRG = np.load(temp_stuff + "mock_wp_LE.npy")
        wp_mock_ELGELG = np.load(temp_stuff + "mock_wp_EE.npy")

    print("Plotting", flush=True)###########################################################################

    rpcent = np.sqrt(rpbins[1:] * rpbins[:-1])

    # fig, axs = plt.subplots(2, 3)
    # axs[0,0].loglog(rpcent, wp_dict_true["LRG_LRG"], label="True LRGa")
    # axs[0,0].loglog(rpcent, wp_dict_pair["LRG_LRG"], label="Pair LRGa")
    # axs[0,0].legend()

    # axs[0,1].loglog(rpcent, wp_dict_true["ELG_ELG"], label="True ELGa")
    # axs[0,1].loglog(rpcent, wp_dict_pair["ELG_ELG"], label="Pair ELGa")
    # axs[0,1].legend()
    
    # axs[0,2].loglog(rpcent, wp_dict_true["QSO_QSO"], label="True QSOa")
    # axs[0,2].loglog(rpcent, wp_dict_pair["QSO_QSO"], label="Pair QSOa")
    # axs[0,2].legend()

    # axs[1,0].loglog(rpcent, wp_dict_true["LRG_ELG"], label="True LEx")
    # axs[1,0].loglog(rpcent, wp_dict_pair["LRG_ELG"], label="Pair LEx")
    # axs[1,0].legend()

    # axs[1,1].loglog(rpcent, wp_dict_true["LRG_QSO"], label="True LQx")
    # axs[1,1].loglog(rpcent, wp_dict_pair["LRG_QSO"], label="Pair LQx")
    # axs[1,1].legend()
    
    # axs[1,2].loglog(rpcent, wp_dict_true["ELG_QSO"], label="True EQx")
    # axs[1,2].loglog(rpcent, wp_dict_pair["ELG_QSO"], label="Pair EQx")
    # axs[1,2].legend()

    #plt.loglog(rpcent, wp_dict_true["LRG_LRG"], label="True LRGa")
    plt.loglog(rpcent, wp_dict_true["LRG_ELG"], label="True LEx")
    #plt.loglog(rpcent, wp_dict_true["ELG_ELG"], label="True ELGa")
    #plt.loglog(rpcent, wp_dict_pair["LRG_LRG"], label="Pair LRGa")
    plt.loglog(rpcent, wp_dict_pair["LRG_ELG"], label="Pair LEx")
    #plt.loglog(rpcent, wp_dict_pair["ELG_ELG"], label="Pair ELGa")

    # plt.loglog(rpcent, wp_dict_pair["QSO_QSO"], label="Pair QSOa")
    # plt.loglog(rpcent, wp_dict_true["QSO_QSO"], label="True QSOa")

    plt.legend()
    plt.title("wp(rp)")
    plt.xlabel("r (Mpc/h)")
    plt.ylabel("wp (Mpc/h)")
    plt.savefig("fig_wprp")
    plt.show()
    plt.clf()

    print("Plotting HODs...")
    M_h = np.logspace(10, 16, 90)
    hod_values = get_hod_values_given_parameters_with_incompleteness(M_h, HOD_params_list, param_set, other_stuff_dict_here, conformity=True)
    print(hod_values)
    ELG_sat_conformity = hod_values["ELG_sat_conformity"]
    plot_HODs("HODs.png", M_h, hod_values, tracer_list, ELG_sat_conformity=ELG_sat_conformity)

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
