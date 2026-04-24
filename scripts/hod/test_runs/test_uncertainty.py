import numpy as np
import yaml
import argparse
from abacusnbody.hod.fitting.hod_fitting_with_paircounting_stochopy import make_other_stuff_dict
from abacusnbody.hod.fitting.setup_paircounting_fitting import get_npart, get_wp
from pycorr import TwoPointCorrelationFunction, twopoint_estimator

def get_target_dicts(target_dict_path, tracers=["LRG", "ELG", "QSO"]):
    """
    Returns dictionaries indexed by "LRG_LRG", "LRG_ELG", etc.
    One for the wp, one for the jackknife.
    """
    wp_dict = {}
    inverse_jackknife_dict = {}
    jackknife_dict = {}
    for i1, tr1 in enumerate(tracers):
        for i2, tr2 in enumerate(tracers):
            if i1 <= i2:
                #crosscorr or autocorr
                # path = target_dict_path + tr1 + "_" + tr2 + ".txt"
                # rpmid, rpavg, corr, std = np.loadtxt(path, unpack=True)
                # wp_dict[tr1+"_"+tr2] = corr
                # jackknife_dict[tr1+"_"+tr2] = std

                path = target_dict_path + tr1 + "_" + tr2 + ".npy"
                estimator = TwoPointCorrelationFunction.load(path)
                rebinned_estimator = estimator[:(estimator.shape[0] // 2) * 2:2] # getting it to be 24 bins
                sep, wp, cov = twopoint_estimator.project_to_wp(rebinned_estimator, return_cov=True)
                cov_inv = np.linalg.inv(cov)
                wp_dict[tr1+"_"+tr2] = wp
                inverse_jackknife_dict[tr1+"_"+tr2] = cov_inv
                jackknife_dict[tr1+"_"+tr2] = cov
    return wp_dict, inverse_jackknife_dict, jackknife_dict

def plot_wp(save_path, hod_params, tracers, paircounts, target_wp, target_jackknife, other_stuff_dict_here, clustering_params):
    import matplotlib.pyplot as plt

    npart = get_npart(hod_params, tracers, other_stuff_dict_here)
    wp_dict = get_wp(hod_params, paircounts, tracers, npart, other_stuff_dict_here, clustering_params)

    bin_params = clustering_params["bin_params"]
    rpbins = np.logspace(bin_params["logmin"], bin_params["logmax"], bin_params["nbins"] + 1)
    rpcent = np.sqrt(rpbins[1:] * rpbins[:-1])

    label_dict = {"0_0": "LRG_LRG", "0_1": "ELG_ELG", "0_2": "QSO_QSO", "1_0": "LRG_ELG", "1_1": "LRG_QSO", "1_2": "ELG_QSO"}

    i0 = 5 # 
    i1 = np.size(target_wp["LRG_LRG"])

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for y in range(2):
        for x in range(3):
            yx_label = str(y)+"_"+str(x)
            rwp_flamingo = rpcent * wp_dict[label_dict[yx_label]]
            axs[y, x].plot(rpcent, rwp_flamingo, "-b", label="FlamingoHOD")

            rwp_data = rpcent * target_wp[label_dict[yx_label]]
            rwp_error_mat = target_jackknife[label_dict[yx_label]]
            rwp_error = np.sqrt(np.diagonal(rwp_error_mat)) * rpcent
            axs[y, x].errorbar(rpcent[i0:i1], rwp_data[i0:i1], yerr=rwp_error[i0:i1], color="orange", label=f"Data ("+label_dict[yx_label]+")")
            axs[y, x].plot(rpcent[:i0], rwp_data[:i0], marker="o", color="black", label="Data (unused)")
            axs[y, x].set_xscale('log')
            axs[y, x].legend()

            if y == 0 and x == 2: # QSO_QSO
                axs[y, x].set_ylim([-50, 250])

    plt.savefig(save_path)

def main(path_config_filename):
    # load the yaml parameters
    config = yaml.safe_load(open(path_config_filename))
    run_params = yaml.safe_load(open(config["Paths"]["params_path"]))

    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    Lp = HOD_params["LRG_params"]
    Ep = HOD_params["ELG_params"]
    Qp = HOD_params["QSO_params"]
    # HOD_params_list = [Lp["logM_cut"], Lp["logM1"], Lp["sigma"], Lp["alpha"], Lp["kappa"],
    #               Ep["p_max"], Ep["Q"], Ep["logM_cut"], Ep["kappa"], Ep["sigma"], Ep["logM1"], Ep["alpha"], Ep["gamma"],
    #               Qp["logM_cut"], Qp["logM1"], Qp["sigma"], Qp["alpha"], Qp["kappa"], Qp["p_max"]]
    HOD_params_list = [12.85, 14.1, 0.02, 1.64, 0.015,
                  0.85, 45., 10.83, 2.8, 2.39, 14.77, 0.07, 84.77,
                  13.37, 14.322, 0.74, 0.38, 4.1, 0.49]

    tracer_list = ["LRG", "ELG", "QSO"]
    clustering_params = config["clustering_params"]

    Labels = config["Labels"]
    subsample_dir = sim_params["subsample_dir"]
    sim_label = Labels["sim_label"]
    boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    ### Paths ###
    save_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/"
    paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/" + sim_label + "/"
    target_dict_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/fitting_data/Y1_z0.8-1.1/"

    print("Loading precomputed things...")
    paircounts = {}
    for pair in ["cencen", "censat", "satsat", "satsat_onehalo"]:
        for pair_type in ["", "_ELGauto", "_ELGcross"]:
            filename = paircount_path + pair + pair_type + ".npy"
            paircounts[pair+pair_type] = np.load(filename)
    target_wp, target_jackknife_inverse, target_jackknife = get_target_dicts(target_dict_path, tracers=tracer_list)
    print(target_wp)
    for key in target_jackknife.keys():
        print(key+":")
        print(np.diag(target_jackknife_inverse[key]))
        print(np.diag(target_jackknife[key]))
    other_stuff_dict_here = make_other_stuff_dict(boxsize=boxsize, num_sat_parts=3, subsample_dir=subsample_dir, sim_label=sim_label)
    print("Plotting wps...", flush=True)
    plot_wp(save_path+"wps.png", hod_params=HOD_params_list, tracers=tracer_list, paircounts=paircounts,
            target_wp=target_wp, target_jackknife=target_jackknife, other_stuff_dict_here=other_stuff_dict_here, clustering_params=clustering_params)

class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'

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
