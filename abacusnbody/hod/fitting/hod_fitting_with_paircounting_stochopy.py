# Fits an HOD to observational data.
# Mostly borrowed from Alex Smith's FastHodFitting
# and modified to work with Flamingo and the LRG/ELG/QSO tracers.

import yaml
import numpy as np
import time

from abacusnbody.hod.flamingo_hod import FlamingoHOD

from abacusnbody.hod.fitting.setup_paircounting_fitting import *

from stochopy.optimize import minimize

nthread = 64 # For debugging

def fit_HOD(path_config_filename, save_chains=False):
    # Initialise the fitting using emcee
    # Use a different function to actually do the fit (modularity)
    # Print to files: the updated parameters, an image of the HODs, the final fit to the wp (text and image), the errors in fitting to the wp (text and image)
    config = yaml.safe_load(open(path_config_filename))
    run_params = yaml.safe_load(open(config["Paths"]["params_path"]))
    fitting_params = config["fitting_params"]
    sim_params = config['sim_params']
    Labels = config["Labels"]
    subsample_dir = sim_params["subsample_dir"]
    sim_label = Labels["sim_label"]
    target_dict_path = fitting_params["target_dict_path"]
    paircount_path = fitting_params["paircounts_save_path"] + sim_label + "/"
    boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    tracer_list = ["LRG", "ELG", "QSO"]

    target_wp, target_jackknife_inverse = get_target_dicts(target_dict_path, tracers=tracer_list)
    target_ngal = get_target_number_density(tracers=tracer_list)

    clustering_params = config["clustering_params"]

    print("Loading precomputed things...")
    paircounts = {}
    for pair in ["cencen", "censat", "satsat", "satsat_onehalo"]:
        for pair_type in ["", "_ELGauto", "_ELGcross"]:
            filename = paircount_path + pair + pair_type + ".npy"
            paircounts[pair+pair_type] = np.load(filename)
    other_stuff_dict_here = make_other_stuff_dict(boxsize=boxsize, num_sat_parts=3, subsample_dir=subsample_dir, sim_label=sim_label)

    nwalkers = fitting_params["nwalkers"]
    num_steps = fitting_params["num_steps"]

    start_time = time.time()

    OptimizeResult = sample_chain(target_wp_dict=target_wp,
                           target_jackknife_inverse_dict=target_jackknife_inverse,
                           target_ngal_dict=target_ngal,
                           paircounts=paircounts,
                           tracer_list=tracer_list,
                           clustering_parameters=clustering_params,
                           other_stuff_dict_here=other_stuff_dict_here,
                           nwalkers=nwalkers,
                           num_steps=num_steps)
    end_time = time.time()
    print("fitting took ", end_time - start_time, " seconds", flush=True)

    print("Optimization done", flush=True)
    best_fit = OptimizeResult["x"]
    print("Best params:")#, best_fit, flush=True)
    print_hod_values(best_fit)
    print("Chi squared:", OptimizeResult["fun"], flush=True)
    print("Iterations:", OptimizeResult["nit"], flush=True)
    print("Successful:", OptimizeResult["success"], flush=True)
    print("Output message:", OptimizeResult["message"], flush=True)

    print("Saving output...", flush=True)
    save_path = fitting_params["sampler_save_path"]
    np.save(save_path+"stoch_xall.npy", OptimizeResult["xall"])
    np.save(save_path+"stoch_funall.npy", OptimizeResult["funall"])

    print("Saving HOD values...", flush=True)
    M_h = np.logspace(10, 16, 90)
    hod_values = get_hod_values_given_parameters(M_h, best_fit, tracer_list, other_stuff_dict_here)
    for key, val in hod_values.items():
        np.save(save_path + key + ".npy", val)

    print("Plotting HODs...")
    plot_HODs(save_path+"HODs.png", M_h, hod_values, tracer_list)

    print("Plotting wps...", flush=True)
    plot_wp(save_path+"wps.png", hod_params=best_fit, tracers=tracer_list, paircounts=paircounts,
            target_wp=target_wp, target_jackknife_inverse=target_jackknife_inverse, other_stuff_dict_here=other_stuff_dict_here, clustering_params=clustering_params)

    return best_fit

def sample_chain(target_wp_dict: dict, target_jackknife_inverse_dict: dict, target_ngal_dict: dict, paircounts: dict, tracer_list: list, clustering_parameters: dict, other_stuff_dict_here: dict, nwalkers: int, num_steps: int):

    bounds = get_priors(type="bounds")
    x0 = get_priors(type="mean") # just needs one initial guess apparently
    minimum = True

    print("Running optimisation...", flush=True)
    OptimizeResult = minimize(log_probability, bounds, x0, method="cmaes",
                              args=(paircounts, tracer_list, target_wp_dict, target_jackknife_inverse_dict, target_ngal_dict, other_stuff_dict_here, clustering_parameters, minimum),
                              options={"maxiter": num_steps, "popsize": nwalkers, "seed": 0, "return_all": True, "workers": -1})

    return OptimizeResult

def plot_HODs(save_path, M_h, hod_values, tracers):
    import matplotlib.pyplot as plt
    tracer_cols = {"LRG": "black", "ELG": "green", "QSO": "orange"}
    for tracer in tracers:
        cen = hod_values[tracer+"_cen"]
        sat = hod_values[tracer+"_sat"]
        plt.loglog(M_h, cen, color=tracer_cols[tracer], label=tracer+" cen")
        plt.loglog(M_h, sat, color=tracer_cols[tracer], linestyle='dashed', label=tracer+" sat")
        plt.ylim(10**-3, 10**3)
        plt.legend()
    plt.savefig(save_path)

def plot_wp(save_path, hod_params, tracers, paircounts, target_wp, target_jackknife_inverse, other_stuff_dict_here, clustering_params):
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
            rwp_error_mat = np.linalg.inv(target_jackknife_inverse[label_dict[yx_label]])
            rwp_error = np.diagonal(rwp_error_mat) * rpcent
            axs[y, x].errorbar(rpcent[i0:i1], rwp_data[i0:i1], yerr=rwp_error[i0:i1], color="orange", label=f"Data ("+label_dict[yx_label]+")")
            axs[y, x].plot(rpcent[:i0], rwp_data[:i0], marker="o", color="black", label="Data (unused)")
            axs[y, x].set_xscale('log')
            axs[y, x].legend()

            if y == 0 and x == 2: # QSO_QSO
                axs[y, x].set_ylim([-50, 250])

    plt.savefig(save_path)