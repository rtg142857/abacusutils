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
        filename = paircount_path + pair + ".npy"
        paircounts[pair] = np.load(filename)
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
    print("Best params:", best_fit, flush=True)
    print("Chi squared:", OptimizeResult["fun"], flush=True)
    print("Iterations:", OptimizeResult["nit"], flush=True)
    print("Successful:", OptimizeResult["success"], flush=True)
    print("Output message:", OptimizeResult["message"], flush=True)

    print("Saving output...", flush=True)
    save_path = fitting_params["sampler_save_path"]
    np.save(save_path+"stoch_xall.npy", OptimizeResult["xall"])
    np.save(save_path+"stoch_funall.npy", OptimizeResult["funall"])

    print("Saving HOD values...")
    M_h = np.logspace(10, 16, 90)
    hod_values = get_hod_values_given_parameters(M_h, best_fit, tracer_list, other_stuff_dict_here)
    for key, val in hod_values.items():
        np.save(save_path + key + ".npy", val)
    plot_HODs(save_path+"HODs.png", M_h, hod_values, tracer_list)    

    return best_fit

def sample_chain(target_wp_dict: dict, target_jackknife_inverse_dict: dict, target_ngal_dict: dict, paircounts: dict, tracer_list: list, clustering_parameters: dict, other_stuff_dict_here: dict, nwalkers: int, num_steps: int):

    bounds = get_priors()
    minimum = True

    print("Running optimisation...", flush=True)
    OptimizeResult = minimize(log_probability, bounds, method="cmaes", args=(paircounts, tracer_list, target_wp_dict, target_jackknife_inverse_dict, target_ngal_dict, other_stuff_dict_here, clustering_parameters, minimum), options={"maxiter": num_steps, "popsize": nwalkers, "seed": 0, "return_all": True})

    return OptimizeResult

def plot_HODs(save_path, M_h, hod_values, tracers):
    import matplotlib.pyplot as plt
    tracer_cols = {"LRG": "black", "ELG": "green", "QSO": "orange"}
    for tracer in tracers:
        cen = hod_values[tracer+"_cen"]
        sat = hod_values[tracer+"_sat"]
        plt.loglog(M_h, cen, color=tracer_cols[tracer], label=tracer+" cen")
        plt.loglog(M_h, sat, color=tracer_cols[tracer], linestyle='dashed', label=tracer+" sat")
    plt.savefig(save_path)