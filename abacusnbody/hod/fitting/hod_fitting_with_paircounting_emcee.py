# Fits an HOD to observational data.
# Mostly borrowed from Alex Smith's FastHodFitting
# and modified to work with Flamingo and the LRG/ELG/QSO tracers.

import yaml
import numpy as np
import time

from abacusnbody.hod.flamingo_hod import FlamingoHOD

from abacusnbody.hod.fitting.setup_paircounting_fitting import *

import emcee

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
    tracers=["LRG", "ELG", "QSO"]

    wp_limit = (12, 20)
    target_wp, target_jackknife_inverse = get_target_dicts(target_dict_path, tracers=tracers, wp_limit=wp_limit)
    target_ngal = get_target_number_density(tracers=tracers)

    nwalkers = fitting_params["nwalkers"]
    num_steps = fitting_params["num_steps"]

    clustering_params = config["clustering_params"]

    print("Loading precomputed things...")
    paircounts = {}
    for pair in ["cencen", "censat", "satsat", "satsat_onehalo"]:
        for pair_type in ["", "_ELGauto", "_ELGcross"]:
            filename = paircount_path + pair + pair_type + ".npy"
            paircounts[pair+pair_type] = np.load(filename)
        # filename = paircount_path + pair + ".npy"
        # paircounts[pair] = np.load(filename)
    other_stuff_dict_here = make_other_stuff_dict(boxsize=boxsize, num_sat_parts=3, subsample_dir=subsample_dir, sim_label=sim_label)
    param_set = Params(tracer_list=tracers)
    ndim = len(param_set.prior_bounds)

    print("Setting up backend...", flush=True)
    start_time = time.time()
    if save_chains:
        filename = fitting_params["sampler_save_path"] + "emcee.hdf5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    sampler = sample_chain(target_wp_dict=target_wp,
                           target_jackknife_inverse_dict=target_jackknife_inverse,
                           target_ngal_dict=target_ngal,
                           paircounts=paircounts,
                           param_set=param_set,
                           clustering_parameters=clustering_params,
                           other_stuff_dict_here=other_stuff_dict_here,
                           backend=backend,
                           nwalkers=nwalkers,
                           num_steps=num_steps,
                           ndim=ndim,
                           wp_limit=wp_limit)
    end_time = time.time()
    print("fitting took ", end_time - start_time, " seconds", flush=True)

    # Print the parameters at the end of the chain to check they are reasonable
    print("Parameters at the end of the fitting chain: (These aren't best fits, just a sanity check)")
    print(sampler.backend.get_chain()[-1,0])

    print("All done!")
    print("Saving outputs")

    plot_sampler(sampler)

    return max_like_params(sampler)

def sample_chain(target_wp_dict: dict, target_jackknife_inverse_dict: dict, target_ngal_dict: dict, paircounts: dict, param_set: Params, clustering_parameters: dict, other_stuff_dict_here: dict, backend: emcee.backends.HDFBackend, nwalkers: int, num_steps: int, ndim=15, wp_limit=(0, 24)):

    print("Initialising walkers...", flush=True)
    walker_init_pos = param_set.get_initial_params(positions=nwalkers).T
    #walker_init_pos = initialise_walkers(initial_params_random=True,num_walkers=nwalkers)

    print("Initialising sampler...", flush=True)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(paircounts, param_set, target_wp_dict, target_jackknife_inverse_dict, target_ngal_dict, other_stuff_dict_here, clustering_parameters), kwargs={"minimise": False, "wp_limit": wp_limit, "verbose": False}, backend=backend)#, pool=pool)

    print("Running chain...", flush=True)
    sampler.run_mcmc(walker_init_pos, num_steps, skip_initial_state_check=True) # It feels like it likes to throw an error for the initial state check with the standard priors
    return sampler

def plot_sampler(sampler: emcee.EnsembleSampler):
    """
    Postprocesses a sampler, displays a corner plot of its parameters, and outputs its maximum posterior values
    """
    tau = sampler.get_autocorr_time()
    print("Sampler autocorrelation time (burn in estimate):",tau, flush=True)
    if np.any(np.isnan(tau)):
        print("NaN autocorrelation time; not burnt in")
        return
    discard_value = int(3 * np.average(tau))
    thin_value = int(0.5 * np.average(tau))
    flat_samples = sampler.get_chain(discard=discard_value, thin=thin_value, flat=True)
    print("Flattened sampler shape:",flat_samples.shape, flush=True)

    # import corner
    # fig = corner.corner(
    #     flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)]
    # )
    # TODO: this

def max_like_params(sampler):
    """Get the parameters which provide the maximum likelihood from the sampling"""

    # Only take after the first 1000 steps to allow burn in
    flat_samples = sampler.backend.get_chain()[:,:,:]
    likelihoods = sampler.backend.get_log_prob()[:,:]

    # Find the maximum likelihood parameter position

    #print(np.shape(likelihoods))
    best_param_index1 = np.unravel_index(np.argmax(likelihoods, axis=None), likelihoods.shape)

    best_params = flat_samples[best_param_index1[0], best_param_index1[1], :]
    print("best param index:",best_param_index1, flush=True)
    print("best params:",best_params, flush=True)
    return best_params