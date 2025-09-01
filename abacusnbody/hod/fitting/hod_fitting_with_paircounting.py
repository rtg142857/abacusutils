# Fits an HOD to observational data.
# Mostly borrowed from Alex Smith's FastHodFitting
# and modified to work with Flamingo and the LRG/ELG/QSO tracers.

import yaml
import numpy as np
import time
import os
from pathlib import Path

from abacusnbody.hod.flamingo_hod import FlamingoHOD
from wp_paircounting import get_wp, get_npart
import emcee
from pycorr import TwoPointCorrelationFunction, twopoint_estimator
import h5py

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
    paircount_path = fitting_params["paircounts_save_path"]
    boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    target_wp, target_jackknife = get_target_dicts(target_dict_path, tracers=["LRG", "ELG", "QSO"])
    #TODO: Get the target number density too

    nwalkers = fitting_params["nwalkers"]
    num_steps = fitting_params["num_steps"]
    ndim = 18

    clustering_params = config["clustering_params"]

    print("Loading precomputed things...")
    paircounts = {}
    for pair in ["cencen", "censat", "satsat", "satsat_onehalo"]:
        filename = paircount_path + pair + ".npy"
        paircounts[pair] = np.load(filename)
    other_stuff_dict_here = make_other_stuff_dict(boxsize=boxsize, num_sat_parts=3, subsample_dir=subsample_dir, sim_label=sim_label)

    print("Setting up backend...", flush=True)
    start_time = time.time()
    if save_chains:
        filename = fitting_params["sampler_save_path"]
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    sampler = sample_chain(target_wp_dict=target_wp,
                           target_jackknife_dict=target_jackknife,
                           paircounts=paircounts,
                           tracer_list=["LRG", "ELG", "QSO"],
                           clustering_parameters=clustering_params,
                           other_stuff_dict_here=other_stuff_dict_here,
                           backend=backend,
                           nwalkers=nwalkers,
                           num_steps=num_steps,
                           ndim=ndim)
    end_time = time.time()
    print("fitting took ", end_time - start_time, " seconds", flush=True)

    # Print the parameters at the end of the chain to check they are reasonable
    print("Parameters at the end of the fitting chain: (These aren't best fits, just a sanity check)")
    print(sampler.backend.get_chain()[-1,0])

    print("All done!")
    print("Saving outputs")

    plot_sampler(sampler)

    return max_like_params(sampler)

def sample_chain(target_wp_dict: dict, target_jackknife_dict: dict, paircounts: dict, tracer_list: list, clustering_parameters: dict, other_stuff_dict_here: dict, backend: emcee.backends.HDFBackend, nwalkers: int, num_steps: int, ndim=15):

    print("Initialising walkers...", flush=True)
    walker_init_pos = initialise_walkers(initial_params_random=True,num_walkers=nwalkers)

    print("Initialising sampler...", flush=True)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(paircounts, tracer_list, target_wp_dict, target_jackknife_dict, other_stuff_dict_here, clustering_parameters), backend=backend)#, pool=pool)

    print("Running chain...", flush=True)
    sampler.run_mcmc(walker_init_pos, num_steps, skip_initial_state_check=True) # It feels like it likes to throw an error for the initial state check with the standard priors
    return sampler

def log_probability(hod_params, paircounts, tracer_list, target_wp_dict, target_jackknife_dict, other_stuff_dict_here, clustering_parameters):
    if params_inside_priors(hod_params):
        # newBall.update_HOD_params(params)
        # print(params, flush=True) # Debugging
        # mock_dict = newBall.run_hod(
        #     newBall.tracers, want_rsd=True, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=nthread, verbose=False
        # )

        # rpbins = np.logspace(clustering_parameters["bin_params"]["logmin"], clustering_parameters["bin_params"]["logmax"], clustering_parameters["bin_params"]["nbins"]+1)
        # pimax = clustering_parameters["pimax"]
        # pi_bin_size = clustering_parameters["pi_bin_size"]
        # wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=nthread)

        npart = get_npart(hod_params, tracer_list, other_stuff_dict_here)
        wp_dict = get_wp(hod_params, paircounts, tracer_list, npart, other_stuff_dict_here, clustering_parameters)
        
        total_log_prob = 0.0
        for i1, tr1 in enumerate(tracer_list):
            for i2, tr2 in enumerate(tracer_list):
                if i1 <= i2:
                    #crosscorr or autocorr
                    fitting_wp = wp_dict[tr1+"_"+tr2]
                    target_wp = target_wp_dict[tr1+"_"+tr2]
                    target_jk = target_jackknife_dict[tr1+"_"+tr2]
                    total_log_prob += negative_chi_squared_single_tracer(fitting_wp, target_wp, target_jk)
        # TODO: Include number density factor
    else:
        total_log_prob = -np.inf

    return total_log_prob

def negative_chi_squared_single_tracer(fitting_wp: np.ndarray, target_wp: np.ndarray, target_jackknife: np.ndarray):
    # TODO: ONLY LOOK AT A SUBSET OF THE DATA POINTS
    i0 = 0
    i1 = np.size(fitting_wp)
    mock = fitting_wp[i0:i1]
    data = target_wp[i0:i1]
    C_matrix = target_jackknife[i0:i1, i0:i1]

    temp = np.matmul(C_matrix, mock-data)
    chi2 = np.dot(mock-data, temp)
    return -chi2

def params_inside_priors(params):
    priors = np.array([[10,16],
                   [10,16],
                   [0,5],
                   [0,5],
                   [0,5],
                   [0,1],
                   [0,100],
                   [10,16],
                   [0,5],
                   [0,5],
                   [10,16],
                   [0,5],
                   [0,100],
                   [10,16],
                   [10,16],
                   [0,5],
                   [0,5],
                   [0,5]
    ])
    for i in range(len(params)):
        if params[i] <= priors[i][0] or params[i] >= priors[i][1]:
            return False
    return True

def plot_sampler(sampler: emcee.EnsembleSampler):
    """
    Postprocesses a sampler, displays a corner plot of its parameters, and outputs its maximum posterior values
    """
    tau = sampler.get_autocorr_time()
    print("Sampler autocorrelation time (burn in estimate):",tau, flush=True)
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

def get_target_dicts(target_dict_path, tracers=["LRG", "ELG", "QSO"]):
    """
    Returns dictionaries indexed by "LRG_LRG", "LRG_ELG", etc.
    One for the wp, one for the jackknife.
    """
    wp_dict = {}
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
                wp_dict[tr1+"_"+tr2] = wp
                jackknife_dict[tr1+"_"+tr2] = cov
    return wp_dict, jackknife_dict

def make_other_stuff_dict(boxsize, num_sat_parts, subsample_dir, sim_label):
    """
    Creates a dict with:
        boxsize
        num_sat_parts
        num_mass_bins_big
        mass_bin_centres_big
        mass_bin_edges
        hmf_big
    """

    # These must match the ones used in paircounting
    mass_bin_edges = 10**10 * np.logspace(0,6,31)
    mass_bin_centres = np.sqrt(mass_bin_edges[1:] * mass_bin_edges[:-1])

    # This is hardcoded here and can be changed
    num_mass_bins_big = 90

    mass_min = mass_bin_edges[0]
    mass_max = mass_bin_edges[-1]
    mass_bins_big = np.logspace(np.log10(mass_min),np.log10(mass_max),num_mass_bins_big + 1)
    mass_bin_centres_big = np.sqrt(mass_bins_big[1:] * mass_bins_big[:-1])

    print("Loading halos for hmf...", flush=True)
    meta_subsample_dir = Path(subsample_dir)
    full_subsample_dir = meta_subsample_dir / sim_label

    subsample_files = [full_subsample_dir / subsample_file for subsample_file in os.listdir(full_subsample_dir)]
    subsample_files.sort()
    num_subsample_files = len(subsample_files)
    if num_subsample_files == 0:
        raise Exception("No subsample files found in directory: "+str(full_subsample_dir))
    hmf_big = np.zeros(len(mass_bin_centres_big))
    for i in range(num_subsample_files):
        print("    Loading halo file",i,flush=True)
        subsample_file = subsample_files[i]
        masked_halos = h5py.File(subsample_file)
        halo_mass = masked_halos["halos"]["M200_crit"]
        halo_weights = masked_halos["halos"]["multi_halos"]

        hmf_big += np.histogram(halo_mass, bins = mass_bins_big, weights=halo_weights)[0]
        #print("Halo mass function from the files that have been loaded so far:",hmf_big)

    stuff = {}
    stuff["boxsize"] = boxsize
    stuff["num_sat_parts"] = num_sat_parts
    stuff["mass_bin_edges"] = mass_bin_edges
    stuff["mass_bin_centres_big"] = mass_bin_centres_big
    stuff["num_mass_bins_big"] = num_mass_bins_big
    stuff["hmf_big"] = hmf_big
    return stuff

def initialise_walkers(initial_params_random: bool, num_walkers):
    """
    Initialise the positions of the walkers for fitting the HOD parameters
    Do this randomly within the prior space if initial_params_random=True
    Else populate in a small region around some provided params

    Params:
    np.array([(LRGs:) logM_cut, logM1, sigma, alpha, kappa,
        (ELGs): p_max, Q, logM_cut, kappa, sigma, logM1, alpha, gamma,
        (QSOs): logM_cut, logM1, sigma, alpha, kappa])
    """
    priors = np.array([[10,16],
                   [10,16],
                   [0,5],
                   [0,5],
                   [0,5],
                   [0,1],
                   [0,100],
                   [10,16],
                   [0,5],
                   [0,5],
                   [10,16],
                   [0,5],
                   [0,100],
                   [10,16],
                   [10,16],
                   [0,5],
                   [0,5],
                   [0,5]
    ])

    mean_priors = np.array([ # Yuan et al.
        13.3,
        14.4,
        0.5,
        1.0,
        0.5,
        0.7,
        20.0,
        13.3,
        0.8,
        0.5,
        14.4,
        0.7,
        6.0,
        13.3,
        14.4,
        0.5,
        1.0,
        0.5
    ])

    std_priors = np.array([ # Yuan et al.
        0.5,
        0.5,
        0.2,
        0.3,
        0.2,
        0.5,
        0.5,
        0.2,
        0.3,
        0.5,
        0.2,
        1.0,
        0.5,
        0.5,
        0.2,
        0.3,
        0.2
    ])

    initial_params = np.array([
        13.3,
        14.4,
        0.8,
        1.0,
        0.4,
        0.7,
        20.0,
        13.3,
        0.8,
        0.5,
        14.4,
        0.7,
        6.0,
        13.3,
        14.4,
        0.8,
        1.0,
        0.4
    ])

    rng = np.random.default_rng(seed=0)

    pos = np.zeros((num_walkers,np.shape(initial_params)[0]))
    if (initial_params_random):
        for i in range(num_walkers):
            for j in range(np.shape(priors)[0]):
                #pos[i,j] = np.random.uniform(priors[j,0],priors[j,1])
                pos[i,j] = rng.normal(loc=mean_priors[j], scale=std_priors[j])

    else:
        #if len(initial_params)!=np.shape(priors)[0]:
        #    raise ValueError("Your initial parameter values and priors have different shapes")
        for i in range(num_walkers):
            # Populate in a 10% region around parameters provided
            # Potential to make the size of this region an input variable if necessary
            pos[i,:] = initial_params*(0.95 + 0.1*np.random.random(np.shape(initial_params)[0]))
    #print(pos)

    # Check none of the walkers lie outside the prior space
    #for i in range(num_walkers):
    #    for j in range(np.shape(priors)[0]):
    #        if pos[i,j] < priors[j,0]:
    #            raise ValueError("Your initial parameter values lie outside the prior space, parameter ",j, " is too low")
    #        if pos[i,j] > priors[j,1]:
    #            raise ValueError("Your initial parameter values lie outside the prior space, parameter ",j, " is too high")
    print(pos, flush=True)
    return pos