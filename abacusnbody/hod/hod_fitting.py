# Fits an HOD to observational data.
# Mostly borrowed from Alex Smith's FastHodFitting
# and modified to work with Flamingo and the LRG/ELG/QSO tracers.

import yaml
import numpy as np
import time

from flamingo_hod import FlamingoHOD
import emcee

def fit_HOD(newBall: FlamingoHOD, path_config_filename, print_to_files: bool):
    # Initialise the fitting using emcee
    # Use a different function to actually do the fit (modularity)
    # Print to files: the updated parameters, an image of the HODs, the final fit to the wp (text and image), the errors in fitting to the wp (text and image)
    config = yaml.safe_load(open(path_config_filename))
    fitting_params = config["fitting_params"]

    # target_wp = {}
    # target_jackknife = {}

    # Get this from the fitting_params
    # nwalkers = 100
    # ndim = 15

    start_time = time.time()
    sampler = sample_chain(newBall=newBall,
                           target_wp_dict=target_wp,
                           target_jackknife_dict=target_jackknife,
                           clustering_parameters=clustering_params,
                           NFW_draw=NFW_draw,
                           nwalkers=nwalkers,
                           num_steps=num_steps,
                           ndim=ndim) # TODO: fill in
    end_time = time.time()
    print("fitting took ", end_time - start_time, " seconds")

    # Print the parameters at the end of the chain to check they are reasonable
    print("Parameters at the end of the fitting chain: (These aren't best fits, just a sanity check)")
    print(sampler.backend.get_chain()[-1,0])

    print("All done!")
    print("Saving outputs")

    plot_sampler(sampler)

    return max_like_params(sampler)

def sample_chain(newBall: FlamingoHOD, target_wp_dict: dict, target_jackknife_dict: dict, clustering_parameters: dict, NFW_draw: np.ndarray, nwalkers: int, num_steps: int, ndim=15):

    walker_init_pos = initialise_walkers(initial_params_random=False,num_walkers=nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(newBall, target_wp_dict, target_jackknife_dict, clustering_parameters, NFW_draw))#, pool=pool)
    sampler.run_mcmc(walker_init_pos, num_steps)
    return sampler

def log_probability(params, newBall: FlamingoHOD, target_wp_dict, target_jackknife_dict, clustering_parameters, NFW_draw):
    newBall.update_HOD_params(params)
    mock_dict = newBall.run_hod(
        newBall.tracers, want_rsd=True, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=16, verbose=False
    )

    rpbins = np.logspace(clustering_parameters["bin_params"]["logmin"], clustering_parameters["bin_params"]["logmax"], clustering_parameters["bin_params"]["nbins"]+1)
    pimax = clustering_parameters["pimax"]
    pi_bin_size = clustering_parameters["pi_bin_size"]
    wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size)
    
    total_log_prob = 0.0
    for i1, tr1 in enumerate(newBall.tracers.keys()):
        for i2, tr2 in enumerate(newBall.tracers.keys()):
            if i1 <= i2:
                #crosscorr or autocorr
                fitting_wp = wp_dict[tr1+"_"+tr2]
                target_wp = target_wp_dict[tr1+"_"+tr2]
                target_jk = target_jackknife_dict[tr1+"_"+tr2]
                total_log_prob += negative_chi_squared_single_tracer(fitting_wp, target_wp, target_jk)

    return total_log_prob

def negative_chi_squared_single_tracer(fitting_wp: np.ndarray, target_wp: np.ndarray, target_jackknife: np.ndarray):
    return -np.sum(((target_wp - fitting_wp)/target_jackknife)**2)

def plot_sampler(sampler: emcee.EnsembleSampler):
    """
    Postprocesses a sampler, displays a corner plot of its parameters, and outputs its maximum posterior values
    """
    tau = sampler.get_autocorr_time()
    print("Sampler autocorrelation time (burn in estimate):",tau)
    discard_value = int(3 * np.average(tau))
    thin_value = int(0.5 * np.average(tau))
    flat_samples = sampler.get_chain(discard=discard_value, thin=thin_value, flat=True)
    print("Flattened sampler shape:",flat_samples.shape)

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

    best_params = flat_samples[best_param_index1,:]
    print("best param index:",best_param_index1)
    print("best params:",best_params)
    return best_params

def initialise_walkers(initial_params_random: bool, num_walkers):
    """
    Initialise the positions of the walkers for fitting the HOD parameters
    Do this randomly within the prior space if initial_params_random=True
    Else populate in a small region around some provided params
    """
    priors = np.array([[10,16],
                   [10,16],
                   [0,5],
                   [0,5],
                   [0,5],
                   [10,16],
                   [10,16],
                   [0,5],
                   [0,5],
                   [0,5],
                   [10,16],
                   [10,16],
                   [0,5],
                   [0,5],
                   [0,5]
    ])

    initial_params = np.array([
        13.3,
        14.4,
        0.8,
        1.0,
        0.4,
        13.3,
        14.4,
        0.8,
        1.0,
        0.4,
        13.3,
        14.4,
        0.8,
        1.0,
        0.4
    ])

    pos = np.zeros((num_walkers,np.shape(initial_params)[0]))
    if (initial_params_random):
        for i in range(num_walkers):
            for j in range(np.shape(priors)[0]):
                pos[i,j] = np.random.uniform(priors[j,0],priors[j,1])

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
    print(pos)
    return pos