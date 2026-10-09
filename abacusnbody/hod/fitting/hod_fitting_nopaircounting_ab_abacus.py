# Fits an HOD to observational data.
# Mostly borrowed from Alex Smith's FastHodFitting
# and modified to work with Flamingo and the LRG/ELG/QSO tracers.

import yaml
import numpy as np
import time

from abacusnbody.hod.abacus_hod import AbacusHOD
from stochopy.optimize import minimize
from pycorr import TwoPointCorrelationFunction, twopoint_estimator

nthread = 64 # For debugging

def fit_HOD(newBall: AbacusHOD, path_config_filename, NFW_draw, save_chains=False):
    # Initialise the fitting using emcee
    # Use a different function to actually do the fit (modularity)
    # Print to files: the updated parameters, an image of the HODs, the final fit to the wp (text and image), the errors in fitting to the wp (text and image)
    config = yaml.safe_load(open(path_config_filename))
    fitting_params = config["fitting_params"]
    tracer_list = ["LRG", "ELG", "QSO"]
    i0, i1 = 12, 20

    target_dict_path = fitting_params["target_dict_path"]
    target_wp, target_jackknife_inverse = get_target_dicts(target_dict_path, tracers=tracer_list, wp_limit=(i0, i1))
    target_ngal = get_target_number_density(tracers=tracer_list)
    nwalkers = fitting_params["nwalkers"]
    num_steps = fitting_params["num_steps"]
    boxsize = newBall.lbox
    #boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    clustering_params = config["clustering_params"]

    print("Fitting...", flush=True)

    OptimizeResult = sample_chain(newBall=newBall,
                            target_wp_dict=target_wp,
                           target_jackknife_inverse_dict=target_jackknife_inverse,
                           target_ngal_dict=target_ngal,
                           boxsize=boxsize,
                           clustering_parameters=clustering_params,
                           nwalkers=nwalkers,
                           num_steps=num_steps,
                           NFW_draw=NFW_draw,
                           wp_limit=(i0, i1))

    print("Optimization done", flush=True)
    best_fit = OptimizeResult["x"]
    print("Best params:")#, best_fit, flush=True)
    print(best_fit)
    print("Chi squared:", OptimizeResult["fun"], flush=True)
    print("Iterations:", OptimizeResult["nit"], flush=True)
    print("Successful:", OptimizeResult["success"], flush=True)
    print("Output message:", OptimizeResult["message"], flush=True)

    return

def sample_chain(newBall, target_wp_dict: dict, target_jackknife_inverse_dict: dict, target_ngal_dict: dict, boxsize, clustering_parameters: dict, nwalkers: int, num_steps: int, NFW_draw, wp_limit=(0, 24)):

    method = "cmaes"
    bounds = np.array([[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]])
    x0 = [0, 0, 0, 0, 0, 0, 0, 0]

    print("Running optimisation...", flush=True)
    OptimizeResult = minimize(log_probability, bounds, x0=x0, method=method,
                              args=(newBall, target_wp_dict, target_jackknife_inverse_dict, target_ngal_dict, boxsize, clustering_parameters, NFW_draw, wp_limit),
                              options={"maxiter": num_steps, "popsize": nwalkers, "seed": 0, "return_all": True, "workers": -1})

    return OptimizeResult

def log_probability(params, newBall: AbacusHOD, target_wp_dict, target_jackknife_inverse_dict, target_ngal_dict, boxsize, clustering_parameters, NFW_draw, wp_limit):
    if params_inside_priors(params):
        newBall.update_AB_params(params)
        #print(params, flush=True) # Debugging
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd=True, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=nthread, verbose=False
        )

        rpbins = np.logspace(clustering_parameters["bin_params"]["logmin"], clustering_parameters["bin_params"]["logmax"], clustering_parameters["bin_params"]["nbins"]+1)
        pimax = clustering_parameters["pimax"]
        pi_bin_size = clustering_parameters["pi_bin_size"]
        wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=nthread)
        
        total_log_prob = 0.0
        for i1, tr1 in enumerate(newBall.tracers.keys()):
            for i2, tr2 in enumerate(newBall.tracers.keys()):
                if i1 <= i2:
                    #crosscorr or autocorr
                    fitting_wp = wp_dict[tr1+"_"+tr2]
                    target_wp = target_wp_dict[tr1+"_"+tr2]
                    target_jk_inv = target_jackknife_inverse_dict[tr1+"_"+tr2]
                    total_log_prob += negative_chi_squared_single_tracer(fitting_wp, target_wp, target_jk_inv, wp_limit)

        for i1, tr1 in enumerate(newBall.tracers.keys()):
            ngal_dict = newBall.compute_ngal(Nthread=1)
            fitting_ngal = ngal_dict[tr1] / boxsize**3
            target_ngal = target_ngal_dict[tr1]
            total_log_prob += negative_chi_squared_ng_single_tracer(fitting_ngal, target_ngal, tr1)
    else:
        total_log_prob = -np.inf

    return total_log_prob * -1

def negative_chi_squared_single_tracer(fitting_wp: np.ndarray, target_wp: np.ndarray, target_jackknife_inverse: np.ndarray, wp_limit=(0, 24)):
    i0, i1 = wp_limit
    mock = fitting_wp[i0:i1]
    data = target_wp
    C_matrix = target_jackknife_inverse

    temp = np.matmul(C_matrix, mock-data)
    chi2 = np.dot(mock-data, temp)
    return -chi2

def negative_chi_squared_ng_single_tracer(fitting_ngal, target_ngal, tracer):
    """
    Using the "rather lenient" sigma_n in Eq. 18 in https://arxiv.org/pdf/2110.11412 results in the wp dominating the chi squared
    Which is bad because it wants to push the incompleteness above 100%
    So we drop sigma_n by a factor of 100
    """
    if fitting_ngal < target_ngal:
        sigma_n = 4 * 10 ** (-7) # 4 * 10 ** (-5)
        return -((fitting_ngal - target_ngal) / sigma_n) **2
    else:
        return 0

def params_inside_priors(params):
    priors = np.array([[-1.0,1.0],
                [-1.0,1.0],
                [-1.0,1.0],
                [-1.0,1.0],
                [-1.0,1.0],
                [-1.0,1.0],
                [-1.0,1.0],
                [-1.0,1.0]
    ])
    for i in range(len(params)):
        if params[i] <= priors[i][0] or params[i] >= priors[i][1]:
            return False
    return True

# def plot_sampler(sampler: emcee.EnsembleSampler):
#     """
#     Postprocesses a sampler, displays a corner plot of its parameters, and outputs its maximum posterior values
#     """
#     tau = sampler.get_autocorr_time()
#     print("Sampler autocorrelation time (burn in estimate):",tau, flush=True)
#     discard_value = int(3 * np.average(tau))
#     thin_value = int(0.5 * np.average(tau))
#     flat_samples = sampler.get_chain(discard=discard_value, thin=thin_value, flat=True)
#     print("Flattened sampler shape:",flat_samples.shape, flush=True)

    # import corner
    # fig = corner.corner(
    #     flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)]
    # )
    # TODO: this

# def max_like_params(sampler):
#     """Get the parameters which provide the maximum likelihood from the sampling"""

#     # Only take after the first 1000 steps to allow burn in
#     flat_samples = sampler.backend.get_chain()[:,:,:]
#     likelihoods = sampler.backend.get_log_prob()[:,:]

#     # Find the maximum likelihood parameter position

#     #print(np.shape(likelihoods))
#     best_param_index1 = np.unravel_index(np.argmax(likelihoods, axis=None), likelihoods.shape)

#     best_params = flat_samples[best_param_index1[0], best_param_index1[1], :]
#     print("best param index:",best_param_index1, flush=True)
#     print("best params:",best_params, flush=True)
#     return best_params

def get_target_dicts(target_dict_path, tracers=["LRG", "ELG", "QSO"], wp_limit=(0, 24)):
    """
    Returns dictionaries indexed by "LRG_LRG", "LRG_ELG", etc.
    One for the wp, one for the jackknife.
    """
    lb, ub=wp_limit
    wp_dict = {}
    inverse_jackknife_dict = {}
    for i1, tr1 in enumerate(tracers):
        for i2, tr2 in enumerate(tracers):
            if i1 <= i2:

                path = target_dict_path + tr1 + "_" + tr2 + ".npy"
                estimator = TwoPointCorrelationFunction.load(path)
                if estimator.shape[0] == 48:
                    estimator = estimator[:(estimator.shape[0] // 2) * 2:2] # getting it to be 24 bins
                sep, wp, cov = twopoint_estimator.project_to_wp(estimator, return_cov=True)
                wp = wp[lb:ub]
                cov = cov[lb:ub, lb:ub]
                cov_inv = np.linalg.inv(cov)
                wp_dict[tr1+"_"+tr2] = wp
                inverse_jackknife_dict[tr1+"_"+tr2] = cov_inv
    return wp_dict, inverse_jackknife_dict

def get_target_number_density(tracers=["LRG", "ELG", "QSO"], source="DESI"):
    """
    Values from eBOSS:
    See page 13 of https://arxiv.org/pdf/2110.11412
    Values from DESI dr3:
    page 10 of https://arxiv.org/pdf/2503.14738 
    """
    numden_dict = {}
    if source == "eBOSS":
        if "LRG" in tracers:
            numden_dict["LRG"] = 1 * 10**(-4)
        if "ELG" in tracers:
            numden_dict["ELG"] = 4 * 10**(-4)
        if "QSO" in tracers:
            numden_dict["QSO"] = 2 * 10**(-5)
    elif source == "DESI":
        if "LRG" in tracers:
            numden_dict["LRG"] = 4 * 10**(-4)
        if "ELG" in tracers:
            numden_dict["ELG"] = 4 * 10**(-4)
        if "QSO" in tracers:
            numden_dict["QSO"] = 2 * 10**(-5)
    return numden_dict

# def initialise_walkers(initial_params_random: bool, num_walkers):
#     """
#     Initialise the positions of the walkers for fitting the HOD parameters
#     Do this randomly within the prior space if initial_params_random=True
#     Else populate in a small region around some provided params
#     """
#     priors = np.array([[10,16],
#                    [10,16],
#                    [0,5],
#                    [0,5],
#                    [0,5],
#                    [10,16],
#                    [10,16],
#                    [0,5],
#                    [0,5],
#                    [0,5],
#                    [10,16],
#                    [10,16],
#                    [0,5],
#                    [0,5],
#                    [0,5]
#     ])

#     mean_priors = np.array([ # Yuan et al.
#         13.3,
#         14.4,
#         0.5,
#         1.0,
#         0.5,
#         13.3,
#         14.4,
#         0.5,
#         1.0,
#         0.5,
#         13.3,
#         14.4,
#         0.5,
#         1.0,
#         0.5
#     ])

#     std_priors = np.array([ # Yuan et al.
#         0.5,
#         0.5,
#         0.2,
#         0.3,
#         0.2,
#         0.5,
#         0.5,
#         0.2,
#         0.3,
#         0.2,
#         0.5,
#         0.5,
#         0.2,
#         0.3,
#         0.2
#     ])

#     initial_params = np.array([
#         13.3,
#         14.4,
#         0.8,
#         1.0,
#         0.4,
#         13.3,
#         14.4,
#         0.8,
#         1.0,
#         0.4,
#         13.3,
#         14.4,
#         0.8,
#         1.0,
#         0.4
#     ])

#     rng = np.random.default_rng(seed=0)

#     pos = np.zeros((num_walkers,np.shape(initial_params)[0]))
#     if (initial_params_random):
#         for i in range(num_walkers):
#             for j in range(np.shape(priors)[0]):
#                 #pos[i,j] = np.random.uniform(priors[j,0],priors[j,1])
#                 pos[i,j] = rng.normal(loc=mean_priors[j], scale=std_priors[j])

#     else:
#         #if len(initial_params)!=np.shape(priors)[0]:
#         #    raise ValueError("Your initial parameter values and priors have different shapes")
#         for i in range(num_walkers):
#             # Populate in a 10% region around parameters provided
#             # Potential to make the size of this region an input variable if necessary
#             pos[i,:] = initial_params*(0.95 + 0.1*np.random.random(np.shape(initial_params)[0]))
#     #print(pos)

#     # Check none of the walkers lie outside the prior space
#     #for i in range(num_walkers):
#     #    for j in range(np.shape(priors)[0]):
#     #        if pos[i,j] < priors[j,0]:
#     #            raise ValueError("Your initial parameter values lie outside the prior space, parameter ",j, " is too low")
#     #        if pos[i,j] > priors[j,1]:
#     #            raise ValueError("Your initial parameter values lie outside the prior space, parameter ",j, " is too high")
#     print(pos, flush=True)
#     return pos