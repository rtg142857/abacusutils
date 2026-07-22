import numpy as np
import h5py
import os
from pathlib import Path

from abacusnbody.hod.fitting.wp_paircounting import get_wp, get_npart
from abacusnbody.hod.fitting.params import Params
from pycorr import TwoPointCorrelationFunction, twopoint_estimator

def log_probability(hod_params, paircounts, param_set: Params, target_wp_dict, target_jackknife_inverse_dict, target_ngal_dict, other_stuff_dict_here, clustering_parameters, minimise = False, wp_limit=(0, 24), verbose=False):
    tracer_list = param_set.tracer_list
    if params_inside_priors(hod_params, param_set=param_set):
        # newBall.update_HOD_params(params)
        # print(params, flush=True) # Debugging
        # mock_dict = newBall.run_hod(
        #     newBall.tracers, want_rsd=True, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=nthread, verbose=False
        # )

        # rpbins = np.logspace(clustering_parameters["bin_params"]["logmin"], clustering_parameters["bin_params"]["logmax"], clustering_parameters["bin_params"]["nbins"]+1)
        # pimax = clustering_parameters["pimax"]
        # pi_bin_size = clustering_parameters["pi_bin_size"]
        # wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=nthread)

        npart = get_npart(hod_params, param_set, other_stuff_dict_here)
        for tr in tracer_list:
            if npart[tr] == 0: # causes get_wp to throw some warnings, so we catch it early
                if minimise:
                    return np.inf
                return -np.inf
        wp_dict = get_wp(hod_params, paircounts, param_set, npart, other_stuff_dict_here, clustering_parameters)
        
        total_log_prob = 0.0

        boxsize = other_stuff_dict_here["boxsize"]

        # w_p chi squared
        for i1, tr1 in enumerate(tracer_list):
            for i2, tr2 in enumerate(tracer_list):
                if i1 <= i2: #TODO: UNDO
                    #crosscorr or autocorr
                    fitting_wp = wp_dict[tr1+"_"+tr2]
                    target_wp = target_wp_dict[tr1+"_"+tr2]
                    target_jk_inv = target_jackknife_inverse_dict[tr1+"_"+tr2]
                    total_log_prob += negative_chi_squared_wp_single_tracer_pair(fitting_wp, target_wp, target_jk_inv, wp_limit=wp_limit)
                    if verbose:
                        print(f"Log prob from {tr1}_{tr2} wp is:")
                        print(negative_chi_squared_wp_single_tracer_pair(fitting_wp, target_wp, target_jk_inv, wp_limit=wp_limit, verbose=True))
        
        # n_g chi squared
        for tracer in tracer_list:
            fitting_ngal = npart[tracer] / boxsize**3
            target_ngal = target_ngal_dict[tracer]
            total_log_prob += negative_chi_squared_ng_single_tracer(fitting_ngal, target_ngal, tracer)
            if verbose:
                print(f"Log prob from {tracer} ngal is:")
                print(negative_chi_squared_ng_single_tracer(fitting_ngal, target_ngal, tracer))

        # making sure there's only one central galaxy; this might make the n_g chi squared redundant?
        total_log_prob += negative_chi_squared_central_occupation(hod_params, param_set=param_set, other_stuff_dict_here=other_stuff_dict_here)
        if verbose:
            print(f"Log prob from central occupation is:")
            print(negative_chi_squared_central_occupation(hod_params, param_set=param_set, other_stuff_dict_here=other_stuff_dict_here))

        total_log_prob += log_prior(hod_params, param_set=param_set)
        if verbose:
            print(f"Log prob from prior is:")
            print(log_prior(hod_params, param_set=param_set))#, tracers=tracer_list, other_stuff_dict_here=other_stuff_dict_here))
    else:
        total_log_prob = -np.inf

    if np.isnan(total_log_prob): # could be due to e.g. npart being zero (would give nans in the wp)
        total_log_prob = -np.inf

    if minimise:
        total_log_prob = total_log_prob * -1

    return total_log_prob

def negative_chi_squared_ng_single_tracer(fitting_ngal, target_ngal, tracer):
    """
    Using the "rather lenient" sigma_n in Eq. 18 in https://arxiv.org/pdf/2110.11412 results in the wp dominating the chi squared
    Which is bad because it wants to push the incompleteness above 100%
    So we drop sigma_n by a factor of 100 (10?)

    Assume QSOs are complete
    """
    if fitting_ngal < target_ngal or tracer == "QSO":
        sigma_n = 4 * 10 ** (-6) #4 * 10 ** (-7) # 4 * 10 ** (-5)
        return -((fitting_ngal - target_ngal) / sigma_n) **2
    else:
        return 0

def negative_chi_squared_wp_single_tracer_pair(fitting_wp: np.ndarray, target_wp: np.ndarray, target_jackknife_inverse: np.ndarray, wp_limit=(0, 24), verbose=False):
    # Only look at a subset of the data points to improve chi squared
    # For the fitting data+variance, this is handled during setup
    # i0 = 5 # 0
    # i1 = np.size(fitting_wp)
    # mock = fitting_wp[i0:i1]
    # data = target_wp[i0:i1]
    # C_matrix = target_jackknife_inverse[i0:i1, i0:i1]
    i0, i1 = wp_limit
    mock = fitting_wp[i0:i1]
    data = target_wp
    C_matrix = target_jackknife_inverse

    temp = np.matmul(C_matrix, mock-data)
    chi2 = np.dot(mock-data, temp)
    if verbose:
        print("Difference in wp between mock and target:")
        print(mock-data)
        print("C^-1 matrix:")
        print(C_matrix)
    return -chi2

def negative_chi_squared_central_occupation(hod_params, param_set: Params, other_stuff_dict_here):
    tracers = param_set.tracer_list
    M_h = np.logspace(10, 16, 90) # doesn't need to be the same as in other cases
    hod_values = get_hod_values_given_parameters(M_h, hod_params, param_set, other_stuff_dict_here)

    cenHOD_sum = np.zeros(len(hod_values["LRG_cen"]))

    for tracer in tracers:
        cenHOD_sum += hod_values[tracer+"_cen"]
    
    amount_greater_than_1 = np.maximum(np.ones(len(cenHOD_sum)), cenHOD_sum) - 1
    chi2 = np.dot(amount_greater_than_1, amount_greater_than_1) * 10 ** 7 # guess at what works
    return -chi2

def log_prior(params, param_set: Params):
    # A has to be positive (NOW REDUNDANT with Q removed)
    #LlogM_cut, LlogM1, Lsigma, Lalpha, Lkappa, Ep_max, EQ, ElogM_cut, Ekappa, Esigma, ElogM1, Ealpha, Egamma, QlogM_cut, QlogM1, Qsigma, Qalpha, Qkappa, Qp_max = tuple(params)
    # if Ep_max - 1/EQ <= 0.0:
    #     return (Ep_max - 1/EQ) * 10 ** 7 # guess at what works
    log_prior = 0.0

    if "LRG" in param_set.tracer_list:
        # LRG kappa
        LRG_params = param_set.get_params_of_specific_tracer(params, "LRG")
        Lkappa = param_set.param_from_name(LRG_params, tracer="LRG", param_name="kappa")
        gaussian_mu = 0.5
        gaussian_sigma = 0.2
        gaussian = 1/(gaussian_sigma*(2*np.pi)**0.5) * np.exp(-(Lkappa-gaussian_mu)**2 / (2 * gaussian_sigma**2))
        log_prior += np.log(gaussian)

    return log_prior

def params_inside_priors(params, param_set: Params):
    priors = param_set.prior_bounds
    for i in range(len(params)):
        if params[i] <= priors[i][0] or params[i] >= priors[i][1]:
            return False
    return True


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
                #crosscorr or autocorr
                # path = target_dict_path + tr1 + "_" + tr2 + ".txt"
                # rpmid, rpavg, corr, std = np.loadtxt(path, unpack=True)
                # wp_dict[tr1+"_"+tr2] = corr
                # jackknife_dict[tr1+"_"+tr2] = std

                path = target_dict_path + tr1 + "_" + tr2 + ".npy"
                estimator = TwoPointCorrelationFunction.load(path)
                if estimator.shape[0] == 48:
                    estimator = estimator[:(estimator.shape[0] // 2) * 2:2] # getting it to be 24 bins
                sep, wp, cov = twopoint_estimator.project_to_wp(estimator, return_cov=True)
                wp = wp[lb:ub]
                cov = cov[lb:ub, lb:ub]
                ### FOR DEBUGGING; TODO: UNDO
                # cov = np.diag(np.diag(cov)) # Making the array diagonal-only, so no covariance between points
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

# def get_priors(type="bounds"):
#     match type:
#         case "bounds":
#             return np.array([[10,16],
#                         [10,16],
#                         [0,5],
#                         [0,5],
#                         [0,5],
#                         [0,1],
#                         [0,100],
#                         [10,16],
#                         [0,5],
#                         [0, 5],
#                         [10,16],
#                         [0,5],
#                         [0,100],
#                         [10,16],
#                         [10,16],
#                         [0,5],
#                         [0,5],
#                         [0,5],
#                         [0,1]
#             ])
#         case "mean":
#             return np.array([ # Yuan et al.
#                 13.3, # LRGs
#                 14.4,
#                 0.5,
#                 1.0,
#                 0.5,
#                 0.7, # ELGs
#                 20.0,
#                 13.3,
#                 0.8,
#                 0.5,
#                 14.4,
#                 0.7,
#                 6.0,
#                 13.3, # QSOs
#                 14.4,
#                 0.5,
#                 1.0,
#                 0.5,
#                 0.5
#             ])
#         case "std":
#             return np.array([ # Yuan et al.
#                 0.5, #LRGs
#                 0.5,
#                 0.2,
#                 0.3,
#                 0.2,
#                 0.5, # ELGs
#                 0.5,
#                 0.5,
#                 0.2,
#                 0.3,
#                 0.5,
#                 0.2,
#                 1.0,
#                 0.5, # QSOs
#                 0.5,
#                 0.2,
#                 0.3,
#                 0.2,
#                 0.5
#             ])

# def print_hod_values(hod_params):
#     print("LRG params:")
#     print("logM_cut:", hod_params[0])
#     print("logM1:", hod_params[1])
#     print("sigma:", hod_params[2])
#     print("alpha:", hod_params[3])
#     print("kappa:", hod_params[4])
#     print("ELG params:")
#     print("p_max:", hod_params[5])
#     print("Q:", hod_params[6])
#     print("logM_cut:", hod_params[7])
#     print("kappa:", hod_params[8])
#     print("sigma:", hod_params[9])
#     print("logM1:", hod_params[10])
#     print("alpha:", hod_params[11])
#     print("gamma:", hod_params[12])
#     print("QSO params:")
#     print("logM_cut:", hod_params[13])
#     print("logM1:", hod_params[14])
#     print("sigma:", hod_params[15])
#     print("alpha:", hod_params[16])
#     print("kappa:", hod_params[17])
#     print("p_max:", hod_params[18])

# def initialise_walkers(initial_params_random: bool, num_walkers):
#     """
#     Initialise the positions of the walkers for fitting the HOD parameters
#     Do this randomly within the prior space if initial_params_random=True
#     Else populate in a small region around some provided params

#     Params:
#     np.array([(LRGs:) logM_cut, logM1, sigma, alpha, kappa,
#         (ELGs): p_max, Q, logM_cut, kappa, sigma, logM1, alpha, gamma,
#         (QSOs): logM_cut, logM1, sigma, alpha, kappa])
#     """
#     priors = get_priors(type="bounds")

#     mean_priors = get_priors(type="mean")

#     std_priors = get_priors(type="std")

#     initial_params = np.array([
#         13.3,
#         14.4,
#         0.8,
#         1.0,
#         0.4,
#         0.7,
#         20.0,
#         13.3,
#         0.8,
#         0.5,
#         14.4,
#         0.7,
#         6.0,
#         13.3,
#         14.4,
#         0.8,
#         1.0,
#         0.4,
#         0.5
#     ])

#     rng = np.random.default_rng(seed=0)

#     pos = np.zeros((num_walkers,np.shape(initial_params)[0]))
#     if (initial_params_random):
#         for i in range(num_walkers):
#             for j in range(np.shape(priors)[0]):
#                 pos[i,j] = np.random.uniform(priors[j,0],priors[j,1])
#                 #pos[i,j] = rng.normal(loc=mean_priors[j], scale=std_priors[j])

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

def get_hod_values_given_parameters(M_h: np.ndarray, params: np.ndarray, param_set: Params, other_stuff_dict_here: dict) -> dict:
    """
    Returns a dict with "LRG_cen", "LRG_sat", ...
    Distinct from get_hods_given_tracer_and_params in that it calculates the incompleteness factor.
    """
    tracers = param_set.tracer_list
    target_numden = get_target_number_density(tracers)
    npart = get_npart(params, param_set=param_set, other_stuff_dict_here=other_stuff_dict_here)
    hod_dict = {}
    for tracer in tracers:
        incompleteness = target_numden[tracer] / (npart[tracer] / other_stuff_dict_here["boxsize"]**3)

        cen_hod, sat_hod = param_set.get_hods_given_tracer_and_params(M_h, params, tracer)
        #cen_hod, sat_hod = get_hods_given_tracer_and_params(M_h, params, tracer)
        # print(f"{tracer} sat_hod: {sat_hod}")
        # print(f"{tracer} incompleteness: {incompleteness}")
        hod_dict[tracer+"_cen"] = cen_hod * incompleteness
        hod_dict[tracer+"_sat"] = sat_hod * incompleteness

    return hod_dict