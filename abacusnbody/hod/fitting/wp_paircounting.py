import numpy as np
from scipy.special import erfc, erf

def vprint(input, verbose):
    if verbose:
        print(input, flush=True)

def N_cen_LRG(M_h: np.ndarray, logM_cut, sigma):
    """
    Standard Zheng et al. (2005) central HOD parametrization for LRGs.
    """
    return 0.5 * erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))

def N_sat_LRG_modified(M_h: np.ndarray, logM_cut, logM_1, sigma, alpha, kappa):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for LRGs, modified with n_cent_LRG
    """
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM_1
    below_cut = M_h - kappa * M_cut < 0
    hod_value = (
        ((M_h - kappa * M_cut) / M_1) ** alpha
        * 0.5
        * erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))
    )
    hod_value[below_cut] = 0
    return hod_value


def N_cen_ELG_v1(M_h: np.ndarray, p_max, Q, logM_cut, sigma, gamma, Anorm=1):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    logM_h = np.log10(M_h)
    phi = phi_fun(logM_h, logM_cut, sigma)
    Phi = Phi_fun(logM_h, logM_cut, sigma, gamma)
    return (
        2.0 * (p_max - 1.0 / Q) * phi * Phi / Anorm
    )  # + 0.5/Q*(1 + math.erf((logM_h-logM_cut-0.8)*3))

def N_sat_ELG(M_h, logM_cut, kappa, logM_1, alpha, A_s=1.0, alpha1=0.0, beta=0.0):
    """
    Standard power law modulated by an exponential fall off at small M
    """
    # return (M_h/M_1)**alpha/(1+np.exp(-A_s*(np.log10(M_h)-np.log10(kappa*M_cut)))) + beta*(M_h/M_1)**(-alpha1)/100
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM_1
    below_cut = M_h - kappa * M_cut < 0
    
    hod_value = A_s * ((M_h - kappa * M_cut) / M_1) ** alpha # + beta*(M_h/M_1)**(-alpha1)/100
    hod_value[below_cut] = 0
    return hod_value

def N_cen_QSO(M_h, logM_cut, sigma):
    """
    HOD function (Zheng et al. (2005) with p_max) for QSO centrals taken from arXiv:2007.09012.
    """
    return 0.5 * (1 + erf((np.log10(M_h) - logM_cut) / 1.41421356 / sigma))

def N_sat_QSO(M_h, logM_cut, kappa, logM_1, alpha, A_s=1.0):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for all tracers with an optional amplitude parameter, A_s.
    """
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM_1
    below_cut = M_h - kappa * M_cut < 0
    hod_value = A_s * ((M_h - kappa * M_cut) / M_1) ** alpha
    hod_value[below_cut] = 0
    return hod_value

def phi_fun(logM_h, logM_cut, sigma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    phi = Gaussian_fun(logM_h, logM_cut, sigma)
    return phi

def Phi_fun(logM_h, logM_cut, sigma, gamma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    x = gamma * (logM_h - logM_cut) / sigma
    Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    return Phi

def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return 0.3989422804014327 / sigma * np.exp(-((x - mean) ** 2) / 2 / sigma**2)

def get_hods_given_tracer_and_params(M_h: np.ndarray, hod_params: np.ndarray, tracer: str):
    match tracer:
        case "LRG":
            logM_cut, logM1, sigma, alpha, kappa = tuple(hod_params[:5])
            hod_cen = N_cen_LRG(M_h, logM_cut, sigma)
            hod_sat = N_sat_LRG_modified(M_h, logM_cut, logM1, sigma, alpha, kappa)
        case "ELG":
            p_max, Q, logM_cut, kappa, sigma, logM1, alpha, gamma = tuple(hod_params[5:13])
            hod_cen = N_cen_ELG_v1(M_h, p_max, Q, logM_cut, sigma, gamma)
            hod_sat = N_sat_ELG(M_h, logM_cut, kappa, logM1, alpha)
        case "QSO":
            logM_cut, logM1, sigma, alpha, kappa = tuple(hod_params[13:18])
            hod_cen = N_cen_QSO(M_h, logM_cut, sigma)
            hod_sat = N_sat_QSO(M_h, logM_cut, kappa, logM1, alpha)
    return hod_cen, hod_sat

#######################################################################################
# Number density stuff
def get_npart_given_tracer(hod_params, tracer, other_stuff_dict_here):
    hmf_big = other_stuff_dict_here["hmf_big"]
    mass_bin_centres_big = other_stuff_dict_here["mass_bin_centres_big"]

    hod_cen_big, hod_sat_big = get_hods_given_tracer_and_params(M_h=mass_bin_centres_big, hod_params=hod_params, tracer=tracer)

    npart_cen = np.sum(hmf_big * hod_cen_big)
    npart_sat = np.sum(hmf_big * hod_sat_big)
    npart_total = npart_cen + npart_sat
    return npart_total

def get_npart(hod_params: np.ndarray, tracer_list: list, other_stuff_dict_here: dict) -> dict:
    """
    Returns a dict of the number of particles
    Keys: LRG, ELG, etc.

    hod_params = np.array([(LRGs:) logM_cut, logM1, sigma, alpha, kappa, (ELGs): p_max, Q, logM_cut, kappa, sigma, logM1, alpha, gamma, (QSOs): logM_cut, logM1, sigma, alpha, kappa])
    tracer_list: ["LRG", "ELG", "QSO"] probably
    other_stuff_dict_here: dict of:
        boxsize
        num_sat_parts
        num_mass_bins_big
        mass_bin_centres_big
        mass_bin_edges
        hmf_big
    """
    npart = {}
    for tracer in tracer_list:
        npart[tracer] = get_npart_given_tracer(hod_params=hod_params, tracer=tracer, other_stuff_dict_here=other_stuff_dict_here)
    return npart

#######################################################################################
# wp stuff

def get_accurate_tracer_HOD(hod_params: np.ndarray, tracer: str, M_h: np.ndarray, hmf_big: np.ndarray, mass_bin_edges: np.ndarray, num_mass_bins_big: int) -> tuple[np.ndarray, np.ndarray]:

    hod_cen_big, hod_sat_big = get_hods_given_tracer_and_params(M_h, hod_params, tracer)

    hod_cen = create_accurate_HOD(hod_cen_big,hmf_big,mass_bin_edges,num_mass_bins_big)
    hod_sat = create_accurate_HOD(hod_sat_big,hmf_big,mass_bin_edges,num_mass_bins_big)
    return hod_cen, hod_sat

def create_accurate_HOD(hod,halos,mass_bin_edges,num_mass_bins_big):
    """
    Using just 30 mass bins for the HOD isn't accurate enough. Take smaller
    mass subdivisions and use these to create an accurate HOD for only 30 mass bins.
    """
    num_mass_bins_small = int(len(mass_bin_edges) -1)
    mass_bins_factor = int(num_mass_bins_big/num_mass_bins_small)
    if num_mass_bins_big % num_mass_bins_small != 0:
        raise ValueError("finer grained mass bins do not evenly divide coarser mass bins:",num_mass_bins_small," is not a factor of ",num_mass_bins_big)
    hod[np.isnan(hod)] = 0
    HOD_halo_product = halos * hod
    HOD_recalc = np.sum(np.reshape(HOD_halo_product,(num_mass_bins_small,mass_bins_factor)),axis=1) / (
                 np.sum(np.reshape(halos,(num_mass_bins_small,mass_bins_factor)),axis=1))
    HOD_recalc[np.isnan(HOD_recalc)] = 0
    return HOD_recalc

def create_weighting_factor(mass_pair_array,hod1,hod2):
    """
    Multiply the array by the relevant HODs and then sum over the mass bins
    to get the number of pairs as a function of r. These can then be divided
    by the randoms to get the correlation function.
    """
    weighting_factor = np.tensordot(np.outer(hod1,hod2),mass_pair_array,axes=([0,1],[0,1]))
    return weighting_factor

def get_galaxy_pairs(tracer1: str, paircounts: dict, hod_cen1, hod_cen2, hod_sat1, hod_sat2, num_sat_parts: int, tracer2: str | None = None):
    if tracer1 == "ELG" and (tracer2 == None or tracer2 == "ELG"):
        CC = create_weighting_factor(paircounts["cencen_ELGauto"],hod_cen1,hod_cen2)
        CS = create_weighting_factor(paircounts["censat_ELGauto"],hod_cen1,hod_sat2) * 2 / num_sat_parts # these paircounts are not doublecounted, but the others (including the randoms) are
        SS = create_weighting_factor(paircounts["satsat_ELGauto"],hod_sat1,hod_sat2) / num_sat_parts**2
        SS1 = create_weighting_factor(paircounts["satsat_onehalo_ELGauto"],hod_sat1,hod_sat2) / ((num_sat_parts*(num_sat_parts-1))/2)

    elif (tracer1 == "ELG" and tracer2 != "ELG") or (tracer2 == "ELG" and tracer1 != "ELG"): # none of these should be doublecounted, and yet final result seems to be double, but only at low rp?
        CC = create_weighting_factor(paircounts["cencen_ELGcross"],hod_cen1,hod_cen2)
        CS = create_weighting_factor(paircounts["censat_ELGcross"],hod_cen1,hod_sat2) / num_sat_parts
        SS = create_weighting_factor(paircounts["satsat_ELGcross"],hod_sat1,hod_sat2) / (num_sat_parts**2) 
        SS1 = create_weighting_factor(paircounts["satsat_onehalo_ELGcross"],hod_sat1,hod_sat2) / (num_sat_parts**2)

    else:
        CC = create_weighting_factor(paircounts["cencen"],hod_cen1,hod_cen2)
        CS = create_weighting_factor(paircounts["censat"],hod_cen1,hod_sat2) * 2 / num_sat_parts # these paircounts are not doublecounted, but the others (including the randoms) are
        SS = create_weighting_factor(paircounts["satsat"],hod_sat1,hod_sat2) / num_sat_parts**2
        SS1 = create_weighting_factor(paircounts["satsat_onehalo"],hod_sat1,hod_sat2) / ((num_sat_parts*(num_sat_parts-1))/2)

    GG = CC + CS + SS + SS1
    print(f"Sum of GG for {tracer1}, {tracer2}:", np.sum(GG))

    return CC + CS + SS + SS1



def create_randoms_for_wp(npart, tracer1, r_bin_edges,pi_max,boxsize, tracer2=None):
    """
    Calculate the analytic randoms for npart particles in a box with
    side length boxsize.  This code is based on the calculation done 
    either in corrfunc but the formula is pretty simple.
    """
    #pis = np.arange(1,pi_max+1)
    RR_out = np.zeros((len(r_bin_edges)-1)*pi_max)
    for p in range(pi_max):
        # do volume calculations
        v = 2*np.pi*r_bin_edges**2 # Volume of cylinders

        dv = np.diff(v)  # difference between r volumes
        
        global_volume = boxsize**3  # volume of simulation

        # calculate the random-random pairs using density * volume
        if tracer2 == tracer1 or tracer2 == None:
            # autocorr
            rhor = (npart[tracer1]*(npart[tracer1]-1))/global_volume # these are doublecounted
        else:
            # crosscorr
            rhor = (npart[tracer1]*npart[tracer2]) /global_volume
        RR = (dv*rhor)
        #print(RR)
        RR_out[p::pi_max] = RR
    print(f"Sum of RR for {tracer1}, {tracer2}:", np.sum(RR_out))
    return RR_out

def xi_to_wps(xis,r_bin_edges,pi_max):
    """
    Integrate over pi bins to get wp from xi
    """
    dpi = 1
    
    wp_out = 2.0 * dpi * np.sum(xis, axis=1)
    return(wp_out)

def get_wp_given_tracer(hod_params: np.ndarray, tracer1: str, paircounts: dict, npart: dict, other_stuff_dict_here: dict, clustering_params: dict, tracer2: str | None = None) -> np.ndarray:
    # getting important values from the given dicts
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']

    boxsize = other_stuff_dict_here["boxsize"]
    num_sat_parts = other_stuff_dict_here["num_sat_parts"]
    num_mass_bins_big = other_stuff_dict_here["num_mass_bins_big"]
    mass_bin_centres_big = other_stuff_dict_here["mass_bin_centres_big"]
    mass_bin_edges = other_stuff_dict_here["mass_bin_edges"]
    hmf_big = other_stuff_dict_here["hmf_big"]

    # Getting HODs
    if tracer2 == None or tracer2 == tracer1:
        # autocorr
        hod_cen1, hod_sat1 = get_accurate_tracer_HOD(hod_params, tracer1, mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big)
        hod_cen2, hod_sat2 = hod_cen1, hod_sat1
    else:
        # crosscorr
        hod_cen1, hod_sat1 = get_accurate_tracer_HOD(hod_params, tracer1, mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big)
        hod_cen2, hod_sat2 = get_accurate_tracer_HOD(hod_params, tracer2, mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big)

    # Galaxy pairs
    GG = get_galaxy_pairs(tracer1=tracer1, tracer2=tracer2, paircounts=paircounts,
                          hod_cen1=hod_cen1, hod_cen2=hod_cen2, hod_sat1=hod_sat1, hod_sat2=hod_sat2, num_sat_parts=num_sat_parts)

    # randoms
    if tracer2 == None or tracer2 == tracer1:
        # autocorr
        rands = create_randoms_for_wp(npart = npart, tracer1=tracer1, r_bin_edges = rpbins,pi_max = pimax,boxsize=boxsize)
    else:
        # crosscorr
        rands = create_randoms_for_wp(npart = npart, tracer1=tracer1, tracer2=tracer2, r_bin_edges = rpbins,pi_max = pimax,boxsize=boxsize)
    wp_rands = np.reshape(rands,newshape=(len(rpbins)-1,pimax))

    # finishing
    xi = np.divide(GG, wp_rands) - 1
    wp = xi_to_wps(xi,rpbins,pimax)
    return wp

def get_wp(hod_params: np.ndarray, paircounts: dict, tracer_list: list, npart: dict, other_stuff_dict_here: dict, clustering_params: dict, verbose=False) -> dict:
    """
    Returns a dict of wp autocorr and crosscorr
    Keys: LRG_LRG, LRG_ELG, etc.
    Values: np.array of wp

    hod_params = np.array([(LRGs:) logM_cut, logM1, sigma, alpha, kappa, (ELGs): p_max, Q, logM_cut, kappa, sigma, logM1, alpha, gamma, (QSOs): logM_cut, logM1, sigma, alpha, kappa])
    paircounts: dict of cencen, censat, satsat, satsat_onehalo (output of paircounting.py)
    tracer_list: ["LRG", "ELG", "QSO"] probably
    npart_dict: one for each tracer, labelled LRG, ELG, QSO as appropriate
    other_stuff_dict_here: dict of:
        boxsize
        num_sat_parts
        num_mass_bins_big
        mass_bin_centres_big
        mass_bin_edges
        hmf_big
    clustering_params: dict of:
        bin_params: {logmin, logmax, nbins}
        rpbins
        pimax
        pi_bin_size (currently hardcoded to 1 tbqh)
    Verbose: bool
    """
    wp_dict = {}

    for i in range(len(tracer_list)):
        for j in range(len(tracer_list)):
            if i > j:
                continue
            if i == j:
                # autocorr
                vprint("Doing autocorr for"+tracer_list[i], verbose)
                wp_dict[f"{tracer_list[i]}_{tracer_list[j]}"] = get_wp_given_tracer(hod_params, tracer1 = tracer_list[i], paircounts = paircounts, npart = npart, other_stuff_dict_here=other_stuff_dict_here, clustering_params=clustering_params, verbose=verbose)
            if i < j:
                # crosscorr
                vprint(f"Doing crosscorr for {tracer_list[i]}, {tracer_list[j]}", verbose)
                wp_dict[f"{tracer_list[i]}_{tracer_list[j]}"] = get_wp_given_tracer(hod_params, tracer1 = tracer_list[i], tracer2 = tracer_list[j], paircounts = paircounts, npart = npart, other_stuff_dict_here=other_stuff_dict_here, clustering_params=clustering_params, verbose=verbose)

    return wp_dict