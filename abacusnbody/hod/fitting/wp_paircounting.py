import numpy as np
from scipy.special import erfc, erf
from abacusnbody.hod.fitting.params import Params

def vprint(input, verbose):
    if verbose:
        print(input, flush=True)

#######################################################################################
# Number density stuff
def get_npart_given_tracer(hod_params: np.ndarray, tracer: str, param_set: Params, other_stuff_dict_here: dict):
    hmf_big = other_stuff_dict_here["hmf_big"]
    mass_bin_centres_big = other_stuff_dict_here["mass_bin_centres_big"]

    hod_cen_big, hod_sat_big = param_set.get_hods_given_tracer_and_params(M_h=mass_bin_centres_big, hod_params=hod_params, tracer=tracer) #get_hods_given_tracer_and_params(M_h=mass_bin_centres_big, hod_params=hod_params, tracer=tracer)
    if tracer == "ELG":
        hod_sat_big = param_set.get_conformity_weighted_sat_hod(mass_bin_centres_big, hod_params, tracer)

    npart_cen = np.sum(hmf_big * hod_cen_big)
    npart_sat = np.sum(hmf_big * hod_sat_big)
    npart_total = npart_cen + npart_sat
    return npart_total

def get_npart(hod_params: np.ndarray, param_set: Params, other_stuff_dict_here: dict) -> dict:
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
    tracer_list = param_set.tracer_list
    npart = {}
    for tracer in tracer_list:
        npart[tracer] = get_npart_given_tracer(hod_params=hod_params, tracer=tracer, param_set=param_set, other_stuff_dict_here=other_stuff_dict_here)
    return npart

#######################################################################################
# wp stuff

def get_accurate_tracer_HOD(hod_params: np.ndarray, tracer: str, M_h: np.ndarray, hmf_big: np.ndarray, mass_bin_edges: np.ndarray, num_mass_bins_big: int, param_set: Params) -> tuple[np.ndarray, np.ndarray]:

    hod_cen_big, hod_sat_big = param_set.get_hods_given_tracer_and_params(M_h, hod_params, tracer)

    hod_cen = create_accurate_HOD(hod_cen_big,hmf_big,mass_bin_edges,num_mass_bins_big)
    hod_sat = create_accurate_HOD(hod_sat_big,hmf_big,mass_bin_edges,num_mass_bins_big)
    return hod_cen, hod_sat

def get_accurate_ELG_sat_HOD_with_conformity(hod_params: np.ndarray, tracer: str, M_h: np.ndarray, hmf_big: np.ndarray, mass_bin_edges: np.ndarray, num_mass_bins_big: int, param_set: Params) -> tuple[np.ndarray, np.ndarray]:
    assert tracer == "ELG"
    _, hod_sat_ELGELG_big = param_set.get_hods_given_tracer_and_params(M_h, hod_params, tracer, ELG_ELG = True)
    hod_sat_weighted_big = param_set.get_conformity_weighted_sat_hod(M_h, hod_params, tracer)

    hod_sat_ELGELG = create_accurate_HOD(hod_sat_ELGELG_big, hmf_big, mass_bin_edges, num_mass_bins_big)
    hod_sat_weighted = create_accurate_HOD(hod_sat_weighted_big, hmf_big, mass_bin_edges, num_mass_bins_big)
    return hod_sat_ELGELG, hod_sat_weighted

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

def get_galaxy_pairs(tracer1: str, paircounts: dict, hod_cen1, hod_cen2, hod_sat1, hod_sat2, num_sat_parts: int, tracer2: str | None = None, verbose=False):
    if tracer1 == "ELG" and (tracer2 == None or tracer2 == "ELG"):
        ccp = paircounts["cencen_ELGauto"]
        csp = paircounts["censat_ELGauto"]
        ssp = paircounts["satsat_ELGauto"]
        ss1p = paircounts["satsat_onehalo_ELGauto"]
        if verbose:
            vprint("Sum of cc paircounts: "+str(np.sum(ccp)), verbose)
            vprint("Sum of cs paircounts: "+str(np.sum(csp)), verbose)
            vprint("Sum of ss paircounts: "+str(np.sum(ssp)), verbose)
            vprint("Sum of ss1 paircounts: "+str(np.sum(ss1p)), verbose)
        CC = create_weighting_factor(ccp,hod_cen1,hod_cen2)
        CS = create_weighting_factor(csp,hod_cen1,hod_sat2) * 2 # these paircounts are not doublecounted, but the others (including the randoms) are
        SS = create_weighting_factor(ssp,hod_sat1,hod_sat2)
        SS1 = create_weighting_factor(ss1p,hod_sat1,hod_sat2) / ((num_sat_parts*(num_sat_parts-1))/2)

    elif (tracer1 == "ELG" and tracer2 != "ELG") or (tracer2 == "ELG" and tracer1 != "ELG"):
        ccp = paircounts["cencen_ELGcross"]
        csp_lcen_esat = paircounts["censat_ELGcross"][0]
        csp_ecen_lsat = paircounts["censat_ELGcross"][1]
        ssp = paircounts["satsat_ELGcross"]
        ss1p = paircounts["satsat_onehalo_ELGcross"]
        if verbose:
            vprint("Sum of cc paircounts: "+str(np.sum(ccp)), verbose)
            vprint("Sum of cs paircounts (LRG cen, ELG sat): "+str(np.sum(csp_lcen_esat)), verbose)
            vprint("Sum of cs paircounts (ELG cen, LRG sat): "+str(np.sum(csp_ecen_lsat)), verbose)
            vprint("Sum of ss paircounts: "+str(np.sum(ssp)), verbose)
            vprint("Sum of ss1 paircounts: "+str(np.sum(ss1p)), verbose)
        CC = create_weighting_factor(ccp,hod_cen1,hod_cen2)
        if tracer1 == "ELG":
            CS_ecen_lsat = create_weighting_factor(csp_ecen_lsat,hod_cen1,hod_sat2)
            CS_lcen_esat = create_weighting_factor(csp_lcen_esat,hod_cen2,hod_sat1)
        else:
            CS_ecen_lsat = create_weighting_factor(csp_lcen_esat,hod_cen1,hod_sat2)
            CS_lcen_esat = create_weighting_factor(csp_ecen_lsat,hod_cen2,hod_sat1)
        SS = create_weighting_factor(ssp,hod_sat1,hod_sat2) #/ (num_sat_parts**2) 
        SS1 = np.zeros(shape=np.shape(SS)) #create_weighting_factor(ss1p,hod_sat1,hod_sat2) / (num_sat_parts**2)

        CS = CS_ecen_lsat + CS_lcen_esat

    else:
        ccp = paircounts["cencen"]
        csp = paircounts["censat"]
        ssp = paircounts["satsat"]
        ss1p = paircounts["satsat_onehalo"]
        if verbose:
            vprint("Sum of cc paircounts: "+str(np.sum(ccp)), verbose)
            vprint("Sum of cs paircounts: "+str(np.sum(csp)), verbose)
            vprint("Sum of ss paircounts: "+str(np.sum(ssp)), verbose)
            vprint("Sum of ss1 paircounts: "+str(np.sum(ss1p)), verbose)
        CC = create_weighting_factor(ccp,hod_cen1,hod_cen2)
        CS = create_weighting_factor(csp,hod_cen1,hod_sat2) + create_weighting_factor(csp, hod_sat1, hod_cen2) # could be LRG-QSO cross so we need both
        SS = create_weighting_factor(ssp,hod_sat1,hod_sat2)
        SS1 = create_weighting_factor(ss1p,hod_sat1,hod_sat2) / ((num_sat_parts*(num_sat_parts-1))/2)

    if verbose:
        vprint("Sum of CC after HOD integration: "+str(np.sum(CC)), verbose)
        vprint("Sum of CS after HOD integration: "+str(np.sum(CS)), verbose)
        vprint("Sum of SS after HOD integration: "+str(np.sum(SS)), verbose)
        vprint("Sum of SS1 after HOD integration: "+str(np.sum(SS1)), verbose)

    GG = CC + CS + SS + SS1
    if verbose:
        vprint(f"Sum of GG for {tracer1}, {tracer2}: "+str(np.sum(GG)), verbose)
        vprint(f"GG values:", verbose)
        vprint(GG, verbose)
    # np.save("/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/temp_stuff/pair_ddrppi", GG)

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
    #vprint(f"Sum of RR for {tracer1}, {tracer2}:", np.sum(RR_out), verbose)
    return RR_out

def xi_to_wps(xis,r_bin_edges,pi_max):
    """
    Integrate over pi bins to get wp from xi
    """
    dpi = 1
    
    wp_out = 2.0 * dpi * np.sum(xis, axis=1)
    return(wp_out)

def get_wp_given_tracer(hod_params: np.ndarray, tracer1: str, paircounts: dict, npart: dict, param_set: Params, other_stuff_dict_here: dict, clustering_params: dict, tracer2: str | None = None, verbose=False) -> np.ndarray:
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
    vprint("Getting HODs", verbose)
    if tracer2 == None or tracer2 == tracer1:
        # autocorr
        hod_cen1, hod_sat1 = get_accurate_tracer_HOD(hod_params, tracer1, mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big, param_set=param_set)
        hod_cen2, hod_sat2 = hod_cen1, hod_sat1
        if tracer1 == "ELG":
            hod_sat_ELGELG, hod_sat_weighted = get_accurate_ELG_sat_HOD_with_conformity(hod_params, tracer1, mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big, param_set)
        # else:
        #     hod_sat_ELGELG, hod_sat_weighted = None, None
    else:
        # crosscorr
        hod_cen1, hod_sat1 = get_accurate_tracer_HOD(hod_params, tracer1, mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big, param_set=param_set)
        hod_cen2, hod_sat2 = get_accurate_tracer_HOD(hod_params, tracer2, mass_bin_centres_big, hmf_big, mass_bin_edges, num_mass_bins_big, param_set=param_set)

    # Galaxy pairs
    vprint("Getting galaxy pairs", verbose)
    GG = get_galaxy_pairs(tracer1=tracer1, tracer2=tracer2, paircounts=paircounts,
                          hod_cen1=hod_cen1, hod_cen2=hod_cen2, hod_sat1=hod_sat1, hod_sat2=hod_sat2, num_sat_parts=num_sat_parts, verbose=verbose)
    if verbose:
        vprint("Sum of ggs: "+str(np.sum(GG)), verbose)

    # randoms
    vprint("Creating randoms", verbose)
    if tracer2 == None or tracer2 == tracer1:
        # autocorr
        rands = create_randoms_for_wp(npart = npart, tracer1=tracer1, r_bin_edges = rpbins,pi_max = pimax,boxsize=boxsize)
    else:
        # crosscorr
        rands = create_randoms_for_wp(npart = npart, tracer1=tracer1, tracer2=tracer2, r_bin_edges = rpbins,pi_max = pimax,boxsize=boxsize)
    wp_rands = np.reshape(rands,newshape=(len(rpbins)-1,pimax))
    if verbose:
        vprint("Sum of randoms: "+str(np.sum(wp_rands)), verbose)

    # finishing
    vprint("Getting xi", verbose)
    xi = np.divide(GG, wp_rands) - 1
    if verbose:
        vprint("Xi: "+str(xi), verbose)
    vprint("Getting wp", verbose)
    wp = xi_to_wps(xi,rpbins,pimax)
    if verbose:
        vprint("WP:, "+str(wp), verbose)
    return wp

def get_wp(hod_params: np.ndarray, paircounts: dict, param_set: Params, npart: dict, other_stuff_dict_here: dict, clustering_params: dict, verbose=False) -> dict:
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
    tracer_list = param_set.tracer_list
    wp_dict = {}

    for i in range(len(tracer_list)):
        for j in range(len(tracer_list)):
            if i > j:
                continue
            if i == j:
                # autocorr
                vprint("Doing autocorr for"+tracer_list[i], verbose)
                wp_dict[f"{tracer_list[i]}_{tracer_list[j]}"] = get_wp_given_tracer(hod_params, tracer1 = tracer_list[i], paircounts = paircounts, npart = npart, param_set=param_set, other_stuff_dict_here=other_stuff_dict_here, clustering_params=clustering_params, verbose=verbose)
            if i < j:
                # crosscorr
                vprint(f"Doing crosscorr for {tracer_list[i]}, {tracer_list[j]}", verbose)
                wp_dict[f"{tracer_list[i]}_{tracer_list[j]}"] = get_wp_given_tracer(hod_params, tracer1 = tracer_list[i], tracer2 = tracer_list[j], paircounts = paircounts, npart = npart, param_set=param_set, other_stuff_dict_here=other_stuff_dict_here, clustering_params=clustering_params, verbose=verbose)

    return wp_dict