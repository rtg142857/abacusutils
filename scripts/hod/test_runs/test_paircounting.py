#!/usr/bin/env python3
"""
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
"""

import argparse
import time
import os
import math
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

import h5py

from abacusnbody.hod.flamingo_hod import FlamingoHOD
from abacusnbody.hod.NFW import nfw_draw
from abacusnbody.hod.fitting.hod_fitting import fit_HOD
import abacusnbody.hod.fitting.paircounting as paircounting

DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'

def spline_kernel_integral(x):
    """
    Returns the integral of the unscaled spline kernel function from -1 to x
    """
    if hasattr(x, "__len__"):
        # x in an array
        integral = np.zeros(len(x))
        absx = abs(x)
        ind = absx < 0.5
        integral[ind] = absx[ind] - 2*absx[ind]**3 + 1.5*absx[ind]**4
        ind = np.logical_and(absx >= 0.5, absx < 1.)
        integral[ind] = 0.375 - 0.5*(1-absx[ind])**4
        ind = absx >= 1.
        integral[ind] = 0.375
        ind = x < 0
        integral[ind] = -integral[ind]
    else:
        # x is a number
        absx = abs(x)
        if   absx < 0.5: integral = absx - 2*absx**3 + 1.5*absx**4
        elif absx < 1:   integral = 0.375 - 0.5*(1-absx)**4
        else:            integral = 0.375
        if x < 0: integral = -integral
    return integral

def cumulative_spline_kernel(x, mean=0, sig=1):
    """
    Returns the integral of the rescaled spline kernel function from -inf to x.
    The spline kernel is rescaled to have the specified mean and standard
    deviation, and is normalized.
    """
    integral = spline_kernel_integral((x-mean)/(sig*np.sqrt(12))) / 0.75
    y = 0.5 * (1. + 2*integral)
    return y

# Now change Cen_HOD definition

# def Cen_HOD(params,mass_bins):
#     """
#     takes params: [M_cut, sigma_logm, something, something, something]
#     """
#     Mmin, sigma_logm = params[:2]
#     result = cumulative_spline_kernel(np.log10(mass_bins), mean = Mmin, sig=sigma_logm/np.sqrt(2))
#     return(result)

# def Sat_HOD(params,cen_hod,mass_bins):
#     """
#     takes params: [Mmin, sigma_logm, logM0, logM1, alpha]
#     """
#     M0, M1, alpha = params[2:].copy()
#     M0 = 10**M0
#     M1 = 10**M1
#     result = cen_hod * (((mass_bins-M0)/M1)**alpha)
#     return(result)

def AbacusCen_HOD(params, masses):
    """
    takes params: [logM_cut, logM1, sigma, alpha, kappa]
    """
    logM_cut = params[0]
    sigma = params[2]

    hod = np.empty(len(masses))
    for i, M_h in enumerate(masses):
        hod[i] = 0.5 * math.erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))

    return hod

def AbacusSat_HOD(params, cen_hod, masses):
    """
    takes params: [logM_cut, logM1, sigma, alpha, kappa]
    """
    logM_cut, logM1, sigma, alpha, kappa = params[:]
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM1
    hod = np.empty(len(masses))
    for i, M_h in enumerate(masses):
            if M_h - kappa * M_cut < 0:
                hod[i] = 0
            else:
                hod[i] = ((M_h - kappa * M_cut) / M_1) ** alpha * cen_hod[i]
    return hod

def create_weighting_factor(mass_pair_array,hod1,hod2):
    """
    Multiply the array by the relevant HODs and then sum over the mass bins
    to get the number of pairs as a function of r. These can then be divided
    by the randoms to get the correlation function.
    """
    weighting_factor = np.tensordot(np.outer(hod1,hod2),mass_pair_array,axes=([0,1],[0,1]))
    return weighting_factor

def create_randoms_for_wp(npart,r_bin_edges,pi_max,boxsize):
    """
    Calculate the analytic randoms for npart particles in a box with
    side length boxsize.  This code is based on the calculation done 
    either in corrfunc but the formula is pretty simple.
    """
    NR = npart
    pis = np.arange(1,pi_max+1)
    RR_out = np.zeros((len(r_bin_edges)-1)*pi_max)
    for p in range(pi_max):
        # do volume calculations
        v = 2*np.pi*r_bin_edges**2 # Volume of cylinders

        dv = np.diff(v)  # difference between r volumes
        
        global_volume = boxsize**3  # volume of simulation

        # calculate the random-random pairs using density * volume
        rhor = (NR*(NR-1))/global_volume
        RR = (dv*rhor)
        #print(RR)
        RR_out[p::pi_max] = RR
    return RR_out

def xi_to_wps(xis,r_bin_edges,pi_max):
    """
    Integrate over pi bins to get wp from xi
    """
    dpi = 1
    
    wp_out = 2.0 * dpi * np.sum(xis, axis=1)
    return(wp_out)

def create_accurate_HOD(hod,halos,mass_bin_edges,num_mass_bins_big):
    """
    Using just 100-400 mass bins for the HOD isn't accurate enough. Take much smaller
    mass subdivisions and use these to create an accurate HOD for only 100-400 mass bins.
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


def main(path_config_filename):
    # load the yaml parameters
    config = yaml.safe_load(open(path_config_filename))
    run_params = yaml.safe_load(open(config["Paths"]["params_path"]))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    Paths = config["Paths"]
    Labels = config["Labels"]
    Params = config["Params"]
    Misc = config["Misc"]
    seed = Misc["random_seed"]

    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(
        bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1
    )
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    subsample_dir = sim_params["subsample_dir"]
    sim_label = Labels["sim_label"]

    temp_stuff = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/temp_stuff/"
    mock_wp_exists = os.path.isfile(temp_stuff + "mock_wp.npy")
    pair_wp_exists = os.path.isfile(temp_stuff + "pair_wp.npy")

    paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/Debugging_fitting/"
    paircounts_exist = os.path.isfile(paircount_path + "cencen.npy")
    if not mock_wp_exists or not pair_wp_exists or not paircounts_exist:

        print("Making new FlamingoHOD object", flush=True)###############################################################
        # create a new FlamingoHOD object
        newBall = FlamingoHOD(path_config_filename)

        print("Getting NFW draw for satellites", flush=True)#############################################################
        max_nfw = 40
        NFW_draw = nfw_draw(10000, max_nfw, seed)

    if not paircounts_exist:

        print("Get mock dict by running HOD; change write to disk as necessary", flush=True)##############################################
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=True, Nthread=16, verbose=True, tabulation_mock=True
        )

        print("Doing paircounting...", flush=True)##################################################################
        paircounts = paircounting.get_paircounts(path_config_filename=path_config_filename, tracer_mock = mock_dict, Nthread=16, save=True, verbose=True)

    if not pair_wp_exists:

        print("Loading paircounts from the tabulation mock", flush=True)###############################################
        paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/Debugging_fitting/"
        cencen = np.load(paircount_path+"cencen.npy")
        censat = np.load(paircount_path+"censat.npy")
        satsat = np.load(paircount_path+"satsat.npy")
        satsat_onehalo = np.load(paircount_path+"satsat_onehalo.npy")
        num_sat_parts = 3

        print("Getting wp from tabulation mock", flush=True)###########################################################

        mass_bin_edges = 10**10 * np.logspace(0,6,31)
        mass_bin_centres = np.sqrt(mass_bin_edges[1:] * mass_bin_edges[:-1])
        num_mass_bins_big = 90
        mass_min = mass_bin_edges[0]
        mass_max = mass_bin_edges[-1]
        # Large number of sub bins for accuracy
        mass_bins_big = np.logspace(np.log10(mass_min),np.log10(mass_max),num_mass_bins_big + 1)
        mass_bin_centres_big = np.sqrt(mass_bins_big[1:] * mass_bins_big[:-1])
        print("Mass bin edges:",mass_bin_edges)
        print("Mass bin centres:",mass_bin_centres)

        print("   Loading halos for hmf...", flush=True)
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
            print("Halo mass function from the files that have been loaded so far:",hmf_big)

        print("Total number of halos:", np.sum(hmf_big), flush=True)
        print("Done loading halos, calculating weighting factors",flush=True)
        newball_HOD_params = newBall.tracers["LRG"]
        logM_cut = newball_HOD_params["logM_cut"]
        logM1 = newball_HOD_params["logM1"]
        sigma = newball_HOD_params["sigma"]
        alpha = newball_HOD_params["alpha"]
        kappa = newball_HOD_params["kappa"]

        hod_params = [logM_cut, logM1, sigma, alpha, kappa]
        print("HOD parameters (logmcut, logm1, sigma, alpha, kappa):",hod_params)

        hod_cen_big = AbacusCen_HOD(hod_params, mass_bin_centres_big)
        
        hod_sat_big = AbacusSat_HOD(hod_params, hod_cen_big, mass_bin_centres_big)
        
        hod_cen = create_accurate_HOD(hod_cen_big,hmf_big,mass_bin_edges,num_mass_bins_big)
        hod_sat = create_accurate_HOD(hod_sat_big,hmf_big,mass_bin_edges,num_mass_bins_big)

        np.save(temp_stuff + "hod_cen_big.npy", hod_cen_big)
        np.save(temp_stuff + "hod_sat_big.npy", hod_sat_big)
        np.save(temp_stuff + "hod_cen.npy", hod_cen)
        np.save(temp_stuff + "hod_sat.npy", hod_sat)
        print("Central HOD:",hod_cen)
        print("Satellite HOD:", hod_sat)

        CC = create_weighting_factor(cencen,hod_cen,hod_cen)
        print("CC weight factor:", CC)
        CS = create_weighting_factor(censat,hod_cen,hod_sat) #* 2 # not doublecounted, but the others (including the randoms) are
        print("CS weighting factor:", CS)
        SS = create_weighting_factor(satsat,hod_sat,hod_sat)
        print("SS weighting factor:", SS)
        SS1 = create_weighting_factor(satsat_onehalo,hod_sat,hod_sat) / ((num_sat_parts*(num_sat_parts-1))/2) 
        print("SS1 weighting factor:", SS1)

        print("Calculating number of particles", flush=True)
        npart_cen = np.sum(hmf_big * hod_cen_big)
        npart_sat = np.sum(hmf_big * hod_sat_big)
        npart_total = npart_cen + npart_sat
        print(f"Central particles: {npart_cen}, satellite particles: {npart_sat}, total particles: {npart_total}")

        print("Calculating randoms", flush=True)
        rands = create_randoms_for_wp(npart = npart_total,r_bin_edges = rpbins,pi_max = pimax,boxsize=boxsize)
        print("Randoms (unshaped):", rands)
        wp_rands = np.reshape(rands,newshape=(len(rpbins)-1,pimax))
        print("Randoms (reshaped:)", wp_rands)

        print("Finalising wp calc", flush=True)
        GG = CC + CS + SS + SS1
        print("Total GG pairs:", GG)
        xi_pair = np.divide(GG, wp_rands) - 1
        print("Xi from paircounting:", xi_pair)
        np.save(temp_stuff + "pair_xi.npy", xi_pair)
        wp_pair = xi_to_wps(xi_pair,rpbins,pimax)
        print("WP from paircounting:", wp_pair)
        np.save(temp_stuff + "pair_wp.npy", wp_pair)
    else:
        print("pair_wp already saved, skipping")
        wp_pair = np.load(temp_stuff + "pair_wp.npy")


    if not mock_wp_exists:
        print("Getting true mock to compare wp against", flush=True)#############################################################
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=16, verbose=True, tabulation_mock=False
        )
        np.save(temp_stuff + "internal_hmf.npy", newBall.halo_mass_func)
        print("Calculating number of galaxies from the in-built function now", flush=True)
        ngal_dict = newBall.compute_ngal(Nthread=1)
        print("Number of LRGs according to compute_ngal:", ngal_dict)

        print("Getting wp from the true mock", flush=True)#############################################################
        wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
        wp_mock = wp_dict["LRG_LRG"]
        np.save(temp_stuff + "mock_wp.npy", wp_mock)
        print("Getting xi from the true mock", flush=True)##############################################################
        xi_dict = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
        xi_mock = xi_dict["LRG_LRG"]
        np.save(temp_stuff + "mock_xi.npy", xi_mock)
    else:
        print("mock_wp already saved, skipping")
        wp_mock = np.load(temp_stuff + "mock_wp.npy")

    print("Plotting", flush=True)###########################################################################

    rpcent = np.sqrt(rpbins[1:] * rpbins[:-1])

    plt.loglog(rpcent, wp_mock, label="'True' wp from mock")
    plt.loglog(rpcent, wp_pair, label="wp from paircounting")
    plt.legend()
    plt.title("wp(rp)")
    plt.savefig("fig_wprp")
    plt.show()

    plt.close()

    plt.loglog(mass_bin_centres_big, hmf_big)
    plt.title("HMF")
    plt.savefig("fig_hmf")
    plt.show()
    
    plt.close()

    plt.loglog(mass_bin_centres_big, hod_cen_big, label="Central HOD")
    plt.loglog(mass_bin_centres_big, hod_sat_big, label="Satellite HOD")
    plt.loglog(mass_bin_centres_big, hod_cen_big + hod_sat_big, label="Total HOD")
    plt.ylim(bottom=10**-3)
    plt.title("HOD")
    plt.legend()
    plt.savefig("fig_hods")
    plt.show()

    plt.close()

    plt.loglog(mass_bin_centres_big, hmf_big * hod_cen_big + hmf_big * hod_sat_big, label="Paircounted")
    plt.title("Halo mass function, weighted by HOD")
    plt.legend()
    plt.savefig("fig_hmf_weighted")
    plt.show()

class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


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
