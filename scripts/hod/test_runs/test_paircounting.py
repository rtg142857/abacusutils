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

def Cen_HOD(params,mass_bins):
    """
    takes params: [M_cut, sigma_logm, something, something, something]
    """
    Mmin, sigma_logm = params[:2]
    result = cumulative_spline_kernel(np.log10(mass_bins), mean = Mmin, sig=sigma_logm/np.sqrt(2))
    return(result)

def Sat_HOD(params,cen_hod,mass_bins):
    """
    takes params: [Mmin, sigma_logm, logM0, logM1, alpha]
    """
    M0, M1, alpha = params[2:].copy()
    M0 = 10**M0
    M1 = 10**M1
    result = cen_hod * (((mass_bins-M0)/M1)**alpha)
    return(result)

def AbacusCen_HOD(params, mass_bins):
    """
    takes params: [logM_cut, logM1, sigma, alpha, kappa]
    """
    logM_cut = params[0]
    sigma = params[2]

    hod = np.empty(len(mass_bins))
    for i, M_h in enumerate(mass_bins):
        hod[i] = 0.5 * math.erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))

    return hod

def AbacusSat_HOD(params, cen_hod, mass_bins):
    """
    takes params: [logM_cut, logM1, sigma, alpha, kappa]
    """
    logM_cut, logM1, sigma, alpha, kappa = params[:]
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM1
    hod = np.empty(len(mass_bins))
    for i, M_h in enumerate(mass_bins):
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

    print("Making new FlamingoHOD object", flush=True)###############################################################
    # create a new FlamingoHOD object
    newBall = FlamingoHOD(path_config_filename)

    print("Getting NFW draw for satellites", flush=True)#############################################################
    max_nfw = 40
    NFW_draw = nfw_draw(10000, max_nfw, seed)

    # print("Throwaway run for jit to compile, don't write to disk", flush=True)##############################################
    # throw away run for jit to compile, don't write to disk
    # mock_dict = newBall.run_hod(
    #     newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=16, verbose=True, tabulation_mock=True
    # )

    # print("Doing paircounting...", flush=True)##################################################################
    #paircounts = paircounting.get_paircounts(path_config_filename=path_config_filename, tracer_mock = mock_dict, Nthread=16, save=True, verbose=True)

    print("Loading paircounts from the tabulation mock", flush=True)###############################################
    paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/Debugging_fitting/"
    cencen = np.load(paircount_path+"cencen.npy")
    censat = np.load(paircount_path+"censat.npy")
    satsat = np.load(paircount_path+"satsat.npy")
    satsat_onehalo = np.load(paircount_path+"satsat_onehalo.npy")

    print("Getting wp from tabulation mock", flush=True)###########################################################

    mass_bin_edges = 10**10 * np.logspace(0,6,31)
    mass_bin_centres = np.sqrt(mass_bin_edges[1:] * mass_bin_edges[:-1])

    print("   Loading halos for hmf...", flush=True)
    meta_subsample_dir = Path(subsample_dir)
    full_subsample_dir = meta_subsample_dir / sim_label

    subsample_files = [full_subsample_dir / subsample_file for subsample_file in os.listdir(full_subsample_dir)]
    subsample_files.sort()
    num_subsample_files = len(subsample_files)
    if num_subsample_files == 0:
        raise Exception("No subsample files found in directory: "+str(full_subsample_dir))
    hmf = np.zeros(len(mass_bin_centres))
    for i in range(num_subsample_files):
        print("    Loading halo",i,flush=True)
        subsample_file = subsample_files[i]
        masked_halos = h5py.File(subsample_file)
        halo_mass = masked_halos["halos"]["M200_crit"]
        hmf += np.histogram(halo_mass, bins = mass_bin_edges)[0]

    print("Done loading halos, calculating weighting factors",flush=True)
    newball_HOD_params = newBall.tracers["LRG"]
    logM_cut = newball_HOD_params["logM_cut"]
    logM1 = newball_HOD_params["logM1"]
    sigma = newball_HOD_params["sigma"]
    alpha = newball_HOD_params["alpha"]
    kappa = newball_HOD_params["kappa"]

    hod_params = [logM_cut, logM1, sigma, alpha, kappa]

    hod_cen = AbacusCen_HOD(hod_params, mass_bin_centres)
    hod_sat = AbacusSat_HOD(hod_params, hod_cen, mass_bin_centres)

    CC = create_weighting_factor(cencen,hod_cen,hod_cen)
    CS = create_weighting_factor(censat,hod_cen,hod_sat)
    SS = create_weighting_factor(satsat,hod_sat,hod_sat)
    SS1 = create_weighting_factor(satsat_onehalo,hod_sat,hod_sat)

    print("Calculating number of particles", flush=True)
    npart_cen = np.sum(hmf * hod_cen)
    npart_sat = np.sum(hmf * hod_sat)
    npart_total = npart_cen + npart_sat

    print("Calculating randoms", flush=True)
    rands = create_randoms_for_wp(npart = npart_total,r_bin_edges = rpbins,pi_max = pimax,boxsize=boxsize)
    wp_rands = np.reshape(rands,newshape=(len(rpbins)-1,pimax))

    print("Finalising wp calc", flush=True)
    GG = CC + CS + SS + SS1
    xi_pair = np.divide(GG, wp_rands) - 1
    wp_pair = xi_to_wps(xi_pair,rpbins,pimax)

    print("Getting true mock to compare wp against", flush=True)#############################################################
    mock_dict = newBall.run_hod(
        newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=False, Nthread=16, verbose=True, tabulation_mock=False
    )

    print("Getting wp from the true mock", flush=True)#############################################################
    wp_dict = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)

    print("Comparing wps", flush=True)###########################################################################

    rpcent = (rpbins[1:] + rpbins[:-1])/2
    wp_mock = wp_dict["LRG_LRG"]

    plt.loglog(rpcent, wp_mock)
    plt.loglog(rpcent, wp_pair)
    plt.legend()
    plt.savefig("fig_paircounts")
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
