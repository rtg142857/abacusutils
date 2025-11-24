from abacusnbody.hod.flamingo_hod import FlamingoHOD
import yaml
import numpy as np
import time
from pathlib import Path
import os
from Corrfunc.theory.DDrppi import DDrppi

def split_cen_sat(mock_galaxies: dict):
    """
    Splits a given mock into central and satellite galaxies.
    Also sorts the satellite arrays according to the parent halo ID.
    """
    Ncent = mock_galaxies["Ncent"]
    Mvir = mock_galaxies["mass"]
    x = mock_galaxies["x"]
    y = mock_galaxies["y"]
    z = mock_galaxies["z"]
    id = mock_galaxies["id"]
    weight = mock_galaxies["hmultis"]
    # Split into centrals and satellites

    #mask1 = np.array(is_central)
    Mvir_sat = Mvir[Ncent:]
    x_sat = x[Ncent:]
    y_sat = y[Ncent:]
    z_sat = z[Ncent:]
    weight_sat = weight[Ncent:]
    halo_id_sat = id[Ncent:]

    halo_id_sat_sorted = np.argsort(halo_id_sat)

    # Sort the satellite arrays according to the parent halo id

    Mvir_sat = Mvir_sat[halo_id_sat_sorted[:]]
    x_sat = x_sat[halo_id_sat_sorted[:]]
    y_sat = y_sat[halo_id_sat_sorted[:]]
    z_sat = z_sat[halo_id_sat_sorted[:]]
    weight_sat = weight_sat[halo_id_sat_sorted[:]]

    Mvir = Mvir[:Ncent]
    x = x[:Ncent]
    y = y[:Ncent]
    z = z[:Ncent]
    weight = weight[:Ncent]

    return x, y, z, Mvir, weight, x_sat, y_sat, z_sat, Mvir_sat, weight_sat

def mass_mask(x,y,z,weight,Mvir,mass_bin_edges):
    """
    Takes arrays of the coordinates and masses of haloes and returns a list
    where the list elements are arrays of halo coordinates and weights filtered into the 
    mass bins provided
    """
    samples_ = []
    for i in range(len(mass_bin_edges)-1):
        mass_mask = (np.array(Mvir>=mass_bin_edges[i]) 
                  & np.array(Mvir<mass_bin_edges[i+1]))
        samples_.append(np.vstack((x[mass_mask],y[mass_mask],z[mass_mask],weight[mass_mask])).T)
    return samples_

def create_npairs_corrfunc_wp(samples1,samples2,r_bin_edges,boxsize,num_threads,pi_max,d_pi=1):
    """"
    Takes two lists created from mass_mask(), radial bins, a boxsize,
    and a number of threads to use. 

    From this the number of pairs is counted between every combination
    of mass bin and radial bin

    This paircounting is achieved using corrfunc and can be multithreaded 
    with num_threads

    Returns a list which contains all the paircounts and is then reordered
    into an array in a later function: npairs_conversion
    """
    n_pairs = []
    # Iterate over the length of both mass filtered arrays
    for i in range(len(samples1)):
        for j in range(len(samples2)):
            if len(samples1[i])>=1 and len(samples2[j])>=1:
                n_pairs.append(DDrppi(autocorr=0, nthreads=num_threads, pimax=pi_max, #npibins=(pi_max//d_pi),
                         binfile=r_bin_edges,
                         X1=samples1[i][:,0],Y1=samples1[i][:,1],Z1=samples1[i][:,2], weights1=samples1[i][:,3], X2=samples2[j][:,0],
                         Y2=samples2[j][:,1],Z2 = samples2[j][:,2],weights2=samples2[j][:,3],periodic=True,verbose=False, boxsize=boxsize, weight_type="pair_product"))
                # We only use Corrfunc if both mass bins are populated, otherwise
                # return 0 for this combination
            else:
                n_pairs.append(0)
            print(i,j)
    return(n_pairs)

def npairs_conversion_wp(samples1,samples2,n_pairs,r_bin_edges,pi_max, d_pi=1):
    """
    A simple function to convert the flat list of paircounts from
    create_npairs_corrfunc into a 3D array representing the pairs
    counted in the separate mass and radial bins

    The object returned is an array which contains the paircounts
    for the ith mass bin 1, jth mass bin 2 and, kth r bin
    where we take the [i,j,k] element of the output array
    """
    n_pairs_mass_r_bins_wp = np.zeros((len(samples1),len(samples2),len(r_bin_edges)-1,(pi_max//d_pi)))
    for i in range(len(samples1)):
        for j in range(len(samples2)):
            for k in range(len(r_bin_edges)-1):
                for l in range((pi_max//d_pi)):
                    if len(samples1[i])>=1 and len(samples2[j])>=1:
                        idx_1 = len(samples1)*i + j
                        idx_2 = k*(pi_max//d_pi) + l
                        npairs_unweighted = n_pairs[idx_1][idx_2][4]
                        npairs_weightavg = n_pairs[idx_1][idx_2][5]
                        n_pairs_mass_r_bins_wp[i,j,k,l] = npairs_unweighted * npairs_weightavg
                    else:
                        n_pairs_mass_r_bins_wp[i,j,k,l] = 0
    return n_pairs_mass_r_bins_wp

def npairs_satsat_onehalo_wp(x,y,z, weights, Ms,num_sat_parts,mass_bin_edges,r_bin_edges,pi_max, d_pi=1,
                             cross=False, x2=None, y2=None, z2=None, weight2=None, Ms2=None):
    """
    We cannot use corrfunc for the one halo satellite-satellite term.
    This is because it would be too inefficient to split by
    halo id and use corrfunc on every single halo.

    We can use the fact that the satellite data here is always
    ordered according to halo id 

    We know the number of satellite tracer particles so can use 
    this to count the one halo pairs

    If calculating crosscorr, set cross=True and add the second values as x2, y2, z2 (weights and Ms are the same because it's by-halo)
    (weights and Ms are only included as arguments for sanity checking)
    """
    # Create empty arrays to hold data
    # For N sat particles we will have num_sat_parts*(num_sat_parts-1)/2 pairs per halo if doing autocorr
    # If crosscorr, it's num_sat_parts^2 pairs per halo
    if cross:
        pairs_per_halo = num_sat_parts**2
    else:
        pairs_per_halo = num_sat_parts*(num_sat_parts-1)/2
    Ms_reduced = np.zeros((len(Ms[::num_sat_parts]),
                           int(pairs_per_halo)))

    # distances in LOS and projected directions are binned
    distances_rp = np.zeros((len(Ms[::num_sat_parts]),
                          int(pairs_per_halo)))

    distances_pi = np.zeros((len(Ms[::num_sat_parts]),
                          int(pairs_per_halo)))
    
    weights_reduced = np.zeros((len(Ms[::num_sat_parts]),
                           int(pairs_per_halo)))
    
    if not cross:
        x2, y2, z2 = x, y, z
    else:
        # sanity check
        assert np.all(weights == weight2)
        assert np.all(Ms == Ms2)

    k = 0
    # For any number of satellite particles can take every combination of ith and 
    # jth satellite particle and find the distance and put them all in a big array
    for i in range(num_sat_parts):
        if cross:
            max_j_index = num_sat_parts
        else:
            max_j_index = i
        for j in range(max_j_index):
            print(k)
            Ms_reduced[:,k] = Ms[::num_sat_parts]
            distances_rp[:,k] = ((x[i::num_sat_parts]-x2[j::num_sat_parts])**2
                            + (y[i::num_sat_parts]-y2[j::num_sat_parts])**2)**0.5

            distances_pi[:,k] = ((z[i::num_sat_parts]-z2[j::num_sat_parts])**2)**0.5

            weights_reduced[:,k] = weights[::num_sat_parts]

            # This works as all halos have the same number of sat particles so 
            # [::num_sat_parts] is actually looping the specific paircount 
            # (particles i and j) over every halo
            k += 1
            print(i,j)

    # Reshape the masses and distances of the pairs so they can
    # be binned
    Ms_reduced = np.reshape(Ms_reduced,(1,-1))[0]
    distances_rp = np.reshape(distances_rp,(1,-1))[0]
    distances_pi = np.reshape(distances_pi,(1,-1))[0]
    weights_reduced = np.ravel(weights_reduced)

    final_data = np.histogramdd(sample = np.array([Ms_reduced,distances_rp,distances_pi]).T,bins=[mass_bin_edges,r_bin_edges,np.arange(0, pi_max+1, d_pi)], weights=weights_reduced)
    final_data = final_data[0]
    # Finally transform into the usual format with 2 separate M bins so that it easily fits into the rest of my existing code

    n_pairs_mass_r_bins = np.zeros((len(mass_bin_edges)-1,len(mass_bin_edges)-1,len(r_bin_edges)-1,(pi_max//d_pi)))
    for i in range(len(mass_bin_edges)-1):
        for j in range(len(r_bin_edges)-1):
            for k in range(pi_max//d_pi):
                n_pairs_mass_r_bins[i,i,j,k] = final_data[i,j,k]


    return n_pairs_mass_r_bins

def correct_doublecounting(npairs, type):
    """
    UNUSED: currently implemented after paircounting is done, to match Alex's
    """
    match type:
        case "cencen":
            return npairs/2
        case "censat":
            return npairs
        case "satsat":
            return npairs
        case "satsat_onehalo":
            return npairs/2


def count_npairs(path_config_filename, tracer_mock: dict, type, category, Nthread=1, save=False, verbose=False):
    """
    Returns a 3D Numpy array of the paircounts of tracers, binned by both halo masses and distance
    
    path_config_filename: the usual
    tracer_mock: mock dict of tracers, with one central and three satellite tracers per halo
    type: "cencen", "censat", "satsat", "satsat_onehalo"
    category: "" (LRG), "_ELGauto", "_ELGcross" (LRG-ELG)
    Nthread: Number of threads to use
    save: Boolean, whether to save to disk
    verbose: Boolean, whether to print logs to stdout
    """
    config = yaml.safe_load(open(path_config_filename))
    clustering_params = config["clustering_params"]
    pi_max = clustering_params["pimax"]
    d_pi = clustering_params["pi_bin_size"]
    rpbins = np.logspace(clustering_params["bin_params"]["logmin"], clustering_params["bin_params"]["logmax"], clustering_params["bin_params"]["nbins"]+1)
    mass_bin_edges = 10**10 * np.logspace(0,6,31)
    num_sat_parts=3

    run_params = yaml.safe_load(open(config["Paths"]["params_path"]))
    h = run_params["Cosmology"]["h"]
    Lbox = config["Params"]["L"] * h

    if verbose:
        print("Splitting centrals and satellites...")

    time0 = time.time()

    if category == "":
        tracer1 = "LRG"
        tracer2 = "LRG"
    elif category == "_ELGauto":
        tracer1 = "ELG"
        tracer2 = "ELG"
    elif category == "_ELGcross":
        tracer1 = "LRG"
        tracer2 = "ELG"

    mock1 = tracer_mock[tracer1] # x, y, z, vx, vy, vz, mass, id, Ncent
    if verbose:
        print("Total number of tracers:",len(mock1["mass"]), flush=True)

    x_cen1, y_cen1, z_cen1, M_cen1, weight_cen1, x_sat1, y_sat1, z_sat1, M_sat1, weight_sat1  = split_cen_sat(mock1)

    if verbose:
        print("Number of central tracers:",len(x_cen1), flush=True)
        print("Number of satellite tracers:", len(x_sat1), flush=True)

    if tracer2 != tracer1:
        mock2 = tracer_mock[tracer2]
        x_cen2, y_cen2, z_cen2, M_cen2, weight_cen2, x_sat2, y_sat2, z_sat2, M_sat2, weight_sat2 = split_cen_sat(mock2)
        #autocorr = False
    else:
        x_cen2, y_cen2, z_cen2, M_cen2, weight_cen2, x_sat2, y_sat2, z_sat2, M_sat2, weight_sat2 = x_cen1, y_cen1, z_cen1, M_cen1, weight_cen1, x_sat1, y_sat1, z_sat1, M_sat1, weight_sat1
        #samples_test2 = tracer_mock[tracer1_dict]
        #autocorr = True

    time1 = time.time()
    if verbose:
        print("Splitting took",time1-time0, "seconds")

    if verbose:
        print("Doing the paircounting...", flush=True)
    match type:
        case "cencen":
            samples_1 = mass_mask(x_cen1, y_cen1, z_cen1, weight_cen1, M_cen1, mass_bin_edges)
            samples_2 = mass_mask(x_cen2, y_cen2, z_cen2, weight_cen2, M_cen2, mass_bin_edges)
            num_threads = Nthread

            npairs_test = create_npairs_corrfunc_wp(samples_1,samples_2,rpbins,Lbox,num_threads,pi_max,d_pi)
            npairs_mass_r_bins_test = npairs_conversion_wp(samples_1,samples_2,npairs_test,rpbins,pi_max, d_pi)
        case "censat":
            # Only want one sat particle per halo
            for i in [x_sat2, y_sat2, z_sat2, M_sat2, weight_sat2]:
                i = i[::num_sat_parts]

            samples_1 = mass_mask(x_cen1, y_cen1, z_cen1, weight_cen1, M_cen1, mass_bin_edges)
            samples_2 = mass_mask(x_sat2, y_sat2, z_sat2, weight_sat2, M_sat2, mass_bin_edges)

            num_threads = Nthread

            npairs_test = create_npairs_corrfunc_wp(samples_1,samples_2,rpbins,Lbox,num_threads,pi_max,d_pi)
            npairs_mass_r_bins_test = npairs_conversion_wp(samples_1,samples_2,npairs_test,rpbins,pi_max, d_pi)

            if category == "_ELGcross":
                for i in [x_sat1, y_sat1, z_sat1, M_sat1, weight_sat1]:
                    i = i[::num_sat_parts]

                samples_1 = mass_mask(x_cen2, y_cen2, z_cen2, weight_cen2, M_cen2, mass_bin_edges)
                samples_2 = mass_mask(x_sat1, y_sat1, z_sat1, weight_sat1, M_sat1, mass_bin_edges)

                num_threads = Nthread

                npairs_test_cross = create_npairs_corrfunc_wp(samples_1,samples_2,rpbins,Lbox,num_threads,pi_max,d_pi)
                npairs_mass_r_bins_test_cross = npairs_conversion_wp(samples_1,samples_2,npairs_test_cross,rpbins,pi_max, d_pi)

                npairs_mass_r_bins_test += npairs_mass_r_bins_test_cross
                
        # case "satcen":
        #     samples_1 = mass_mask(x_sat2, y_sat2, z_sat2, M_sat2, mass_bin_edges)
        #     samples_2 = mass_mask(x_cen1, y_cen1, z_cen1, M_cen1, mass_bin_edges)

        #     num_threads = 1

        #     npairs_test = create_npairs_corrfunc_wp(samples_1,samples_2,rpbins,Lbox,num_threads,pi_max,d_pi)
        #     npairs_mass_r_bins_test = npairs_conversion_wp(samples_1,samples_2,npairs_test,rpbins,pi_max, d_pi)
        case "satsat":
            # Only want one sat particle per halo
            for i in [x_sat1, y_sat1, z_sat1, M_sat1, weight_sat1, x_sat2, y_sat2, z_sat2, M_sat2, weight_sat2]:
                i = i[::num_sat_parts]

            samples_1 = mass_mask(x_sat1, y_sat1, z_sat1, weight_sat1, M_sat1, mass_bin_edges)
            samples_2 = mass_mask(x_sat2, y_sat2, z_sat2, weight_sat2, M_sat2, mass_bin_edges)

            num_threads = Nthread

            npairs_test = create_npairs_corrfunc_wp(samples_1,samples_2,rpbins,Lbox,num_threads,pi_max,d_pi)
            npairs_mass_r_bins_test = npairs_conversion_wp(samples_1,samples_2,npairs_test,rpbins,pi_max, d_pi)

        case "satsat_onehalo":
            # Want all 3 sat particles per halo
            if category == "_ELGcross":
                npairs_mass_r_bins_test = npairs_satsat_onehalo_wp(x_sat1,y_sat1,z_sat1, weight_sat1, M_sat1,num_sat_parts,mass_bin_edges,rpbins,pi_max, d_pi, cross=True, x2=x_sat2, y2=y_sat2, z2=z_sat2, weight2=weight_sat2, Ms2=M_sat2)
            else:
                npairs_mass_r_bins_test = npairs_satsat_onehalo_wp(x_sat1,y_sat1,z_sat1, weight_sat1, M_sat1,num_sat_parts,mass_bin_edges,rpbins,pi_max, d_pi)
        
    time2 = time.time()
    if verbose:
        print("Paircounting took",time2-time1,"seconds")

    #npairs_mass_r_bins_test = correct_doublecounting(npairs_mass_r_bins_test, type)

    return npairs_mass_r_bins_test

def get_paircounts(path_config_filename, tracer_mock: dict, Nthread=1, save=False, verbose=False):
    """
    Returns a dict of paircounts, binned by M1, M2, and rp
    where M1 is the mass of the first halo and M2 is the mass of the second

    Args:
        path_config_filename: path to the config file
        tracer_mock: mock dict of tracers built from the newBall
        Nthread: number of threads to use
        save: Boolean, whether to save the output to a file
        verbose: Boolean, whether to print logs to stdout
    Returns:
        paircounts: Dict of paircounts, in the following format:
        paircounts["cencen"], ''["censat"], ''["satsat"], ''["satsat_onehalo"] (for LRGs and QSOs)
        paircounts["cencen_ELGauto"], ... (for ELGs specifically)
        paircounts["cencen_ELGcross"], ... (where censat counts BOTH LRG cen-ELG sat and ELG cen-LRG sat)
        Each value is a 3d numpy array of the following form:
        paircounts["cencen"][i,j,k] = number of pairs with halo 1 in mass bin i, halo 2 in mass bin j, distance in bin k


        Ignore below this line:
        paircounts["LRG_LRG"], paircounts["LRG_ELG"], etc. for each pair of tracers (autocorr and crosscorr)
        Each of the above values is itself a dict
        paircounts["LRG_LRG"] and each other autocorr dicts take the form:
        paircounts["LRG_LRG"]["cencen"], ''["censat"], ''["satsat"], ''["satsat_onehalo"]
        Crosscorr dicts also have e.g. paircounts["LRG_ELG"]["satcen"]
        Each value in _those_ dicts is a 3d numpy array of the following form:
        paircounts["LRG_ELG"]["satsat"][i, j, k] = number of pairs with halo 1 in mass bin i, halo 2 in mass bin j, distance in bin k
    """
    #tracer_list = ["LRG", "ELG", "QSO"]
    config = yaml.safe_load(open(path_config_filename))

    paircounts = {}

    # for i in range(len(tracer_list)):
    #     for j in range(i):
    #         if verbose:
    #             print(f"Paircounting tracer {tracer_1} against {tracer_2}")
    #         tracer_1 = tracer_list[i]
    #         tracer_2 = tracer_list[j]
            
    #         label = f"{tracer_1}_{tracer_2}"
    #         paircounts[label] = {}

    #         if tracer_1 == tracer_2:
    #             pair_type = "autocorr"
    #             pairs_list = ["cencen", "censat", "satsat", "satsat_onehalo"]
    #         else:
    #             pair_type = "crosscorr"
    #             pairs_list = ["cencen", "censat", "satcen", "satsat", "satsat_onehalo"]

    #         for pair in pairs_list:
    #             if verbose:
    #                 print(f"Paircounting {pair}")
    #             paircounts[label][pair] = count_npairs(path_config_filename=path_config_filename, tracer_mock=tracer_mock, type=pair, tracer1=tracer_1, tracer2=tracer_2, save=save, verbose=verbose)


    pairs_list = ["cencen", "censat", "satsat", "satsat_onehalo"]
    category_list = ["", "_ELGauto", "_ELGcross"]
    for category in category_list:
        for pair in pairs_list:
            if verbose:
                if category == "":
                    category_name = "normal"
                else:
                    category_name = category
                print(f"Paircounting {pair}, {category_name}")

            save_path = config["fitting_params"]["paircounts_save_path"] + config["Labels"]["sim_label"] + "/"
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            filename = save_path + f"{pair}{category}.npy"

            if not os.path.exists(filename):
                paircount = count_npairs(path_config_filename=path_config_filename, tracer_mock=tracer_mock, type=pair, category=category, Nthread=Nthread, save=save, verbose=verbose)
                if save:
                    np.save(filename,paircount)
                paircounts[pair+category] = paircount
            else:
                print("Paircount file exists, skipping", flush=True)
                paircounts[pair+category] = np.load(filename)

    return paircounts