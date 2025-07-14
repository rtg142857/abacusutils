from abacusnbody.hod.flamingo_hod import FlamingoHOD
import yaml
import numpy as np

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
                n_pairs.append(DDrppi(autocorr=0, nthreads=num_threads, pimax=pi_max, npibins=(pi_max//d_pi), binfile=r_bin_edges,
                         X1=samples1[i][:,0],Y1=samples1[i][:,1],Z1=samples1[i][:,2],X2=samples2[j][:,0],
                         Y2=samples2[j][:,1],Z2 = samples2[j][:,2],periodic=True,verbose=False, boxsize=boxsize))
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
                        n_pairs_mass_r_bins_wp[i,j,k,l] = n_pairs[len(samples1)*i + j][k*(pi_max//d_pi) + l][4]
                    else:
                        n_pairs_mass_r_bins_wp[i,j,k,l] = 0
    return n_pairs_mass_r_bins_wp


def count_cencen(path_config_filename, tracer_mock, Lbox, tracer1_dict, tracer2_dict=None, save=False):
    """
    Returns a list of the paircounts of central-central tracers, binned by halo mass and distance
    
    path_config_filename:
    tracer1: str e.g. "LRG"
    tracer2: str: 2nd list of tracers, for cross-correlation (optional)
    save: Boolean, whether to save to disk
    """
    config = yaml.safe_load(open(path_config_filename))
    clustering_params = config["clustering_params"]
    pi_max = clustering_params["pi_max"]
    d_pi = clustering_params["pi_bin_size"]
    rpbins = np.logspace(clustering_params["bin_params"]["logmin"], clustering_params["bin_params"]["logmax"], clustering_params["bin_params"]["nbins"]+1)

    samples_test = tracer_mock[tracer1_dict] # x, y, z, vx, vy, vz, mass, id, Ncent
    if tracer2_dict is not None:
        # TODO: make sure this is in the format expected by npairs_corrfunc, i.e. created from mass_mask
        # Also should only be the centrals
        #samples_test2 = tracer_mock[tracer2_dict]
        autocorr = False
    else:
        #samples_test2 = tracer_mock[tracer1_dict]
        autocorr = True

    num_threads = 1

    npairs_test = create_npairs_corrfunc_wp(samples_test,samples_test2,rpbins,Lbox,num_threads,pi_max,d_pi, autocorr=autocorr)
    npairs_mass_r_bins_test = npairs_conversion_wp(samples_test,samples_test2,npairs_test,rpbins,pi_max, d_pi, autocorr=autocorr) # check interface

    if save:
        np.save(run_label+"_cencen.npy",npairs_mass_r_bins_test)
    
    return npairs_mass_r_bins_test

