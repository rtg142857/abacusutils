import numpy as np
import yaml
import argparse
from abacusnbody.hod.fitting.hod_fitting_with_paircounting_stochopy import plot_wp, get_target_dicts, make_other_stuff_dict

def main(path_config_filename):
    # load the yaml parameters
    config = yaml.safe_load(open(path_config_filename))
    run_params = yaml.safe_load(open(config["Paths"]["params_path"]))

    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    Lp = HOD_params["LRG_params"]
    Ep = HOD_params["ELG_params"]
    Qp = HOD_params["QSO_params"]
    # HOD_params_list = [Lp["logM_cut"], Lp["logM1"], Lp["sigma"], Lp["alpha"], Lp["kappa"],
    #               Ep["p_max"], Ep["Q"], Ep["logM_cut"], Ep["kappa"], Ep["sigma"], Ep["logM1"], Ep["alpha"], Ep["gamma"],
    #               Qp["logM_cut"], Qp["logM1"], Qp["sigma"], Qp["alpha"], Qp["kappa"], Qp["p_max"]]
    HOD_params_list = [12.85, 14.1, 0.02, 1.64, 0.015,
                  0.85, 45., 10.83, 2.8, 2.39, 14.77, 0.07, 84.77,
                  13.37, 14.322, 0.74, 0.38, 4.1, 0.49]

    tracer_list = ["LRG", "ELG", "QSO"]
    clustering_params = config["clustering_params"]

    Labels = config["Labels"]
    subsample_dir = sim_params["subsample_dir"]
    sim_label = Labels["sim_label"]
    boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    ### Paths ###
    save_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/"
    paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/" + sim_label + "/"
    target_dict_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/fitting_data/Y1_z0.8-1.1/"

    print("Loading precomputed things...")
    paircounts = {}
    for pair in ["cencen", "censat", "satsat", "satsat_onehalo"]:
        for pair_type in ["", "_ELGauto", "_ELGcross"]:
            filename = paircount_path + pair + pair_type + ".npy"
            paircounts[pair+pair_type] = np.load(filename)
    target_wp, target_jackknife_inverse = get_target_dicts(target_dict_path, tracers=tracer_list)
    other_stuff_dict_here = make_other_stuff_dict(boxsize=boxsize, num_sat_parts=3, subsample_dir=subsample_dir, sim_label=sim_label)
    print("Plotting wps...", flush=True)
    plot_wp(save_path+"wps.png", hod_params=HOD_params_list, tracers=tracer_list, paircounts=paircounts,
            target_wp=target_wp, target_jackknife_inverse=target_jackknife_inverse, other_stuff_dict_here=other_stuff_dict_here, clustering_params=clustering_params)

class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass

DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'

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
