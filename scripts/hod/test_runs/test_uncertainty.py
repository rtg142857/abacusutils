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
    HOD_params_list = [12.8, 14.0, 0.1, 0.78, 0.63,
                  0.53, 10., 12.3, 1., 0.58, 13.53, 0.9, 4.12,
                  0.33, 12.21, 1.0, 0.56, 13.94, 0.4]

    tracer_list = ["LRG", "ELG", "QSO"]
    clustering_params = config["clustering_params"]

    Labels = config["Labels"]
    subsample_dir = sim_params["subsample_dir"]
    sim_label = Labels["sim_label"]
    boxsize = config["Params"]["L"] * run_params["Cosmology"]["h"]

    ### Paths ###
    save_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/samplers/"
    paircount_path = "/cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/output/paircounts/"
    target_dict_path = config["fitting_params"]["target_dict_path"]

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
