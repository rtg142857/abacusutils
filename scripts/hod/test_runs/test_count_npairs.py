import numpy as np

import abacusnbody.hod.fitting.paircounting as paircounting

def main():
    path_config_filename = "../config/test_same_r_dist_fixed_rng.yaml"

    tracer_mock = {}
    tracer_mock["LRG"] = {}
    fields = ["x", "y", "z", "vx", "vy", "vz", "id", "hmultis", "mass"]

    cen1 = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1, 1.0, 10**13]
    cen2 = [10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3, 1.0, 10**13]

    sat_11 = [0.0, 1.1, 1.0, 0.0, 0.0, 0.0, 1, 1.0, 10**13]
    sat_12 = [0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 1, 1.0, 10**13]
    sat_13 = [0.05, 1.0, 1.1, 0.0, 0.0, 0.0, 1, 1.0, 10**13]

    sat_21 = [10.0, 1.1, 1.0, 0.0, 0.0, 0.0, 3, 1.0, 10**13]
    sat_22 = [10.0, 0.9, 1.0, 0.0, 0.0, 0.0, 3, 1.0, 10**13]
    sat_23 = [10.05, 1.0, 1.1, 0.0, 0.0, 0.0, 3, 1.0, 10**13]

    tracers = zip(cen1, cen2, sat_11, sat_12, sat_13, sat_21, sat_22, sat_23)
    for i, field in enumerate(fields):
        tracer_mock["LRG"][field] = np.array(tracers[i])
    tracer_mock["LRG"]["Ncent"] = 2
    print("Test set of tracers:")
    print(tracer_mock["LRG"])

    paircounts = {}
    pairs_list = ["cencen", "censat", "satsat", "satsat_onehalo"]
    for pair in pairs_list:
        paircounts[pair] = paircounting.count_npairs(path_config_filename=path_config_filename, tracer_mock=tracer_mock, type=pair, category="", Nthread=1, save=False, verbose=True)
    
    return pairs_list