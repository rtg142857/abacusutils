#!/usr/bin/env python3
"""
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
"""

import argparse
import time

import numpy as np
import yaml

from abacusnbody.hod.flamingo_hod import FlamingoHOD
from abacusnbody.hod.NFW import nfw_draw

DEFAULTS = {}
DEFAULTS['path_config_filename'] = 'config/abacus_hod.yaml'


def main(path_config_filename):
    # load the yaml parameters
    config = yaml.load(open(path_config_filename))
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

    print("Making new FlamingoHOD object")
    # create a new FlamingoHOD object
    #newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    newBall = FlamingoHOD(path_config_filename)

    print("Getting NFW draw for satellites")
    max_nfw = 40
    NFW_draw = nfw_draw(10000, max_nfw, seed)

    print("Throwaway run for jit to compile, write to disk")
    # throw away run for jit to compile, write to disk
    mock_dict = newBall.run_hod(
        newBall.tracers, want_rsd, want_nfw=True, NFW_draw=NFW_draw, write_to_disk=write_to_disk, Nthread=16
    )
    # mock_dict = newBall.gal_reader()
    start = time.time()
    newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
    print('Done xi, total time ', time.time() - start)
    # print(xirppi)
    # wp = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size)
    # print(wp)

    print("Running the fit 10 times for timing...")
    # run the fit 10 times for timing
    meantime = 0
    Ntest = 20
    for i in range(Ntest):
        print(i)
        # # run hod, ngal, xirppi
        newBall.tracers['LRG']['alpha'] += 0.01
        print("alpha = ",newBall.tracers['LRG']['alpha'])
        start = time.time()
        mock_dict = newBall.run_hod(
            newBall.tracers, want_rsd, write_to_disk=False, Nthread=64
        )
        print('Done hod, took time ', time.time() - start)
        start = time.time()
        ngal_dict = newBall.compute_ngal()
        print("Done ngal, took time ", time.time() - start, ngal_dict)
        newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread=32)
        deltat = time.time() - start
        print('Done xi, total time ', deltat)
        meantime += deltat
    print('meantime ', meantime / Ntest)


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
