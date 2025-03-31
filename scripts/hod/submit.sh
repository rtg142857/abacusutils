#!/bin/bash -l

#!/bin/bash
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J make_mocks #Give it something meaningful.
#SBATCH -o logs/out_makemocks
#SBATCH -e logs/err_makemocks
#SBATCH -p cosma8 #or some other partition, e.g. cosma, cosma8, etc.
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 60
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=tlrt88@durham.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

module purge
module use /cosma/apps/dp004/dc-mene1/desi/cosmodesiconda/my-desiconda/modulefiles
module load cosmodesiconda/my-desiconda
module unload Corrfunc

python run_hod.py --path_config_filename /cosma8/data/dp004/dc-mene1/abacusutils/scripts/hod/config/test_flamingo_hod.yaml