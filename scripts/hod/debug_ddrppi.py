import numpy as np
from os.path import dirname, abspath, join as pjoin
import Corrfunc
from Corrfunc.theory.DDrppi import DDrppi

Nthread = 1
binfile = np.logspace(-2, 2, 25)
#binfile = "bins.txt"

N = 100000

boxsize = 420.0
nthreads = 4
autocorr = 1
pimax = 40.0
seed = 42
np.random.seed(seed)
X = np.random.uniform(0, boxsize, N)
Y = np.random.uniform(0, boxsize, N)
Z = np.random.uniform(0, boxsize, N)

results = DDrppi(
    autocorr=autocorr,
    nthreads=Nthread,
    binfile=binfile,
    pimax=pimax,
    # binfile=rpbins,
    # pimax=pimax,
    # npibins=pimax,
    X1=X,
    Y1=Y,
    Z1=Z,
    boxsize=boxsize,
    periodic=True,
    #max_cells_per_dim=num_cells,
)
DD_counts = results['npairs']
print(f"DD_counts: {DD_counts}")