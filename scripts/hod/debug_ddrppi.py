import numpy as np
from Corrfunc.theory import DDrppi

autocorr = 1
Nthread = 1
rpbins = np.logspace(-2, 2, 25)
pimax = 80
lbox = 100.0
num_cells = 30

Npart = 100
x1 = np.random.uniform(low=0.0, high=lbox, size=Npart)
y1 = np.random.uniform(low=0.0, high=lbox, size=Npart)
z1 = np.random.uniform(low=0.0, high=lbox, size=Npart)

results = DDrppi(
    autocorr=autocorr,
    nthreads=Nthread,
    binfile=rpbins,
    pimax=pimax,
    npibins=pimax,
    # binfile=rpbins,
    # pimax=pimax,
    # npibins=pimax,
    X1=x1,
    Y1=y1,
    Z1=z1,
    boxsize=lbox,
    periodic=True,
    max_cells_per_dim=num_cells,
)
DD_counts = results['npairs']
print(f"DD_counts: {DD_counts}")