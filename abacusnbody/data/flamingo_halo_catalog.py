import numpy as np
import yaml
import h5py
# import astropy.table
# from astropy.table import Table
#from cosmology import CosmologyFlamingo
from abacusnbody.data.cosmology import CosmologyFlamingo

# This is a halo catalog loading module designed to 
# imitate the compaso halo catalog for Flamingo and Peregrinus.

from scipy.optimize import fsolve

def frac_of_mass_in_radius_nfw(etavir: np.ndarray, c: np.ndarray):
    """
    etavir: fractional distance to the virial radius
    c: concentration
    """
    etavir_negative_mask = etavir < 0
    pos_mask = etavir >= 0
    etavir_pos = etavir[pos_mask]
    c_pos = c[pos_mask]
    result = np.empty(len(etavir))
    result[etavir_negative_mask] = etavir[etavir_negative_mask]
    result[pos_mask] = np.log(1 + etavir_pos * c_pos) - c_pos / (1/etavir_pos + c_pos)
    result[pos_mask] /= np.log(1 + c_pos) - c_pos/(1+c_pos)
    # num = np.log(1+etavir * c) - c/(1/etavir + c)
    # den = np.log(1+c) - c/(1+c)
    # result = num/den
    # result[etavir_negative_mask] = etavir[etavir_negative_mask]
    return result

def jacobian_fracm_nfw(etavir: np.ndarray, c: np.ndarray, matrix: bool):
    etavir_negative_mask = etavir <= 0
    pos_mask = etavir > 0
    etavir_pos = etavir[pos_mask]
    c_pos = c[pos_mask]
    result = np.empty(len(etavir))
    result[etavir_negative_mask] = 1
    result[pos_mask] = c_pos**2 * etavir_pos / (c_pos * etavir_pos + 1) **2
    result[pos_mask] /= np.log(1 + c_pos) - c_pos/(1+c_pos)
    if matrix:
        return np.diag(result)
    else:
        return result

def newton(f,Df,x0,epsilon=0.01,max_iter=10):
    '''Approximate solution of f(x)=0 by Newton's method.
    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.
    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.
    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if np.max(np.abs(fxn)) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if np.any(Dfxn) == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

class SwiftHaloCatalog(object):
    def __init__(self, path_config_filename):
        """
        Initiate halo catalogue but don't look at any halos

        Gets self.run_params (dict of the run parameters), NOT YET self.ic_params (dict of the ics), YES self.h, self.UnitMass_in_Msol_h, self.halo_type, self.cosmology
        """

        with open(path_config_filename, "r") as file:
            path_config = yaml.safe_load(file)
        self.path_config = path_config
        self.param_file_path = path_config["Paths"]["params_path"]
        self.ic_file_path = path_config["Paths"]["ics_path"]

        with open(self.param_file_path, "r") as file:
            self.run_params = yaml.safe_load(file)
        # need to use a config parser if I want the ics; see cosmology.py
        # with open(self.ic_file_path, "r") as file:
        #     self.ic_params = yaml.safe_load(file)

        self.h = self.run_params["Cosmology"]["h"]
        UnitMass_in_cgs = float(self.run_params["InternalUnitSystem"]["UnitMass_in_cgs"])
        self.UnitMass_in_Msol_h = UnitMass_in_cgs * self.h / 1.98841e33
        self.boxsize_h = self.path_config["Params"]["L"] * self.h

        self.halo_type = path_config["Misc"]["halo_type"]

        self.cosmology = CosmologyFlamingo(path_config_filename)

        self.halo_lc = False # lightcone catalogues not yet supported

    def get_halo_data(self, file_path):
        if self.halo_type == "soap":
            self.read_soap_file(file_path)
        elif self.halo_type == "peregrinus":
            self.read_peregrinus_file(file_path)
        else:
            raise Exception("Only SOAP- or Peregrinus-type halo data supported")
        
        zcos = self.path_config["Params"]["redshift"]
        self.compute_additional_halo_data(zcos)
        # self.compute_concentration_abacus_style(zcos)

    def compute_concentration_abacus_style(self):
        """
        Gets the "concentration" of the halo via the abacus method:
        c = r98_L2com / r25_L2com
        Requires the concentration to be set first (via the flamingo method)
        """
        r98_minimiser = lambda x: frac_of_mass_in_radius_nfw(x, self.halos["concentration"]) - 0.98
        r25_minimiser = lambda x: frac_of_mass_in_radius_nfw(x, self.halos["concentration"]) - 0.25
        initial_ones = np.full(np.size(self.halos["concentration"]), fill_value=1.0)
        jac = lambda x: jacobian_fracm_nfw(x, self.halos["concentration"], matrix=False)

        # r98_solver = root(r98_minimiser, x0=initial_ones*0.98, method="diagbroyden")
        # r25_solver = root(r25_minimiser, x0=initial_ones*0.25, method="diagbroyden")

        # r98_solver = fsolve(r98_minimiser, x0=initial_ones*0.98, fprime = jac, full_output=True)
        # r25_solver = fsolve(r25_minimiser, x0=initial_ones*0.25, fprime = jac, full_output=True)
        
        r98 = newton(r98_minimiser, jac, x0=initial_ones*0.98)
        r25 = newton(r25_minimiser, jac, x0=initial_ones*0.25)

        if r98 == None or r25 == None:
            raise Exception()

        # if r98_solver[2] != 1:
        #     raise Exception(r98_solver[3])
        # if r25_solver[2] != 1:
        #     raise Exception(r25_solver[3])
        
        r98_over_r25 = r98 / r25 #r98_solver[0] / r25_solver[0] # both the top and bottom are divided by rvir, so they cancel out
        self.halos["concentration_abacus"] = r98_over_r25


    def compute_additional_halo_data(self, zcos):
        """
        Gets the 3d velocity dispersion, concentration, and "virial radius" (here just r200 crit)

        Use read_XXXX_file() beforehand to get id, pos, vel, M200_crit
        """
        self.halos["R200_crit"] = self.get_r200(zcos)
        if "concentration" not in self.halos.keys():
            self.halos["concentration"] = self.get_concentration()
        self.halos["sigmav3d"] = self.get_sigmav3d(zcos)

    def get_sigmav3d(self, zcos):
        """
        Get the 3d velocity dispersion: "i.e., the square root of the sum of eigenvalues of the second moment tensor of the velocities relative to the center of mass."

        Uses Eq. 12 of Skibba+2006 (in proper km/s) as presented in Alex Smith's HOD_Mock_Pipeline
        """
        sigmav = np.sqrt(2.151e-9 * (self.halos["M200_crit"]*\
                          (1.+zcos)/self.halos["R200_crit"]))
        return sigmav/0.577

    # def get_concentration(self):
    #     """
    #     Returns NFW concentration of each halo, calculated from
    #     R200 and RVmax

    #     Uses the calculation given by Alex Smith's HOD_Mock_Pipeline

    #     WARNING: This seems to return very weird values (distribution over halos looks almost uniform up to c=80).

    #     Returns:
    #         array of halo concentrations
    #     """
    #     conc = 2.16 * self.halos["R200_crit"] / self.halos["rvmax"]

    #     return np.clip(conc, 0.1, 1e4)

    def get_r200(self, zcos, comoving=True, rho_type="crit"):
        """
        Returns R200 of each halo

        Args:
            comoving: (optional) if True convert to comoving distance
            rho_type: (optional) "mean" or "crit", default "crit"
        Returns:
            array of R200 [Mpc/h]
        """
        if rho_type == "crit":
            rho = self.cosmology.critical_density(zcos)
            r200 = (3./(800*np.pi) * self.halos["M200_crit"] / rho)**(1./3)
        elif rho_type == "mean":
            rho = self.cosmology.mean_density(zcos)
            raise Exception("Need to implement r200mean first")
        
        if comoving:
            return r200 * (1.+zcos)
        else:
            return r200
        
    def calc_hmf(self, mass_bin_edges):
        """
        Calculates the halo mass function of central halos, given a set of mass bin edges
        Returns a histogram of the halo masses given the bins
        Uses M200_crit for mass
        """
        halo_hist = np.histogram(self.halos["M200_crit"],bins = mass_bin_edges)[0]
        return halo_hist
    
    def read_soap_file(self, file_path):
        halo_cat = h5py.File(file_path, "r")

        is_not_subhalo = np.array(halo_cat["InputHalos"]["HBTplus"]["Depth"]) == 0
        
        rvmax_threshold = halo_cat["BoundSubhalo"]["MaximumDarkMatterCircularVelocityRadius"].attrs["Mask Threshold"]
        is_above_rvmax_threshold = np.array(halo_cat["SO"]["200_crit"]["NumberOfDarkMatterParticles"]) >= rvmax_threshold
        is_nonzero_rvmax = np.array(halo_cat["BoundSubhalo"]["MaximumDarkMatterCircularVelocityRadius"]) != 0

        relevant_field_halos = np.logical_and(is_above_rvmax_threshold, is_not_subhalo)
        relevant_field_halos = np.logical_and(relevant_field_halos, is_nonzero_rvmax)

        number_of_halos = np.count_nonzero(relevant_field_halos)

        halos = {}
        halos["id"] = np.array(halo_cat["InputHalos"]["HBTplus"]['TrackId'])[relevant_field_halos]
        halos["pos"] = (np.array(halo_cat["SO"]["200_crit"]["CentreOfMass"])[relevant_field_halos] * self.h) % self.boxsize_h # some values are just outside the box
        halos["vel"] = np.array(halo_cat["SO"]["200_crit"]["CentreOfMassVelocity"])[relevant_field_halos]
        halos["M200_crit"] = np.array(halo_cat["SO"]["200_crit"]["TotalMass"])[relevant_field_halos] * self.UnitMass_in_Msol_h
        #halos["rvmax"] = np.array(halo_cat["BoundSubhalo"]["MaximumDarkMatterCircularVelocityRadius"])[relevant_field_halos] * self.h
        halos["concentration"] = np.array(halo_cat["SO"]["200_crit"]["Concentration"])[relevant_field_halos]

        self.halos = halos
        #self.halos = Table(halos, copy=False)
        #self.halos.meta.update(self.header)
        # TODO: Get other stuff, possibly via Colossus?

    def read_peregrinus_file(self, file_path):
        halo_cat = h5py.File(file_path, "r")
        
        is_not_subhalo = np.array(halo_cat["Subhalos"]["Rank"]) == 0
        is_not_0mass = np.array(halo_cat["Subhalos"]["BoundM200Crit"]) != 0
        relevant_field_halos = np.logical_and(is_not_0mass, is_not_subhalo)

        halos = {}
        halos["id"] = np.array(halo_cat["Subhalos"]['TrackId'])[relevant_field_halos]
        halos["pos"] = (np.array(halo_cat["Subhalos"]["ComovingAveragePosition"])[relevant_field_halos] * self.h) % self.boxsize_h
        halos["vel"] = np.array(halo_cat["Subhalos"]["PhysicalAverageVelocity"])[relevant_field_halos]
        halos["M200_crit"] = np.array(halo_cat["Subhalos"]["BoundM200Crit"])[relevant_field_halos] * self.UnitMass_in_Msol_h
        halos["rvmax"] = np.array(halo_cat["Subhalos"]["RmaxComoving"])[relevant_field_halos] * self.h

        self.halos = halos
        #self.halos = Table(halos, copy=False)
        #self.halos.meta.update(self.header)

