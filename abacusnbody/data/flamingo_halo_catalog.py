import numpy as np
import yaml
import h5py
import astropy.table
from astropy.table import Table
#from cosmology import CosmologyFlamingo
from abacusnbody.data.cosmology import CosmologyFlamingo

# This is a halo catalog loading module designed to 
# imitate the compaso halo catalog for Flamingo and Peregrinus.

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
    
    def compute_additional_halo_data(self, zcos):
        """
        Gets the 3d velocity dispersion, concentration, and "virial radius" (here just r200 crit)

        Use read_XXXX_file() beforehand to get id, pos, vel, M200_crit
        """
        self.halos["R200_crit"] = self.get_r200(zcos)
        self.halos["concentration"] = self.get_concentration()
        self.halos["sigmav3d"] = self.get_sigmav3d(zcos)

    def get_sigmav3d(self, zcos):
        """
        Get the 3d velocity dispersion: "i.e., the square root of the sum of eigenvalues of the second moment tensor of the velocities relative to the center of mass."

        Uses Eq. 12 of Skibba+2006 (in proper km/s) as presented in Alex Smith's HOD_Mock_Pipeline
        """
        return np.sqrt(2.151e-9 * (self.halos["mass"]*\
                          (1.+zcos)/self.halos["R200_crit"]))

    def get_concentration(self):
        """
        Returns NFW concentration of each halo, calculated from
        R200 and RVmax

        Uses the calculation given by Alex Smith's HOD_Mock_Pipeline

        Returns:
            array of halo concentrations
        """
        conc = 2.16 * self.halos["R200_crit"] / self.halos["rvmax"]

        return np.clip(conc, 0.1, 1e4)

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
        halos["pos"] = np.array(halo_cat["SO"]["200_crit"]["CentreOfMass"])[relevant_field_halos] * self.h
        halos["vel"] = np.array(halo_cat["SO"]["200_crit"]["CentreOfMassVelocity"])[relevant_field_halos]
        halos["M200_crit"] = np.array(halo_cat["SO"]["200_crit"]["DarkMatterMass"])[relevant_field_halos] * self.UnitMass_in_Msol_h
        halos["rvmax"] = np.array(halo_cat["BoundSubhalo"]["MaximumDarkMatterCircularVelocityRadius"])[relevant_field_halos] * self.h

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
        halos["pos"] = np.array(halo_cat["Subhalos"]["ComovingAveragePosition"])[relevant_field_halos] * self.h
        halos["vel"] = np.array(halo_cat["Subhalos"]["PhysicalAverageVelocity"])[relevant_field_halos]
        halos["M200_crit"] = np.array(halo_cat["Subhalos"]["BoundM200Crit"])[relevant_field_halos] * self.UnitMass_in_Msol_h
        halos["rvmax"] = np.array(halo_cat["Subhalos"]["RmaxComoving"])[relevant_field_halos] * self.h

        self.halos = halos
        #self.halos = Table(halos, copy=False)
        #self.halos.meta.update(self.header)

