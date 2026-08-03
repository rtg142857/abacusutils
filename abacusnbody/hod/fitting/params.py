import numpy as np
from scipy.special import erfc, erf

# include:
# structure for the params, including their associated tracer, bounds, mean, and std
# functions for setting them up
# HOD functions for taking in a list of values???
# a changeable function which chooses which params to use

# WARNING! Assumes dictionaries in python are ORDERED. Does not work with python <3.7!

def N_cen_LRG(M_h: np.ndarray, logM_cut, sigma):
    """
    Standard Zheng et al. (2005) central HOD parametrization for LRGs.
    """
    return 0.5 * erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma)) * 0 # for debugging; TODO: UNDO

def N_sat_LRG_modified(M_h: np.ndarray, logM_cut, logM_1, sigma, alpha, kappa):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for LRGs, modified with n_cent_LRG
    """
    # mass_bin_edges = 10**10 * np.logspace(0,6,31)
    # mass_lower = mass_bin_edges[9]
    # mass_upper = mass_bin_edges[10]
    # below_cut = M_h - mass_lower < 0
    # above_cut = M_h - mass_upper > 0
    # hod_value = np.ones(np.size(M_h))
    # hod_value[below_cut] = 0
    # hod_value[above_cut] = 0
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM_1
    below_cut = M_h - kappa * M_cut < 0
    hod_value = (
        ((M_h - kappa * M_cut) / M_1) ** alpha # might warn about an invalid value if M_h - kappa * M_cut < 0, but that's handled elsewhere
        * 0.5
        * erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))
    )
    hod_value[below_cut] = 0
    return hod_value

def N_cen_ELG_v1(M_h: np.ndarray, p_max, Q, logM_cut, sigma, gamma, Anorm=1):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    logM_h = np.log10(M_h)
    phi = phi_fun(logM_h, logM_cut, sigma)
    Phi = Phi_fun(logM_h, logM_cut, sigma, gamma)
    return ((
        2.0 * (p_max - 1.0 / Q) * phi * Phi / Anorm
    ) + 0.5/Q*(1 + erf((logM_h-logM_cut)/0.01)))
    # + 0.5/Q*(1 + math.erf((logM_h-logM_cut-0.8)*3))

def N_sat_ELG(M_h, logM_cut, kappa, logM_1, alpha, A_s=1.0, alpha1=0.0, beta=0.0):
    """
    Standard power law modulated by an exponential fall off at small M
    """
    # return (M_h/M_1)**alpha/(1+np.exp(-A_s*(np.log10(M_h)-np.log10(kappa*M_cut)))) + beta*(M_h/M_1)**(-alpha1)/100
    # mass_bin_edges = 10**10 * np.logspace(0,6,31)
    # mass_lower = mass_bin_edges[9]
    # mass_upper = mass_bin_edges[10]
    # below_cut = M_h - mass_lower < 0
    # above_cut = M_h - mass_upper > 0
    # hod_value = np.ones(np.size(M_h))
    # hod_value[below_cut] = 0
    # hod_value[above_cut] = 0
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM_1
    below_cut = M_h - kappa * M_cut < 0
    
    hod_value = A_s * ((M_h - kappa * M_cut) / M_1) ** alpha # + beta*(M_h/M_1)**(-alpha1)/100 # might warn about an invalid value if M_h - kappa * M_cut < 0, but that's handled elsewhere
    hod_value[below_cut] = 0
    return hod_value

def N_cen_QSO(M_h, logM_cut, sigma, p_max):
    """
    HOD function (Zheng et al. (2005) with p_max) for QSO centrals taken from arXiv:2007.09012.
    """
    # mass_bin_edges = 10**10 * np.logspace(0,6,31)
    # mass_lower = mass_bin_edges[10]
    # mass_upper = mass_bin_edges[30]
    # below_cut = M_h - mass_lower < 0
    # above_cut = M_h - mass_upper > 0
    # hod_value = np.ones(np.size(M_h))
    # hod_value[below_cut] = 0
    # hod_value[above_cut] = 0
    # return hod_value
    return p_max * 0.5 * (1 + erf((np.log10(M_h) - logM_cut) / 1.41421356 / sigma)) # For debugging; TODO: UNDO

def N_sat_QSO(M_h, logM_cut, kappa, logM_1, alpha, A_s=1.0):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for all tracers with an optional amplitude parameter, A_s.
    """
    M_cut = 10 ** logM_cut
    M_1 = 10 ** logM_1
    below_cut = M_h - kappa * M_cut < 0
    hod_value = A_s * ((M_h - kappa * M_cut) / M_1) ** alpha
    hod_value[below_cut] = 0
    return hod_value

def phi_fun(logM_h, logM_cut, sigma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    phi = Gaussian_fun(logM_h, logM_cut, sigma)
    return phi

def Phi_fun(logM_h, logM_cut, sigma, gamma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    x = gamma * (logM_h - logM_cut) / sigma
    Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    return Phi

def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return 0.3989422804014327 / sigma * np.exp(-((x - mean) ** 2) / 2 / sigma**2)

# def get_hods_given_tracer_and_params(M_h: np.ndarray, hod_params: np.ndarray, tracer: str):
    # match tracer:
    #     case "LRG":
    #         logM_cut, logM1, sigma, alpha, kappa = tuple(hod_params[:5])
    #         hod_cen = N_cen_LRG(M_h, logM_cut, sigma)
    #         hod_sat = N_sat_LRG_modified(M_h, logM_cut, logM1, sigma, alpha, kappa)
    #     case "ELG":
    #         p_max, Q, logM_cut, kappa, sigma, logM1, alpha, gamma = tuple(hod_params[5:13])
    #         hod_cen = N_cen_ELG_v1(M_h, p_max, Q, logM_cut, sigma, gamma)
    #         hod_sat = N_sat_ELG(M_h, logM_cut, kappa, logM1, alpha)
    #     case "QSO":
    #         logM_cut, logM1, sigma, alpha, kappa, p_max = tuple(hod_params[13:19])
    #         hod_cen = N_cen_QSO(M_h, logM_cut, sigma, p_max)
    #         hod_sat = N_sat_QSO(M_h, logM_cut, kappa, logM1, alpha)
    # return hod_cen, hod_sat

class SingleParam:
    def __init__(self, name, idx, lb, ub, mean, std):
        # Index is only for tracers; so QSO logM-cut, logM1, sigma, alpha, kappa would still be 0 to 4
        # Mean and std are just for initialising walkers, not priors (std isn't actually used atm but it might be in future)
        self.name = name
        self.idx = idx
        self.lb = lb
        self.ub = ub
        self.mean = mean
        self.std = std

class Params:
    """
    Class for bookkeeping the list of parameters. If you want to remove or add params, change the init function.

    Also stores the current HOD (initialised with update_hod) and M_h.
    """
    def __init__(self, tracer_list: list[str]):#, M_h: np.ndarray):
        self.tracer_list = tracer_list
        #self.M_h = M_h
        self.tracer_dict = {}
        self.prior_bounds = []
        for tracer in tracer_list:
            match tracer:
                case "LRG":
                    self.tracer_dict["LRG"] = {
                        "logM_cut": SingleParam(name="logM_cut", idx=0, lb=10, ub=16, mean=13.3, std=0.5),
                        "logM1": SingleParam(name="logM1", idx=1, lb=10, ub=16, mean=14.4, std=0.5),
                        "sigma": SingleParam(name="sigma", idx=2, lb=0, ub=5, mean=0.5, std=0.2),
                        "alpha": SingleParam(name="alpha", idx=3, lb=0, ub=5, mean=1.0, std=0.3),
                        "kappa": SingleParam(name="kappa", idx=4, lb=0, ub=5, mean=0.5, std=0.2)
                    }
                case "ELG":
                    self.tracer_dict["ELG"] = {
                        "p_max": SingleParam(name="p_max", idx=0, lb=0, ub=1, mean=0.7, std=0.5),
                        #"Q": SingleParam(name="logM1", idx=1, lb=0, ub=100, mean=20, std=5),
                        "logM_cut": SingleParam(name="logM_cut", idx=1, lb=10, ub=16, mean=13.3, std=0.5),
                        "kappa": SingleParam(name="kappa", idx=2, lb=0, ub=5, mean=0.8, std=0.2),
                        "sigma": SingleParam(name="sigma", idx=3, lb=0, ub=5, mean=0.5, std=0.2),
                        "logM1": SingleParam(name="logM1", idx=4, lb=10, ub=16, mean=14.4, std=0.5),
                        "logM1_EE": SingleParam(name="logM1_EE", idx=5, lb=10, ub=16, mean=14.2, std=0.5),
                        "alpha": SingleParam(name="alpha", idx=6, lb=0, ub=5, mean=1.0, std=0.3),
                        "gamma": SingleParam(name="gamma", idx=7, lb=0, ub=100, mean=6.0, std=1.0)
                    }
                case "QSO":
                    self.tracer_dict["QSO"] = {
                        "logM_cut": SingleParam(name="logM_cut", idx=0, lb=10, ub=16, mean=13.3, std=0.5),
                        "logM1": SingleParam(name="logM1", idx=1, lb=10, ub=16, mean=14.4, std=0.5),
                        "sigma": SingleParam(name="sigma", idx=2, lb=0, ub=5, mean=0.5, std=0.2),
                        "alpha": SingleParam(name="alpha", idx=3, lb=0, ub=5, mean=1.0, std=0.3),
                        "kappa": SingleParam(name="kappa", idx=4, lb=0, ub=5, mean=0.5, std=0.2),
                        "p_max": SingleParam(name="p_max", idx=5, lb=0, ub=5, mean=0.5, std=0.2)
                    }
            for val in self.tracer_dict[tracer].values():
                self.prior_bounds.append([val.lb, val.ub])
        
        # self.hod_dict = {}
        # for tracer in tracer_list:
        #     self.hod_dict[tracer] = {}
        #     self.hod_dict[tracer]["cen"] = np.zeros(len(M_h))
        #     self.hod_dict[tracer]["sat"] = np.zeros(len(M_h))

        # # Conformity
        # if "ELG" in tracer_list:
        #     self.hod_dict["ELG_sat_EE"] = np.zeros(len(M_h))
        #     self.hod_dict["ELG_sat_avg"] = np.zeros(len(M_h))
        #     self.hod_dict["ELG_sat_ss1"] = np.zeros(len(M_h))
    
    def get_initial_params(self, positions = 1) -> np.ndarray:
        """
        positions: number of initial positions; if 1, puts them right on the mean; otherwise, puts them randomly in a gaussian with the params' mean and std
        returns a 1d array if positions = 1, 2d array if it's anything else
        """
        if positions == 1:
            x0 = []
            for tracer in self.tracer_list:
                for param in self.tracer_dict[tracer].values():
                    # use the fact that dictionaries in python 3.7+ are ordereed
                    x0.append(param.mean)
        
        else:
            rng = np.random.default_rng(seed=0)
            x0 = []
            for tracer in self.tracer_list:
                for param in self.tracer_dict[tracer].values():
                    x0.append(rng.uniform(low=param.lb, high=param.ub, size=positions))
        return np.array(x0)
    
    def param_from_name(self, params_of_tracer, tracer, param_name, default=None):
        if param_name in self.tracer_dict[tracer].keys():
            return params_of_tracer[self.tracer_dict[tracer][param_name].idx]
        else:
            return default
        
    def get_params_of_specific_tracer(self, hod_params: np.ndarray, tracer: str):
        """
        Given the str of the tracer you want and a list of HOD params, slices out the tuple of params corresponding to that tracer.
        """
        if tracer == self.tracer_list[0]:
            param_array_lower_idx = 0
            param_array_upper_idx = len(self.tracer_dict[tracer])
        elif tracer == self.tracer_list[1]:
            param_array_lower_idx = len(self.tracer_dict[self.tracer_list[0]])
            param_array_upper_idx = len(self.tracer_dict[self.tracer_list[0]]) + len(self.tracer_dict[self.tracer_list[1]])
        elif tracer == self.tracer_list[2]:
            param_array_lower_idx = len(self.tracer_dict[self.tracer_list[0]]) + len(self.tracer_dict[self.tracer_list[1]])
            param_array_upper_idx = len(self.tracer_dict[self.tracer_list[0]]) + len(self.tracer_dict[self.tracer_list[1]]) + len(self.tracer_dict[self.tracer_list[2]])
        else:
            raise Exception
        
        return tuple(hod_params[param_array_lower_idx:param_array_upper_idx])

    
    def get_hods_given_tracer_and_params(self, M_h: np.ndarray, hod_params: np.ndarray, tracer: str, ELG_ELG: bool = False):
        # First get the range of indices corresponding to the particular tracer; ELG_ELG indicates that we are getting the satellite HOD for an ELG with central ELG

        params_of_tracer = self.get_params_of_specific_tracer(hod_params, tracer)

        match tracer:
            case "LRG":
                logM_cut = self.param_from_name(params_of_tracer, tracer, "logM_cut")
                logM1 = self.param_from_name(params_of_tracer, tracer, "logM1")
                sigma = self.param_from_name(params_of_tracer, tracer, "sigma")
                alpha = self.param_from_name(params_of_tracer, tracer, "alpha")
                kappa = self.param_from_name(params_of_tracer, tracer, "kappa")
                hod_cen = N_cen_LRG(M_h, logM_cut, sigma)
                hod_sat = N_sat_LRG_modified(M_h, logM_cut, logM1, sigma, alpha, kappa)
            case "ELG":
                p_max = self.param_from_name(params_of_tracer, tracer, "p_max")
                Q = self.param_from_name(params_of_tracer, tracer, "Q", default=np.inf)
                logM_cut = self.param_from_name(params_of_tracer, tracer, "logM_cut")
                sigma = self.param_from_name(params_of_tracer, tracer, "sigma")
                gamma = self.param_from_name(params_of_tracer, tracer, "gamma")
                kappa = self.param_from_name(params_of_tracer, tracer, "kappa")

                alpha = self.param_from_name(params_of_tracer, tracer, "alpha")

                hod_cen = N_cen_ELG_v1(M_h, p_max, Q, logM_cut, sigma, gamma)
                if ELG_ELG:
                    logM1_EE = self.param_from_name(params_of_tracer, tracer, "logM1_EE")
                    hod_sat = N_sat_ELG(M_h, logM_cut, kappa, logM1_EE, alpha)
                else:
                    logM1 = self.param_from_name(params_of_tracer, tracer, "logM1")
                    hod_sat = N_sat_ELG(M_h, logM_cut, kappa, logM1, alpha)
            case "QSO":
                logM_cut = self.param_from_name(params_of_tracer, tracer, "logM_cut")
                logM1 = self.param_from_name(params_of_tracer, tracer, "logM1")
                sigma = self.param_from_name(params_of_tracer, tracer, "sigma")
                alpha = self.param_from_name(params_of_tracer, tracer, "alpha")
                kappa = self.param_from_name(params_of_tracer, tracer, "kappa")
                p_max = self.param_from_name(params_of_tracer, tracer, "p_max", default=1)
                hod_cen = N_cen_QSO(M_h, logM_cut, sigma, p_max)
                hod_sat = N_sat_QSO(M_h, logM_cut, kappa, logM1, alpha)
        return hod_cen, hod_sat
    
    def get_conformity_weighted_sat_hod(self, M_h: np.ndarray, hod_params: np.ndarray, tracer: str = "ELG") -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the satellite HOD for ELGs, taken as an average of that with an LRG+QSO central (no conformity) and that with an ELG central (with fconformity).
        Average is weighted in each mass bin by the proportion of central galaxies

        Returns a tuple: the ordinary average, and the value for ELG-ELG one-halo.
        """
        assert tracer=="ELG"
        #print("Debugging: Finding conformity weighted sat HODs")
        ELG_hod_cen, ELG_hod_sat_noconform = self.get_hods_given_tracer_and_params(M_h, hod_params, tracer="ELG", ELG_ELG=False)
        #print(f"ELG HOD cen:\n{ELG_hod_cen}")
        #print(f"ELG HOD sat baseline:\n{ELG_hod_sat_noconform}")
        _, ELG_hod_sat_conform = self.get_hods_given_tracer_and_params(M_h, hod_params, tracer="ELG", ELG_ELG=True)
        #print(f"ELG HOD sat with conformity (SHOULD BE THE SAME AS BASELINE):\n{ELG_hod_sat_conform}")

        QSO_params = self.get_params_of_specific_tracer(hod_params, "QSO")
        LRG_hod_cen_unweighted, _ = self.get_hods_given_tracer_and_params(M_h, hod_params, tracer="LRG")
        LRG_hod_cen = LRG_hod_cen_unweighted * (1 - self.param_from_name(QSO_params, "QSO", "p_max", default=0)) # incompleteness factor for LRGs
        QSO_hod_cen, _ = self.get_hods_given_tracer_and_params(M_h, hod_params, tracer="QSO")
        
        nonELG_hod_cen = LRG_hod_cen + QSO_hod_cen
        #print(f"Sum of non-ELG hod cens (SHOULD BE 1.0 at high end):\n{nonELG_hod_cen}")
        ELG_cen_ratio = np.nan_to_num(ELG_hod_cen / (nonELG_hod_cen + ELG_hod_cen))
        #print(f"ELG cen ratio:\n{ELG_cen_ratio}")
        ELG_hod_sat_avg = ELG_hod_sat_conform * ELG_cen_ratio + ELG_hod_sat_noconform * (1 - ELG_cen_ratio)
        #print(f"Cen-weighted conformity ELG sat HOD (should be the same as ELG sat HOD):\n{ELG_hod_sat_avg}")

        # derivation is complicated; average of X(X-1), where X is a linear combination of Poisson distributions
        # actually the below derivation is wrong I think; check with Max
        # ELG_hod_sat_ss1_squared = ELG_hod_sat_avg**2 - ELG_hod_sat_avg + ELG_hod_sat_conform * ELG_cen_ratio**2 + ELG_hod_sat_noconform * (1-ELG_cen_ratio)**2
        ELG_hod_sat_ss1_squared = ELG_hod_sat_conform**2 * ELG_cen_ratio + ELG_hod_sat_noconform**2 * (1 - ELG_cen_ratio)
        #print(f"ELG-ELG 1-halo satsat HOD (should be the same as ELG sat HOD):\n{np.sqrt(ELG_hod_sat_ss1_squared)}")
        return ELG_hod_sat_avg, np.sqrt(ELG_hod_sat_ss1_squared)
    
    # def update_hods(self, hod_params):
    #     for tracer in self.tracer_list:
    #         hod_cen, hod_sat = self.get_hods_given_tracer_and_params(self.M_h, hod_params, tracer, ELG_ELG=False)
    #         self.hod_dict[tracer]["cen"] = hod_cen
    #         self.hod_dict[tracer]["sat"] = hod_sat
    #     if "ELG" in self.tracer_list:
    #         _, ELG_hod_sat_conform = self.get_hods_given_tracer_and_params(self.M_h, hod_params, tracer="ELG", ELG_ELG=True)
    #         self.hod_dict["ELG_sat_EE"] = ELG_hod_sat_conform
    #         self.hod_dict["ELG_sat_avg"] = self.get_conformity_weighted_sat_hod(self, self.M_h, tracer="ELG")
    #         self.hod_dict["ELG_sat_ss1"] = TODO

    
    def print_hod_values(self, hod_params: np.ndarray):
        i = 0
        for tracer in self.tracer_list:
            print(f"{tracer} params:")
            for param in self.tracer_dict[tracer].values():
                value = hod_params[i]
                print(f"    {param.name}: {value}")
                i += 1