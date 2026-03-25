import numpy as np
from scipy.optimize import minimize as minimise

def nfw_radial_pdf(x):
    """
    P(x) = x**2/(x*(1+x)**2)
    """
    return x**2 / (x*(1+x)**2)

def nfw_radial_cdf(R):
    """
    Integrate x**2/(x*(1+x)**2) from 0 to R
    Calculated using wolfram alpha
    """
    return 1/(R+1) + np.log(R + 1) - 1

def normalised_nfw_cdf(x: float, R: float):
    """
    Normalised cdf of P(x) = x**2/(x*(1+x)**2), up to some cutoff R
    Clipped to be minimum 0 and maximum 1, to make it "bijective"
    """
    if x < 0.0:
        return 0.0
    normalised_cdf = nfw_radial_cdf(x) / nfw_radial_cdf(R)
    if normalised_cdf > 1.0:
        return 1.0
    elif normalised_cdf < 0.0:
        return 0.0
    else:
        return normalised_cdf

# def normalised_nfw_cdf_bijective(x: np.ndarray, R: float):
#     """
#     A version of normalised_nfw_cdf that's bijective to avoid minimisation difficulties
#     """
#     if x >= 0:
#         return normalised_nfw_cdf(x, R)
#     else:
#         return x**3

def squared_error(x: float, y_0: float, R: float):
    """
    Calculates the squared error in a given value a from being equal to C(x, R) = a, where C(x, R) is the CDF of the NFW with cutoff R
    """
    y = normalised_nfw_cdf(x, R)
    return (y - y_0)**2

def invert_nfw_cdf(y: np.ndarray, R: float):
    """
    Given an array of values y, finds x such that P(X <= x) = y, where P(x) is the radial nfw profile cut off at R
    Takes in y and R; returns array of values x
    Doesn't work if R <= 0.5
    """
    assert R > 0.5
    initial_guess = 0.5
    result = np.empty(len(y))
    bnds = [(0, R)]
    for idx, y_value in enumerate(y):
        optimisation_output = minimise(squared_error, initial_guess, args=(y_value, R), bounds=bnds, tol=1e-6)
        assert optimisation_output.success, optimisation_output.message
        minimum = optimisation_output.x[0]
        if minimum > R or minimum < 0:
            raise Exception("Minimisation screwed up: gave value of "+str(minimum)+" for input "+str(y_value)+", when maximum should've been "+str(R))
        result[idx] = minimum
    return result

def nfw_draw(N: int, max: float, seed=None):
    """
    Returns a numpy array of random values drawn from an NFW profile:
    P(x) = x**2/(x*(1+x)**2)
    
    Takes about 12 seconds to run for N=10000; should be linear in N

    Arguments:
        N (integer): Number of random values to return

        max (float): Cutoff of the profile (otherwise it diverges); must be greater than 0.5 for technical reasons

        seed: integer, defaults to None
    """
    # Implemented by taking a random sample uniform in [0, 1), and then inverting the NFW cumulative distribution function
    rng = np.random.default_rng(seed=seed)
    randuniform = rng.random(N)
    return invert_nfw_cdf(randuniform, max)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    print("Testing NFW draw")
    max = 40 #5.5 # about when the CDF = 1, so the two are normalised about the same
    N = 10000
    sample = nfw_draw(N, max)
    print(np.count_nonzero(sample == 0.0))

    # counts, bins = np.histogram(sample, bins=30, density=True)

    # print(counts)
    
    # true_xvals = np.linspace(0.001, max, num=50)
    # true_yvals = nfw_radial_pdf(true_xvals)

    # plt.stairs(counts, bins)
    # plt.plot(true_xvals, true_yvals)
    # plt.show()

    # xvals = np.linspace(-1, 5, 90)
    # yvals = [normalised_nfw_cdf(xval, 40) for xval in xvals]
    # plt.plot(xvals, yvals)
    # plt.show()
