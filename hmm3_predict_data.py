import numpy as np
import pandas as pd
import os
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit 
from hmmlearn import hmm
# from hmmlearn.vhmm import VariationalGaussianHMM
from hmmlearn.hmm import GaussianHMM
# from hmmlearn.hmm import GMMHMM
from scipy.stats import norm

from get_data_functions import *
from assemble_data_functions import *
from HMM_functions_multistates import *
from plot_dist_functions import *
from stat_tests import *

def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2*np.pi) * sigma)

def gaussian_mixture(x, *params):
    """
    params = [w1, mu1, sigma1, w2, mu2, sigma2, ..., wK, muK, sigmaK]
    """
    K = len(params) // 3
    y = np.zeros_like(x)
    for k in range(K):
        w, mu, sigma = params[3*k:3*k+3]
        y += w * gaussian(x, mu, sigma)
    return y

def init_params_from_data(X, K):
    """
    Returns initial guess for curve_fit.
    """
    X = X.ravel()
    mus = np.linspace(X.min(), X.max(), K)
    sigmas = np.full(K, X.std() / K)
    weights = np.full(K, 1.0 / K)

    p0 = []
    for k in range(K):
        p0.extend([weights[k], mus[k], sigmas[k]])
    return p0

def fit_gmm_curvefit(X, K, bins=100):
    X = X.ravel()

    hist, edges = np.histogram(X, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p0 = init_params_from_data(X, K)

    bounds = (
        [0, X.min(), 1e-6] * K,          # lower
        [1, X.max(), X.std()*10] * K     # upper
    )

    popt, _ = curve_fit(
        gaussian_mixture,
        centers,
        hist,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    # unpack
    weights = np.array([popt[3*k] for k in range(K)])
    means   = np.array([popt[3*k+1] for k in range(K)])
    sigmas  = np.array([popt[3*k+2] for k in range(K)])

    # normalize weights (important)
    weights /= weights.sum()

    # sort by means
    order = np.argsort(means)

    weights = weights[order]
    means   = means[order]
    sigmas  = sigmas[order]

    return weights, means, sigmas

def sort_hmm_by_means(model):
    means = model.means_
    sort_key = means[:, 0]
    order = np.argsort(sort_key)

    model.startprob_ = model.startprob_[order]
    model.transmat_  = model.transmat_[order][:, order]
    model.means_     = model.means_[order]

    if hasattr(model, "covars_"):
        covars = model.covars_[order]

        if model.covariance_type == "diag":
            # handle weird (K, 1, 1) case
            if covars.ndim == 3:
                covars = np.squeeze(covars, axis=-1)

        model.covars_ = covars

    return model

#format data to input to HMM
def predict_states(all_cells, obs_param):
    data = []
    lengths = []
    for ind in range(len(all_cells)):
        avg_track_mag = all_cells[ind][obs_param]
        lengths.append(len(avg_track_mag))
        data.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1))) 

    Data = np.vstack(data)


    n_states = 3

    weights, means, sigmas = fit_gmm_curvefit(Data, n_states)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        init_params="t",
        params="stmc",
        n_iter=500,
        random_state=29
    )

    model.startprob_ = weights
    model.means_ = means.reshape(n_states, 1)
    model.covars_ = (sigmas**2).reshape(n_states, 1) + 1e-4

    model.fit(Data.reshape(-1, 1),lengths)

    model = sort_hmm_by_means(model)

    data = []
    lengths = []
    for ind in range(len(all_cells)):
        avg_track_mag = all_cells[ind][obs_param]
        lengths.append(len(avg_track_mag))
        data.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))


    #Predict WT states
    cell_states, state_probs = get_states_and_probs(lengths, model, np.vstack(data))

    #Get state segment metrics
    cells_states, pooled_state_df = calculate_segment_metrics(all_cells, cell_states, state_probs)


    return cells_states, pooled_state_df