import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit 
from hmmlearn import hmm
# from hmmlearn.vhmm import VariationalGaussianHMM
from hmmlearn.hmm import GaussianHMM
# from hmmlearn.hmm import GMMHMM
from scipy.stats import norm
from scipy.stats import pearsonr

from get_data_functions import *
from assemble_data_functions import *
from HMM_functions_multistates import *
from plot_dist_functions import *
from stat_tests import *

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

save_path = './figures'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

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

    return weights, means, sigmas

#import data 
data_dir = './data'
dipole_dict = np.load(data_dir + '/dipole_dict.npy', allow_pickle=True).item()
ellipse_dict = np.load(data_dir + '/ellipse_dict.npy', allow_pickle=True).item()
quad_dict = np.load(data_dir + '/quad_dict.npy', allow_pickle=True).item()

tracksgeo_path = data_dir + '/tracks_geo_region.pkl'
tracksgeo_dict = load_tracksgeo(tracksgeo_path, dipole_dict)

protrusion_df = pd.read_pickle(data_dir + "/protrusion.pkl")

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)

#format data to input to HMM
all_cells = arpc2ko_cells

obs_param = 'avg_trac_mag'

save_path = './figures/HMM_{}_ARPC2KO_AIC'.format(obs_param)
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

data = []
lengths = []
for ind in range(len(all_cells)):
    avg_track_mag = all_cells[ind][obs_param]
    lengths.append(len(avg_track_mag))
    data.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1))) 

Data = np.vstack(data)

aic = []
bic = []
lls = []
ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for n in ns:
  best_ll = None
  best_model = None
  for i in range(5):
    weights, means, sigmas = fit_gmm_curvefit(Data, n)

    h = GaussianHMM(
        n_components=n,
        covariance_type="diag",
        init_params="t",
        params="stmc",
        n_iter=500,
    )

    h.startprob_ = weights
    h.means_ = means.reshape(n, 1)
    h.covars_ = (sigmas**2).reshape(n, 1) + 1e-4

    h.fit(Data.reshape(-1, 1))
    score = h.score(Data)
    if not best_ll or best_ll < score:
            best_ll = score
            best_model = h

  aic.append(best_model.aic(Data))
  bic.append(best_model.bic(Data))
  lls.append(best_model.score(Data))

fig, ax = plt.subplots()
ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
ax2 = ax.twinx()
ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")

ax.legend(handles=ax.lines + ax2.lines)
ax.set_title("Using AIC/BIC for Model Selection")
ax.set_ylabel("Criterion Value (lower is better)")
ax2.set_ylabel("LL (higher is better)")
ax.set_xlabel("Number of HMM Components")
fig.tight_layout()

plt.savefig(save_path+'/aic_bic_{}.pdf'.format(obs_param), format='pdf')
plt.clf()
