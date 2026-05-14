import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit 
from hmmlearn import hmm
# from hmmlearn.vhmm import VariationalGaussianHMM
from hmmlearn.hmm import GaussianHMM
# from hmmlearn.hmm import GMMHMM
from scipy.stats import norm

from skimage.io import imread, imsave, imshow

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

from get_data_functions import *
from assemble_data_functions import *
from HMM_functions_multistates import *
from plot_dist_functions import *
from plot_state_functions import *

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

save_path = './HMM3StateMovies'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)


track_name = '20240918_WT_movie25_track6' #'20240918_WT_movie29_track2'
base_path = './data/tracks/20240918_WT_movie25_track6'
save_path = './HMM3StateMovies/HMM_3States_avgtracmag_{}'.format(track_name)
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

save_path_movie = save_path + '/cell_states'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path_movie):
  os.mkdir(save_path_movie)

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
all_cells = wt_cells

obs_param = 'avg_trac_mag'

data = []
lengths = []
for ind in range(len(all_cells)):
    avg_track_mag = all_cells[ind][obs_param]
    lengths.append(len(avg_track_mag))
    data.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1))) 

Data = np.vstack(data)

#Fit sum of 3 Gaussians functions to avg trac mag histogram
n_states = 3

weights, means, sigmas = fit_gmm_curvefit(Data, n_states)

model = GaussianHMM(
    n_components=n_states,
    covariance_type="diag",
    init_params="t",
    params="stmc",
    n_iter=500,
    random_state=42
)

model.startprob_ = weights
model.means_ = means.reshape(n_states, 1)
model.covars_ = (sigmas**2).reshape(n_states, 1) + 1e-4

model.fit(Data.reshape(-1, 1),lengths)

#get fitted model outputs
transmat = model.transmat_
dt = 15/60 #hours since dt is 15 min and I want to convert to rate per hour
transmat_rate = -(1/dt)*np.log(1-transmat)
stds = np.sqrt(model.covars_.flatten())
means = model.means_.flatten()
 # Empirical weights from predicted state sequence
X = Data
X = X.ravel()
state_seq = model.predict(X.reshape(-1, 1))
weights = np.bincount(state_seq, minlength=n_states) / len(X)

### Plot model
feature_name="{}".format(obs_param)
# Plot histogram of data
plt.figure(figsize=(10, 5))
plt.hist(X.flatten(), bins=50, density=True, alpha=0.3, color='gray', label="Observed data")
# Overlay Gaussian PDFs for each state
x_vals = np.linspace(X.min(), X.max(), 1000)
weighted_sum = np.zeros_like(x_vals)
for i, (mean, std) in enumerate(zip(means, stds)):
    y = norm.pdf(x_vals, loc=mean, scale=std)
    plt.plot(x_vals, weights[i] * y, label=f"State {i} (mean={means[i]:.2f})", linewidth=2)
    weighted_pdf = weights[i] * y
    weighted_sum += weighted_pdf
# Plot the mixture
plt.plot(x_vals, weighted_sum, 'k--', lw=2, label='Weighted mixture')
plt.title(f"HMM Emission Distributions for {feature_name}")
plt.xlabel(feature_name)
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig(save_path + '/HMM_Model.pdf', format='pdf')
plt.clf()

#Get transition rates
#get fitted model outputs
transmat = model.transmat_
dt = 15/60 #hours since dt is 15 min and I want to convert to rate per hour
transmat_rate = -(1/dt)*np.log(1-transmat)

title="HMM Transition Rates"
state_labels = [f"State {i}" for i in range(n_states)]
plt.figure(figsize=(6, 5))
sns.heatmap(transmat_rate, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=state_labels, yticklabels=state_labels,
            square=True, cbar_kws={'label': 'P(transition)'})
plt.xlabel("To State")
plt.ylabel("From State")
plt.title(title)
plt.tight_layout()
plt.savefig(save_path+'/transition_matrix_rate.pdf',format='pdf')
plt.clf()

#set up data into type to predict cell states
data_wt = []
lengths_wt = []
for ind in range(len(wt_cells)):
    avg_track_mag = wt_cells[ind][obs_param]
    lengths_wt.append(len(avg_track_mag))
    data_wt.append(np.reshape(avg_track_mag.values,(len(avg_track_mag.values),1)))


#Predict WT states
cell_states_wt, state_probs_wt = get_states_and_probs(lengths_wt, model, np.vstack(data_wt))

#Get state segment metrics
wt_cells_states, pooled_wt_state_df = calculate_segment_metrics(wt_cells, cell_states_wt, state_probs_wt)




for df in wt_cells_states:
    if track_name in df['name'].iloc[0]:
        avg_trac_mag = df['avg_trac_mag']
        states = df['state']
        probs0 = df['prob_state0']
        probs1 = df['prob_state1']
        probs2 = df['prob_state2']
        frames = df['frame'].to_numpy()
        framenum = frames-1
        # plot_1d_hmm_state_trace_and_probs(avg_trac_mag, states, probs, track_name, frames, save_path)
        # plot_1d_hmm_state_trace_and_probs_coloredby_avgtracmag(avg_trac_mag, states, probs, track_name, frames, save_path)
        plot_1d_hmm_state_trace_and_probs_coloredby_state(avg_trac_mag, states, probs0, probs1, probs2, track_name, frames, save_path)
        plot_state_track(base_path, states, framenum, save_path_movie)
