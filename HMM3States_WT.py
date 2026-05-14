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

save_path = './figures/HMM_{}_WT_3comp'.format(obs_param)
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


n_states = 3

weights, means, sigmas = fit_gmm_curvefit(Data, n_states)

logLs = []
for seed in range(30):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        init_params="t",
        params="stmc",
        n_iter=500,
        tol=1e-4,
        random_state=seed
    )
    model.startprob_ = weights
    model.means_ = means.reshape(n_states, 1)
    model.covars_ = (sigmas**2).reshape(n_states, 1) + 1e-4
    model.fit(Data.reshape(-1, 1),lengths)
    logLs.append(model.score(Data.reshape(-1, 1)))

np.sort(logLs)

plt.plot(logLs, 'o')
plt.ylabel("Final log-likelihood")
plt.xlabel("Random initialization")
plt.savefig(save_path + '/HMM_Model_Convergence_optima.pdf', format='pdf')
plt.clf()

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

plt.plot(model.monitor_.history)
plt.xlabel("EM iteration")
plt.ylabel("Log-likelihood")
plt.savefig(save_path + '/HMM_Model_Convergence.pdf', format='pdf')
plt.clf()

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

colors_for_state_plot = [(242/256,140/256,40/256), (11/256,218/256,81/256), (255/256,0,255/256)] 

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
    plt.plot(x_vals, weights[i] * y, label=f"State {i} (mean={means[i]:.2f})", linewidth=2, color=colors_for_state_plot[i])
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
transmat_rate = -1*(1/dt)*np.log(1-transmat)

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

#clears out sentinel file if it exists
open('{}/transition_wt_names.txt'.format(save_path),'w').close()
#create new sentinel file to write to
names_trans_wt = open('{}/transition_wt_names.txt'.format(save_path),'w')
file_lines_wt = []


low_to_med = []
med_to_low = []
med_to_high = []
high_to_med = []
low_to_high = []
high_to_low = []
no_transition = []
num_transitions = []
all_three_states = []
for df in wt_cells_states:
    states = df['state']
    states = states.mask(states==2,5)
    name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
    if len(np.unique(states)) > 1:
        n_transitions = np.sum(np.diff(states) != 0)
        num_transitions.append(n_transitions)
        if np.any(np.diff(states) == 1):
            low_to_med.append(name + ', ')
        if np.any(np.diff(states) == -1):
            med_to_low.append(name + ', ')
        if np.any(np.diff(states) == 4):
            med_to_high.append(name + ', ')
        if np.any(np.diff(states) == -4):
            high_to_med.append(name + ', ')
        if np.any(np.diff(states) == -5):
            high_to_low.append(name + ', ')
        if np.any(np.diff(states) == 5):
            low_to_high.append(name + ', ')
    if len(np.unique(states)) == 3:
        all_three_states.append(name + ', ')

    else:
       no_transition.append(name + ', ')


transition_cells = len(num_transitions)

file_lines_wt.append('low to med \n {} \n'.format(low_to_med) + '\n med to low \n {}'.format(low_to_med) + '\n med to high \n {}'.format(med_to_high)+'\n high to med \n {}'.format(high_to_med) +'\n low to high \n {}'.format(low_to_high)+'\n high to low \n {}'.format(high_to_low) + '\n all three states \n {}'.format(all_three_states))

#write lines to text file 
names_trans_wt.writelines(file_lines_wt)
names_trans_wt.close() 

median_segment_df_cells = []
for df in wt_cells_states:
  # define segment IDs whenever state changes
  df['segment_id'] = (df['state'].diff().fillna(1) != 0).cumsum()
  # now get one row per segment (e.g. first row of each)
  segments = df.groupby('segment_id', as_index=False).first()
  segment_df = df.groupby('segment_id').median()
  segment_df['track_name'] = [df['track_name'].iloc[0]] * len(segment_df)
  median_segment_df_cells.append(segment_df)

median_segment_df_all = pd.concat(median_segment_df_cells, ignore_index=True)

####Median Plots###
save_path_fig = '{}/median_segment_stripplots'.format(save_path)
numeric_segment_df_all = median_segment_df_all.select_dtypes(include=np.number)

if not os.path.exists(save_path_fig):
  os.mkdir(save_path_fig)

colors_for_state_plot = [(242/256,140/256,40/256), (11/256,218/256,81/256), (255/256,0,255/256)] 

params = ['segment_speed','area','absskew','avg_trac_mag','segment_DT','segment_duration','eccentricity','solidity','dip_ratio','turning_angle','step_length',"length","width",'protrusion_id','perim_ratio','protrusion_act','retraction_act','all_act']
for column in params: #numeric_segment_df_all.columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=median_segment_df_all, x='state', y=column, hue="state", palette=colors_for_state_plot, alpha=0.8, inner='quart', legend=False)
    sns.swarmplot(data=median_segment_df_all, x='state', y=column, hue='track_name', palette='deep',legend=False) 
    state0 = median_segment_df_all.loc[median_segment_df_all["state"] == 0, column].to_numpy()
    state1 = median_segment_df_all.loc[median_segment_df_all["state"] == 1, column].to_numpy()
    state2 = median_segment_df_all.loc[median_segment_df_all["state"] == 2, column].to_numpy()
    p_value_01 = permutation_test(state0, state1)
    p_value_02 = permutation_test(state0, state2)
    p_value_12 = permutation_test(state1, state2)
    plt.plot([],[], ' ', label ="p val 0 1: {}".format(np.round(p_value_01,6)))
    plt.plot([],[], ' ', label ="p val 0 2: {}".format(np.round(p_value_02,6)))
    plt.plot([],[], ' ', label ="p val 1 2: {}".format(np.round(p_value_12,6)))
    plt.xlabel("Type")
    plt.ylabel("{}".format(column))
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig('{}/{}_stripplot.pdf'.format(save_path_fig,column),bbox_inches='tight',format='pdf')
    plt.clf()