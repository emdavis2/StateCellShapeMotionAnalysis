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

from get_data_functions import *
from assemble_data_functions import *
from HMM_functions_multistates import *
from plot_dist_functions import *
from stat_tests import *
from hmm3_predict_data import *

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

#import data 
data_dir = './data'
dipole_dict = np.load(data_dir + '/dipole_dict.npy', allow_pickle=True).item()
ellipse_dict = np.load(data_dir + '/ellipse_dict.npy', allow_pickle=True).item()
quad_dict = np.load(data_dir + '/quad_dict.npy', allow_pickle=True).item()

tracksgeo_path = data_dir + '/tracks_geo_region.pkl'
tracksgeo_dict = load_tracksgeo(tracksgeo_path, dipole_dict)


protrusion_df = pd.read_pickle(data_dir + "/protrusion.pkl")

save_path = './figures/protrusions_3hmmstates'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)

pixel_size = 0.645 #microns


wt_cells_states, wt_pooled_state_df= predict_states(wt_cells, 'avg_trac_mag')


arpc2ko_cells_states, arpc2ko_pooled_state_df = predict_states(arpc2ko_cells, 'avg_trac_mag')



save_path = './figures/protrusions_3hmmstates/median'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

median_wt_segment_df_cells = []
for df in wt_cells_states:
  # define segment IDs whenever state changes
  df['segment_id'] = (df['state'].diff().fillna(1) != 0).cumsum()
  # now get one row per segment (e.g. first row of each)
  segments = df.groupby('segment_id', as_index=False).first()
  segment_df = df.groupby('segment_id').median()
  segment_df['track_name'] = [df['track_name'].iloc[0]] * len(segment_df)
  median_wt_segment_df_cells.append(segment_df)

median_wt_segment_df_all = pd.concat(median_wt_segment_df_cells, ignore_index=True)
wt_df = median_wt_segment_df_all

median_arpc2ko_segment_df_cells = []
for df in arpc2ko_cells_states:
  # define segment IDs whenever state changes
  df['segment_id'] = (df['state'].diff().fillna(1) != 0).cumsum()
  # now get one row per segment (e.g. first row of each)
  segments = df.groupby('segment_id', as_index=False).first()
  segment_df = df.groupby('segment_id').median()
  segment_df['track_name'] = [df['track_name'].iloc[0]] * len(segment_df)
  median_arpc2ko_segment_df_cells.append(segment_df)

median_arpc2ko_segment_df_all = pd.concat(median_arpc2ko_segment_df_cells, ignore_index=True)
arpc2ko_df = median_arpc2ko_segment_df_all

# params = ['filtered_protrusion_lengths', 'filtered_protrusion_widths', 'filtered_num_protrusions']
params = ['width','length','protrusion_id','protrusion_act','retraction_act','all_act','mean_wid','mean_len','mean_num','max_len','mean_protr_lt']

colors_for_state_plot = [(242/256,140/256,40/256), (11/256,218/256,81/256), (255/256,0,255/256)] 

for column in params:
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(10, 4))
    sns.violinplot(data=wt_df, x='state', y=column, hue="state",  palette=colors_for_state_plot, alpha=0.8, inner='quart', legend=False, ax=axs[0])
    sns.swarmplot(data=wt_df, x='state', y=column, hue='track_name', palette='deep',legend=False,ax=axs[0],size=3)
    state0 = wt_df.loc[wt_df["state"] == 0, column].to_numpy()
    state1 = wt_df.loc[wt_df["state"] == 1, column].to_numpy()
    state2 = wt_df.loc[wt_df["state"] == 2, column].to_numpy()
    p_value_01 = permutation_test(state0, state1)
    p_value_02 = permutation_test(state0, state2)
    p_value_12 = permutation_test(state1, state2)
    axs[0].plot([],[], ' ', label ="p val 0 1: {}".format(np.round(p_value_01,6)))
    axs[0].plot([],[], ' ', label ="p val 0 2: {}".format(np.round(p_value_02,6)))
    axs[0].plot([],[], ' ', label ="p val 1 2: {}".format(np.round(p_value_12,6)))
    axs[0].legend()
    sns.violinplot(data=arpc2ko_df, x='state', y=column, hue="state", palette=colors_for_state_plot, alpha=0.8, inner='quart', legend=False, ax=axs[1])
    sns.swarmplot(data=arpc2ko_df, x='state', y=column, hue='track_name', palette='deep',legend=False, ax=axs[1],size=3) 
    state0 = arpc2ko_df.loc[arpc2ko_df["state"] == 0, column].to_numpy()
    state1 = arpc2ko_df.loc[arpc2ko_df["state"] == 1, column].to_numpy()
    state2 = arpc2ko_df.loc[arpc2ko_df["state"] == 2, column].to_numpy()
    p_value_01 = permutation_test(state0, state1)
    p_value_02 = permutation_test(state0, state2)
    p_value_12 = permutation_test(state1, state2)
    axs[1].plot([],[], ' ', label ="p val 0 1: {}".format(np.round(p_value_01,6)))
    axs[1].plot([],[], ' ', label ="p val 0 2: {}".format(np.round(p_value_02,6)))
    axs[1].plot([],[], ' ', label ="p val 1 2: {}".format(np.round(p_value_12,6)))
    axs[1].legend()
    axs[0].set_title('WT')
    axs[1].set_title('ARPC2KO')
    plt.xticks(rotation=90)

    plot_name = '{}_stripplot'.format(column)
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

    #######################################################################################

    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(10, 4))
    sns.boxplot(data=wt_df, x='state', y=column, hue="state",  palette=colors_for_state_plot, legend=False, ax=axs[0])
    sns.swarmplot(data=wt_df, x='state', y=column, hue='track_name', palette='deep',legend=False,ax=axs[0],size=3)
    state0 = wt_df.loc[wt_df["state"] == 0, column].to_numpy()
    state1 = wt_df.loc[wt_df["state"] == 1, column].to_numpy()
    state2 = wt_df.loc[wt_df["state"] == 2, column].to_numpy()
    p_value_01 = permutation_test(state0, state1)
    p_value_02 = permutation_test(state0, state2)
    p_value_12 = permutation_test(state1, state2)
    axs[0].plot([],[], ' ', label ="p val 0 1: {}".format(np.round(p_value_01,6)))
    axs[0].plot([],[], ' ', label ="p val 0 2: {}".format(np.round(p_value_02,6)))
    axs[0].plot([],[], ' ', label ="p val 1 2: {}".format(np.round(p_value_12,6)))
    axs[0].legend()
    sns.boxplot(data=arpc2ko_df, x='state', y=column, hue="state", palette=colors_for_state_plot, legend=False, ax=axs[1])
    sns.swarmplot(data=arpc2ko_df, x='state', y=column, hue='track_name', palette='deep',legend=False, ax=axs[1],size=3) 
    state0 = arpc2ko_df.loc[arpc2ko_df["state"] == 0, column].to_numpy()
    state1 = arpc2ko_df.loc[arpc2ko_df["state"] == 1, column].to_numpy()
    state2 = arpc2ko_df.loc[arpc2ko_df["state"] == 2, column].to_numpy()
    p_value_01 = permutation_test(state0, state1)
    p_value_02 = permutation_test(state0, state2)
    p_value_12 = permutation_test(state1, state2)
    axs[1].plot([],[], ' ', label ="p val 0 1: {}".format(np.round(p_value_01,6)))
    axs[1].plot([],[], ' ', label ="p val 0 2: {}".format(np.round(p_value_02,6)))
    axs[1].plot([],[], ' ', label ="p val 1 2: {}".format(np.round(p_value_12,6)))
    axs[1].legend()
    axs[0].set_title('WT')
    axs[1].set_title('ARPC2KO')
    plt.xticks(rotation=90)

    plot_name = '{}_boxplot'.format(column)
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

    ######################################################################################

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4))
    if column == 'filtered_num_protrusions':
      bool_val = 'True'
    else:
      bool_val = 'False' 

    plot_name = '{}_hist'.format(column)
    sns.histplot(data=wt_df, x=column, hue="state", stat="density", common_norm=False, discrete=bool_val, palette=colors_for_state_plot, ax=axs[0],element='step')
    sns.histplot(data=arpc2ko_df, x=column, hue="state", stat="density", common_norm=False, discrete=bool_val, palette=colors_for_state_plot, ax=axs[1],element='step')
    axs[0].set_title('WT')
    axs[1].set_title('ARPC2KO')
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()
