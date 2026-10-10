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


protrusion_df = pd.read_pickle(data_dir + "/protrusion_force.pkl")


arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)

pixel_size = 0.645 #microns


wt_cells_states, wt_pooled_state_df= predict_states(wt_cells, 'avg_trac_mag')


arpc2ko_cells_states, arpc2ko_pooled_state_df = predict_states(arpc2ko_cells, 'avg_trac_mag')



save_path = './figures/wt_arpc2ko_3hmmstates'

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

###################Plot differences in force states between WT and Arpc2KO##################

params = ['segment_speed','area','absskew','avg_trac_mag','segment_DT','segment_duration','eccentricity','solidity','dip_ratio','turning_angle','step_length',"length","width",'protrusion_id','perim_ratio','protrusion_act','retraction_act','all_act']

wt_df['type'] = ['WT'] * len(wt_df)
arpc2ko_df['type'] = ['ARPC2KO'] * len(arpc2ko_df)

data_bp = pd.concat([wt_df, arpc2ko_df]).reset_index()

for column in params: #numeric_segment_df_all.columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data_bp, x='state', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], dodge=True, alpha=0.8, inner='quart', legend=False)
    sns.swarmplot(data=data_bp, x='state', y=column, hue='type', palette=[(243/256,243/256,243/256),(243/256,243/256,243/256)],dodge=True,legend=False,linewidth=.5,edgecolor="black",size=3) 
    state0_wt = wt_df.loc[wt_df["state"] == 0, column].to_numpy()
    state0_arpc2ko = arpc2ko_df.loc[arpc2ko_df["state"] == 0, column].to_numpy()
    state1_wt = wt_df.loc[wt_df["state"] == 1, column].to_numpy()
    state1_arpc2ko = arpc2ko_df.loc[arpc2ko_df["state"] == 1, column].to_numpy()
    state2_wt = wt_df.loc[wt_df["state"] == 2, column].to_numpy()
    state2_arpc2ko = arpc2ko_df.loc[arpc2ko_df["state"] == 2, column].to_numpy()
    p_value_0 = permutation_test(state0_wt, state0_arpc2ko)
    p_value_1 = permutation_test(state1_wt, state1_arpc2ko)
    p_value_2 = permutation_test(state2_wt, state2_arpc2ko)
    plt.plot([],[], ' ', label ="p val 0: {}".format(np.round(p_value_0,6)))
    plt.plot([],[], ' ', label ="p val 1: {}".format(np.round(p_value_1,6)))
    plt.plot([],[], ' ', label ="p val 2: {}".format(np.round(p_value_2,6)))
    plt.xlabel("state")
    plt.ylabel("{}".format(column))
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig('{}/{}_stripplot.pdf'.format(save_path,column),bbox_inches='tight',format='pdf')
    plt.clf()

