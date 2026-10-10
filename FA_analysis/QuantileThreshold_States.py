import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
import pickle
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from stat_tests import *

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

traction_df_path = './FA_analysis/data/traction_df.pkl'
traction_df = pd.read_pickle(traction_df_path)

fa_df_path = './FA_analysis/data/fa_df.pkl'
fa_df = pd.read_pickle(fa_df_path)

shape_df_path = './FA_analysis/data/shape_df.pkl'
shape_df = pd.read_pickle(shape_df_path)

protrusion_df_path = './FA_analysis/data/protrusion_df.pkl'
protrusion_df = pd.read_pickle(protrusion_df_path)

body_df_path = './FA_analysis/data/body_df.pkl'
body_df = pd.read_pickle(body_df_path)
# body_df = pd.DataFrame([body_df])

spatial_fa_df_path = './FA_analysis/data/spatial_fa_df.pkl'
spatial_fa_df = pd.read_pickle(spatial_fa_df_path)

save_path = './figures/QuantThresh_Pax'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

combined_rows = []

all_keys = set(traction_df.keys()) | set(fa_df.keys()) | set(shape_df.keys()) | set(spatial_fa_df.keys()) | set(body_df.keys())

for key in all_keys:
    df1 = traction_df.get(key, pd.DataFrame([{}]))
    df2 = fa_df.get(key, pd.DataFrame([{}]))
    df3 = shape_df.get(key, pd.DataFrame([{}]))
    df4 = spatial_fa_df.get(key, pd.DataFrame([{}]))
    df5 = body_df.get(key, pd.DataFrame([{}]))

    combined_df = pd.concat([df1, df2, df3, df4, df5], axis=1)
    combined_df['name'] = key

    combined_rows.append(combined_df)

final_df = pd.concat(combined_rows, ignore_index=True)

# Get avg features per cell for protrusion data and add to final df
all_protrusions = []

for cell, protrusions in protrusion_df.items():

    for p in protrusions:
        p["name"] = cell
        all_protrusions.append(p)

protrusion_data = pd.DataFrame(all_protrusions)

# protrusion_avg_df = (
#     protrusion_data
#     .groupby("name")
#     .mean(numeric_only=True)
#     .reset_index()
# )

protrusion_avg_df = (
    protrusion_data
    .copy()
)

numeric_cols = protrusion_data.select_dtypes(include=[np.number]).columns

protrusion_avg_df[numeric_cols] = protrusion_avg_df[numeric_cols].replace(0, np.nan)

protrusion_avg_df = (
    protrusion_avg_df
    .groupby("name")
    .mean(numeric_only=True)
    .reset_index()
)

cell_num_protrusions = protrusion_data.groupby("name").size()

protrusion_avg_df['num_protr'] = cell_num_protrusions

final_df = pd.merge(final_df, protrusion_avg_df, on="name")

final_df['fa_density'] = final_df['num_fa']/final_df['area']
final_df['protr_fa_density'] = final_df['fa_count_total']/final_df['protr_area']
final_df['body_fa_density'] = final_df["body_fa_count"]/final_df["body_area"]

#######################################################

obs_param = 'avg_trac_mag'

lower_thresh = final_df[obs_param].quantile(0.2)
upper_thresh = final_df[obs_param].quantile(0.8)

final_df['state'] = np.select(
    [
        final_df[obs_param] < lower_thresh,
        final_df[obs_param] > upper_thresh
    ],
    [0, 2],
    default=1
)

final_df = final_df[final_df['state'] != 1]

colors_for_state_plot = [(242/256,140/256,40/256), (11/256,218/256,81/256)] 

# 'fa_area_avg','fa_count_total' refer to FAs only in protrusion region
params = ['fa_avg_area','fa_med_area','num_fa', 'faai', 'avg_trac_mag','traction_mag', 'dipole_ratio', 'base_fa_count', 'base_fa_area', 'base_fa_avg', 'mid_fa_count', 'mid_fa_area', 'mid_fa_avg', 'tip_fa_count', 'tip_fa_area', 'tip_fa_avg', 'fa_area_avg','fa_count_total', 'protr_area', 'area', 'solidity', 'eccentricity', 'fa_density', 'protr_fa_density','total_force', 'total_force_avg', 'fa_mean_dist', 'fa_med_dist', "body_force_avg", "body_fa_avg", "body_fa_count", 'body_fa_density']

zero_invalid_cols = ['num_fa', 'base_fa_count', 'mid_fa_count', 'tip_fa_count', 'fa_count_total']


for column in params: #numeric_segment_df_all.columns:
    df_plot = final_df.copy()
        
    if column in zero_invalid_cols:
        df_plot = df_plot[df_plot[column] != 0]
    
    df_plot = df_plot.dropna(subset=[column])

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_plot, x='state', y=column, hue="state", palette=colors_for_state_plot, alpha=0.8, inner='quart', legend=False)
    sns.swarmplot(data=df_plot, x='state', y=column, color=(243/256,243/256,243/256), legend=False) 
    state0 = df_plot.loc[df_plot["state"] == 0, column].dropna()
    state2 = df_plot.loc[df_plot["state"] == 2, column].dropna()
    U1, p_value_01 = mannwhitneyu(state0, state2)
    plt.plot([],[], ' ', label ="p val 0 2: {}".format(np.round(p_value_01,6)))
    plt.xlabel("Type")
    plt.ylabel("{}".format(column))
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig('{}/{}_stripplot.pdf'.format(save_path,column),bbox_inches='tight',format='pdf')
    plt.clf()
    plt.close()