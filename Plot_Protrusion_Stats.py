import numpy as np
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu

from get_data_functions import *
from assemble_data_functions import *
from plot_dist_functions import *
from stat_tests import *

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

# skel_csv_path = data_dir + '/skeleton.csv'
# skeleton_df = load_skeletondf(skel_csv_path)

# protr_csv_path = data_dir + '/protrusion.csv'
# protrusion_df = pd.read_csv(protr_csv_path, converters={
#    "protrusion_lengths": parse_list_comma,
#     "median_protrusion_widths": parse_list_space,
#     "mean_protrusion_widths": parse_list_space
# })

protrusion_df = pd.read_pickle(data_dir + "/protrusion.pkl")

### want 25 pixels as my cut off length for definition of protrusion
threshold = 30

pixel_size = 0.645 #microns

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)

# filtered_arpc2ko_cells = []
# filtered_median_arpc2ko_df = []
# for df in arpc2ko_cells:
#   df['filtered_num_protrusions'] = df['protrusion_lengths'].apply(lambda lst: sum(x > threshold for x in lst))
#   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)
#   filtered_df = df[['protrusion_lengths', 'median_protrusion_widths', 'filtered_num_protrusions', 'type', 'track_name']].explode(['protrusion_lengths', 'median_protrusion_widths']).reset_index(drop=True)
#   #filter out small branches
#   filtered_df = filtered_df[filtered_df['protrusion_lengths'] > threshold].reset_index()
#   #convert to microns
#   filtered_df['protrusion_lengths'] = filtered_df['protrusion_lengths'].astype(float) * pixel_size
#   filtered_df['median_protrusion_widths'] = filtered_df['median_protrusion_widths'].astype(float) * pixel_size
#   filtered_arpc2ko_cells.append(filtered_df)

#   medians = filtered_df.median(numeric_only = True)        
#   medians.loc['track_name']=filtered_df['track_name'].iloc[0]
#   medians.loc['type']=filtered_df['type'].iloc[0]
#   filtered_median_arpc2ko_df.append(medians)

# pooled_arpc2ko_df = pd.concat(filtered_arpc2ko_cells, ignore_index=True)
# filtered_median_arpc2ko_df = pd.concat(filtered_median_arpc2ko_df,axis=1).T

# filtered_wt_cells = []
# filtered_median_wt_df = []
# for df in wt_cells:
#   df['filtered_num_protrusions'] = df['protrusion_lengths'].apply(lambda lst: sum(x > threshold for x in lst))
#   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)
#   filtered_df = df[['protrusion_lengths', 'median_protrusion_widths', 'filtered_num_protrusions', 'type', 'track_name']].explode(['protrusion_lengths', 'median_protrusion_widths']).reset_index(drop=True)
#   #filter out small branches
#   filtered_df = filtered_df[filtered_df['protrusion_lengths'] > threshold].reset_index()
#   #convert to microns
#   filtered_df['protrusion_lengths'] = filtered_df['protrusion_lengths'].astype(float) * pixel_size
#   filtered_df['median_protrusion_widths'] = filtered_df['median_protrusion_widths'].astype(float) * pixel_size
#   filtered_wt_cells.append(filtered_df)

#   medians = filtered_df.median(numeric_only = True)        
#   medians.loc['track_name']=filtered_df['track_name'].iloc[0]
#   medians.loc['type']=filtered_df['type'].iloc[0]
#   filtered_median_wt_df.append(medians)


# pooled_wt_df = pd.concat(filtered_wt_cells, ignore_index=True)
# filtered_median_wt_df = pd.concat(filtered_median_wt_df,axis=1).T

median_wt_df = []
for df in wt_cells:
  df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)
  medians = df.median(numeric_only = True)        
  medians.loc['track_name']=df['track_name'].iloc[0]
  medians.loc['type']=df['type'].iloc[0]
  median_wt_df.append(medians)

filtered_median_wt_df = pd.concat(median_wt_df,axis=1).T

median_arpc2ko_df = []
for df in arpc2ko_cells:
  df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)
  medians = df.median(numeric_only = True)        
  medians.loc['track_name']=df['track_name'].iloc[0]
  medians.loc['type']=df['type'].iloc[0]
  median_arpc2ko_df.append(medians)

filtered_median_arpc2ko_df = pd.concat(median_arpc2ko_df,axis=1).T

save_path = './figures/protrusions_newcode'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

# params = ['protrusion_lengths', 'median_protrusion_widths', 'filtered_num_protrusions'] #, 'mean_protrusion_widths', 'number_protrusions']
params = ['width','length','protrusion_id']


# pooled_wt_df["filtered_num_protrusions"] = pooled_wt_df["protrusion_lengths"].apply(
#     lambda lst: sum(x > threshold for x in lst)
# )

# pooled_arpc2ko_df["filtered_num_protrusions"] = pooled_arpc2ko_df["protrusion_lengths"].apply(
#     lambda lst: sum(x > threshold for x in lst)
# )

# wt_df = pooled_wt_df[['protrusion_lengths', 'median_protrusion_widths', 'filtered_num_protrusions', 'type', 'track_name']].explode(['protrusion_lengths', 'median_protrusion_widths']).reset_index(drop=True)
# arpc2ko_df = pooled_arpc2ko_df[['protrusion_lengths', 'median_protrusion_widths', 'filtered_num_protrusions', 'type', 'track_name']].explode(['protrusion_lengths', 'median_protrusion_widths']).reset_index(drop=True)
# #filter out small branches
# wt_df = wt_df[wt_df['protrusion_lengths'] > threshold].reset_index()
# arpc2ko_df = arpc2ko_df[arpc2ko_df['protrusion_lengths'] > threshold].reset_index()
# #convert to microns
# wt_df['protrusion_lengths'] = wt_df['protrusion_lengths'].astype(float) * pixel_size
# wt_df['median_protrusion_widths'] = wt_df['median_protrusion_widths'].astype(float) * pixel_size
# arpc2ko_df['protrusion_lengths'] = arpc2ko_df['protrusion_lengths'].astype(float) * pixel_size
# arpc2ko_df['median_protrusion_widths'] = arpc2ko_df['median_protrusion_widths'].astype(float) * pixel_size

wt_df = pooled_wt_df
arpc2ko_df = pooled_arpc2ko_df

data_bp = pd.concat([wt_df, arpc2ko_df]).reset_index(drop=True)

for column in params:
    sns.violinplot(data=data_bp, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner=None, legend=False)
    # sns.swarmplot(data=data_bp, x='type', y=column, hue='track_name', palette='deep',legend=False)
    # data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
    # data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
    # U1, p = mannwhitneyu(data1, data2)
    # plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
    # plt.legend()
    plt.xlabel("Type")
    plt.ylabel("{}".format(column))
    plt.xticks(rotation=90)

    plot_name = '{}_stripplot'.format(column)
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

    if column == 'filtered_num_protrusions':
      bool_val = 'True'
    else:
      bool_val = 'False' 

    plot_name = '{}_hist'.format(column)
    sns.histplot(data=data_bp, x=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], stat="density", common_norm=False, discrete=bool_val)
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

   
save_path = save_path + '/median'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

data_bp = pd.concat([filtered_median_wt_df, filtered_median_arpc2ko_df]).reset_index(drop=True)

for column in params:
    sns.violinplot(data=data_bp, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner='quart', legend=False)
    sns.swarmplot(data=data_bp, x='type', y=column, color=(243/256,243/256,243/256),legend=False)
    data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
    data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
    U1, p = mannwhitneyu(data1, data2)
    plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
    plt.legend()
    plt.xlabel("Type")
    plt.ylabel("{}".format(column))
    plt.xticks(rotation=90)

    plot_name = '{}_stripplot'.format(column)
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

    if column == 'filtered_num_protrusions':
      bool_val = 'True'
    else:
      bool_val = 'False' 

    plot_name = '{}_hist'.format(column)
    sns.histplot(data=data_bp, x=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], stat="density", common_norm=False, discrete=bool_val)
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()


# save_path = './figures/stripplots_protrusions'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)


# for column in params:
#     if column == 'number_protrusions':
#        pixel_size = 1
#     else:
#        pixel_size = 0.645 #microns
#     wt_df = pooled_wt_df[[column, 'type', 'track_name']].explode(column)
#     wt_df[column] = wt_df[column].astype(float) * pixel_size
#     wt_df = wt_df.reset_index(drop=True)
#     arpc2ko_df = pooled_arpc2ko_df[[column, 'type', 'track_name']].explode(column)
#     arpc2ko_df[column] = arpc2ko_df[column].astype(float) * pixel_size
#     arpc2ko_df = arpc2ko_df.reset_index(drop=True)
#     data_bp = pd.concat([wt_df, arpc2ko_df]).reset_index(drop=True)
#     sns.violinplot(data=data_bp, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner=None, legend=False)
#     # sns.swarmplot(data=data_bp, x='type', y=column, hue='track_name', palette='deep',legend=False)
#     # data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
#     # data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
#     # U1, p = mannwhitneyu(data1, data2)
#     # plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
#     # plt.legend()
#     plt.xlabel("Type")
#     plt.ylabel("{}".format(column))
#     plt.xticks(rotation=90)

#     plot_name = '{}_stripplot'.format(column)

#     plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
#     plt.clf()


# save_path = './figures/stripplots_protrusions_medianvals_percellprotrusions'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)


# for column in params:
#     if column == 'number_protrusions':
#        pixel_size = 1
#     else:
#        pixel_size = 0.645 #microns
#     wt_df = pooled_wt_df[[column, 'type', 'track_name']].copy()
#     wt_df.loc[wt_df[column].apply(lambda x: isinstance(x, list) and len(x) == 0), column] = np.nan
#     wt_df[column] = wt_df[column].apply(np.median) * pixel_size
#     wt_df = wt_df.reset_index(drop=True)
#     arpc2ko_df = pooled_arpc2ko_df[[column, 'type', 'track_name']].copy()
#     arpc2ko_df.loc[arpc2ko_df[column].apply(lambda x: isinstance(x, list) and len(x) == 0), column] = np.nan
#     arpc2ko_df[column] = arpc2ko_df[column].apply(np.median) * pixel_size
#     arpc2ko_df = arpc2ko_df.reset_index(drop=True)
#     data_bp = pd.concat([wt_df, arpc2ko_df]).reset_index(drop=True)
#     sns.violinplot(data=data_bp, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner=None, legend=False)
#     # sns.swarmplot(data=data_bp, x='type', y=column, hue='track_name', palette='deep',legend=False)
#     # data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
#     # data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
#     # U1, p = mannwhitneyu(data1, data2)
#     # plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
#     # plt.legend()
#     plt.xlabel("Type")
#     plt.ylabel("{}".format(column))
#     plt.xticks(rotation=90)

#     plot_name = '{}_stripplot'.format(column)

#     plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
#     plt.clf()

# save_path = './figures/stripplots_protrusions_medianvals_percellovertrack'

# #check to see if the path exists, if not make the directory
# if not os.path.exists(save_path):
#   os.mkdir(save_path)


# for column in params:
#     if column == 'number_protrusions':
#        pixel_size = 1
#     else:
#        pixel_size = 0.645 #microns
#     wt_df = pooled_wt_df[[column, 'type', 'track_name']].copy()
#     wt_df.loc[wt_df[column].apply(lambda x: isinstance(x, list) and len(x) == 0), column] = np.nan
#     wt_df[column] = wt_df[column].apply(np.median) * pixel_size
#     med_wt_df = wt_df.groupby(['track_name', 'type'])[column].median().reset_index()
#     arpc2ko_df = pooled_arpc2ko_df[[column, 'type', 'track_name']].copy()
#     arpc2ko_df.loc[arpc2ko_df[column].apply(lambda x: isinstance(x, list) and len(x) == 0), column] = np.nan
#     arpc2ko_df[column] = arpc2ko_df[column].apply(np.median) * pixel_size
#     med_arpc2ko_df = arpc2ko_df.groupby(['track_name', 'type'])[column].median().reset_index()
#     data_bp = pd.concat([med_wt_df, med_arpc2ko_df]).reset_index(drop=True)
#     sns.violinplot(data=data_bp, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], alpha=0.8, inner=None, legend=False)
#     sns.swarmplot(data=data_bp, x='type', y=column, color=(243/256,243/256,243/256),legend=False)
#     data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
#     data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
#     U1, p = mannwhitneyu(data1, data2)
#     plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
#     plt.legend()
#     plt.xlabel("Type")
#     plt.ylabel("{}".format(column))
#     plt.xticks(rotation=90)

#     plot_name = '{}_stripplot'.format(column)

#     plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
#     plt.clf()