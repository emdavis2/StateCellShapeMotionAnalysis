import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr

from get_data_functions import *
from assemble_data_functions import *
from plot_dist_functions import *

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

save_path = './figures'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

#import data 
data_dir = './data'
dipole_dict = np.load(data_dir + '/dipole_dict.npy', allow_pickle=True).item()
ellipse_dict = np.load(data_dir + '/ellipse_dict.npy', allow_pickle=True).item()
quad_dict = np.load(data_dir + '/quad_dict.npy', allow_pickle=True).item()

tracksgeo_path = data_dir + '/tracks_geo_region.pkl'
tracksgeo_dict = load_tracksgeo(tracksgeo_path, dipole_dict)

protrusion_df = pd.read_pickle(data_dir + "/protrusion_force.pkl")

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)

numeric_pooled_arpc2ko_df = pooled_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_wt_df.select_dtypes(include=np.number)

for df in arpc2ko_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_arpc2ko_df = pd.concat(arpc2ko_cells, ignore_index=True)

for df in wt_cells:
   df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)

pooled_wt_df = pd.concat(wt_cells, ignore_index=True)


params = ['area','avg_trac_mag','eccentricity','solidity','dip_ratio','step_length','width','length','protrusion_id','perim_ratio','perimeter','perim_area_ratio','protrusion_act','retraction_act','all_act','mean_wid','mean_len','mean_num','max_len','mean_protr_lt']

wt_median_summary_df_dir = [] 
wt_median_summary_df_abs = [] 
for df in wt_cells:

    diff = df[params].diff()
    medians = diff.median(numeric_only = True)        
    medians.loc['track_name']=df['track_name'].iloc[0]
    medians.loc['type']=df['type'].iloc[0]
    wt_median_summary_df_dir.append(medians)

    diff = df[params].diff().abs()
    medians = diff.median(numeric_only = True)        
    medians.loc['track_name']=df['track_name'].iloc[0]
    medians.loc['type']=df['type'].iloc[0]
    wt_median_summary_df_abs.append(medians)

wt_median_summary_df_dir = pd.concat(wt_median_summary_df_dir,axis=1).T
wt_median_summary_df_abs = pd.concat(wt_median_summary_df_abs,axis=1).T

arpc2ko_median_summary_df_dir = []
arpc2ko_median_summary_df_abs = []
for df in arpc2ko_cells:
    
    diff = df[params].diff()
    medians = diff.median(numeric_only = True)          
    medians.loc['track_name']=df['track_name'].iloc[0]
    medians.loc['type']=df['type'].iloc[0]
    arpc2ko_median_summary_df_dir.append(medians)

    diff = df[params].diff().abs()
    medians = diff.median(numeric_only = True)          
    medians.loc['track_name']=df['track_name'].iloc[0]
    medians.loc['type']=df['type'].iloc[0]
    arpc2ko_median_summary_df_abs.append(medians)

arpc2ko_median_summary_df_dir = pd.concat(arpc2ko_median_summary_df_dir,axis=1).T
arpc2ko_median_summary_df_abs = pd.concat(arpc2ko_median_summary_df_abs,axis=1).T

wt_median_summary_df_abs[params] = wt_median_summary_df_abs[params].apply(
    pd.to_numeric, errors='coerce'
)

wt_median_summary_df_dir[params] = wt_median_summary_df_dir[params].apply(
    pd.to_numeric, errors='coerce'
)

arpc2ko_median_summary_df_abs[params] = arpc2ko_median_summary_df_abs[params].apply(
    pd.to_numeric, errors='coerce'
)

arpc2ko_median_summary_df_dir[params] = arpc2ko_median_summary_df_dir[params].apply(
    pd.to_numeric, errors='coerce'
)


save_path = './figures/diff_shape_wt_arpc2ko'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)


save_path = './figures/diff_shape_wt_arpc2ko/pdf_vp'

save_path_bp = './figures/diff_shape_wt_arpc2ko/pdf_bp'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path_bp):
  os.mkdir(save_path_bp)


data_bp = pd.concat([wt_median_summary_df_abs, arpc2ko_median_summary_df_abs]).reset_index()
for column in params: #numeric_pooled_arpc2ko_df.columns:
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

    ######################################################################################

    sns.boxplot(data=data_bp, x='type', y=column, hue="type", palette=[(128/256,0,0),(0,102/256,102/256)], legend=False)
    sns.swarmplot(data=data_bp, x='type', y=column, color=(243/256,243/256,243/256),linewidth=.5,edgecolor="black",legend=False)
    data1 = data_bp[data_bp['type'] == 'WT']['{}'.format(column)].astype(float).dropna()
    data2 = data_bp[data_bp['type'] == 'ARPC2KO']['{}'.format(column)].astype(float).dropna()
    U1, p = mannwhitneyu(data1, data2)
    plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
    plt.legend()
    plt.xlabel("Type")
    plt.ylabel("{}".format(column))
    plt.xticks(rotation=90)

    plot_name = '{}_boxplot'.format(column)

    plt.savefig('{}/{}.pdf'.format(save_path_bp, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()

column1 = 'avg_trac_mag'
column2 = 'area'
r_wt, pval_wt = pearsonr(wt_median_summary_df_abs[column1],wt_median_summary_df_abs[column2])
r_arpc2ko, pval_arpc2ko = pearsonr(arpc2ko_median_summary_df_abs[column1],arpc2ko_median_summary_df_abs[column2])
# rho, p_value = stats.spearmanr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
plt.plot(wt_median_summary_df_abs[column1],wt_median_summary_df_abs[column2],'.',color='maroon')
plt.plot(arpc2ko_median_summary_df_abs[column1],arpc2ko_median_summary_df_abs[column2],'.',color=(0,102/256,102/256))
plt.plot([],[], ' ', label='r wt {}'.format(np.round(r_wt,2)))
plt.plot([],[], ' ', label='p value wt {}'.format(np.round(pval_wt,5)))
plt.plot([],[], ' ', label='r arpc2ko {}'.format(np.round(r_arpc2ko,2)))
plt.plot([],[], ' ', label='p value arpc2ko {}'.format(np.round(pval_arpc2ko,5)))

sns.regplot(data=wt_median_summary_df_abs, x=column1, y=column2, scatter=False, ci=None, color="maroon")
sns.regplot(data=arpc2ko_median_summary_df_abs, x=column1, y=column2, scatter=False, ci=None, color=(0,102/256,102/256))

plt.xlabel('{}'.format(column1))
plt.ylabel('{}'.format(column2))
plt.legend()
plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
plt.clf()



save_path = './figures/diff_shape_wt_arpc2ko/direction_pdf_vp'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

data_bp = pd.concat([wt_median_summary_df_dir, arpc2ko_median_summary_df_dir]).reset_index()
for column in params: #numeric_pooled_arpc2ko_df.columns:
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



column1 = 'avg_trac_mag'
column2 = 'area'
r_wt, pval_wt = pearsonr(wt_median_summary_df_dir[column1],wt_median_summary_df_dir[column2])
r_arpc2ko, pval_arpc2ko = pearsonr(arpc2ko_median_summary_df_dir[column1],arpc2ko_median_summary_df_dir[column2])
# rho, p_value = stats.spearmanr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
plt.plot(wt_median_summary_df_dir[column1],wt_median_summary_df_dir[column2],'.',color='maroon')
plt.plot(arpc2ko_median_summary_df_dir[column1],arpc2ko_median_summary_df_dir[column2],'.',color=(0,102/256,102/256))
plt.plot([],[], ' ', label='r wt {}'.format(np.round(r_wt,2)))
plt.plot([],[], ' ', label='p value wt {}'.format(np.round(pval_wt,5)))
plt.plot([],[], ' ', label='r arpc2ko {}'.format(np.round(r_arpc2ko,2)))
plt.plot([],[], ' ', label='p value arpc2ko {}'.format(np.round(pval_arpc2ko,5)))

sns.regplot(data=wt_median_summary_df_dir, x=column1, y=column2, scatter=False, ci=None, color="maroon")
sns.regplot(data=arpc2ko_median_summary_df_dir, x=column1, y=column2, scatter=False, ci=None, color=(0,102/256,102/256))

plt.xlabel('{}'.format(column1))
plt.ylabel('{}'.format(column2))
plt.legend()
plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
plt.clf()