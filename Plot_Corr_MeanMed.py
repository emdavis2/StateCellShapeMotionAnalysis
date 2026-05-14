import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess

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


protrusion_df = pd.read_pickle(data_dir + "/protrusion.pkl")

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)

pixel_size = 0.645 #microns

avg_arpc2ko_cells = []
for df in arpc2ko_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 15 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    mean_df = df.mean(numeric_only=True)
    mean_df['speed'] = speed
    mean_df['DT'] = DT
    
    avg_arpc2ko_cells.append(mean_df)

avg_wt_cells = []
for df in wt_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 15 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    mean_df = df.mean(numeric_only=True)
    mean_df['speed'] = speed
    mean_df['DT'] = DT
    
    avg_wt_cells.append(mean_df)

pooled_avg_arpc2ko_df = pd.concat(avg_arpc2ko_cells, axis=1, ignore_index=False).T
pooled_avg_wt_df = pd.concat(avg_wt_cells, axis=1, ignore_index=False).T

pooled_avg_arpc2ko_df['type'] = ['ARPC2KO']*len(pooled_avg_arpc2ko_df)
pooled_avg_wt_df['type'] = ['WT']*len(pooled_avg_wt_df)

numeric_pooled_arpc2ko_df = pooled_avg_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_avg_wt_df.select_dtypes(include=np.number)


med_arpc2ko_cells = []
for df in arpc2ko_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 15 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    med_df = df.median(numeric_only=True)
    med_df['speed'] = speed
    med_df['DT'] = DT
    
    med_arpc2ko_cells.append(med_df)

med_wt_cells = []
for df in wt_cells:
    x0, y0 = df.iloc[0][['approximate-medoidx', 'approximate-medoidy']]
    x1, y1 = df.iloc[-1][['approximate-medoidx', 'approximate-medoidy']]
    net_displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) * pixel_size
    
    dx = df['approximate-medoidx'].diff().values[1:]
    dy = df['approximate-medoidy'].diff().values[1:]
    step_lengths = np.sqrt(dx**2 + dy**2)
    path_length = np.sum(step_lengths) * pixel_size
    duration = (len(df) - 1) * 15 #minutes
    speed = path_length / duration if duration > 0 else np.nan
    DT = net_displacement/path_length

    med_df = df.median(numeric_only=True)
    med_df['speed'] = speed
    med_df['DT'] = DT
    
    med_wt_cells.append(med_df)

pooled_med_arpc2ko_df = pd.concat(med_arpc2ko_cells, axis=1, ignore_index=False).T
pooled_med_wt_df = pd.concat(med_wt_cells, axis=1, ignore_index=False).T

pooled_med_arpc2ko_df['type'] = ['ARPC2KO']*len(pooled_med_arpc2ko_df)
pooled_med_wt_df['type'] = ['WT']*len(pooled_med_wt_df)

numeric_pooled_arpc2ko_df = pooled_med_arpc2ko_df.select_dtypes(include=np.number)
numeric_pooled_wt_df = pooled_med_wt_df.select_dtypes(include=np.number)


######################################################################################################
save_path = './figures/corr_plots_wt_only_median'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

A = ['speed','area','DT','eccentricity','solidity','dip_ratio','turning_angle','step_length']
B = ['speed','area','DT','eccentricity','solidity','dip_ratio','turning_angle','step_length']
unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    r, pval = pearsonr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
    rho, p_value = stats.spearmanr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
    plt.plot(pooled_med_wt_df[column1],pooled_med_wt_df[column2],'.',color='maroon')
    plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
    plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))

    sns.regplot(data=pooled_med_wt_df, x=column1, y=column2, scatter=False, ci=None, color="maroon")

    plt.xlabel('{}'.format(column1))
    plt.ylabel('{}'.format(column2))

    plt.legend()

    plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()

A = ['speed','area','DT','eccentricity','solidity','dip_ratio','turning_angle','step_length']
unique_pairs = [('avg_trac_mag', x) for x in A]
for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    r, pval = pearsonr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
    rho, p_value = stats.spearmanr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
    plt.plot(pooled_med_wt_df[column1],pooled_med_wt_df[column2],'.',color='maroon')
    plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
    plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))

    sns.regplot(data=pooled_med_wt_df, x=column1, y=column2, scatter=False, ci=None, color="maroon")


    plt.xlabel('{}'.format(column1))
    plt.ylabel('{}'.format(column2))

    plt.legend()

    plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()

column1 = 'area'
column2 = 'DT'
r, pval = pearsonr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
rho, p_value = stats.spearmanr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
plt.plot(pooled_med_wt_df[column1],pooled_med_wt_df[column2],'.',color='maroon')
plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))
sns.regplot(data=pooled_med_wt_df, x=column1, y=column2, scatter=False, ci=None, color="maroon")
plt.xlabel('{}'.format(column1))
plt.ylabel('{}'.format(column2))
plt.legend()
plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
plt.clf()

column1 = 'solidity'
column2 = 'DT'
r, pval = pearsonr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
rho, p_value = stats.spearmanr(pooled_med_wt_df[column1],pooled_med_wt_df[column2])
plt.plot(pooled_med_wt_df[column1],pooled_med_wt_df[column2],'.',color='maroon')
plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))
sns.regplot(data=pooled_med_wt_df, x=column1, y=column2, scatter=False, ci=None, color="maroon")
plt.xlabel('{}'.format(column1))
plt.ylabel('{}'.format(column2))
plt.legend()
plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
plt.clf()

save_path = './figures/corr_plots_arpc2ko_only_median'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

A = ['speed','area','DT','eccentricity','solidity','dip_ratio','turning_angle','step_length']
B = ['speed','area','DT','eccentricity','solidity','dip_ratio','turning_angle','step_length']
unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    r, pval = pearsonr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
    rho, p_value = stats.spearmanr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
    plt.plot(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2],'.',color=(0,102/256,102/256))
    plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
    plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))

    sns.regplot(data=pooled_med_arpc2ko_df, x=column1, y=column2, scatter=False, ci=None, color=(0,102/256,102/256))


    plt.xlabel('{}'.format(column1))
    plt.ylabel('{}'.format(column2))

    plt.legend()

    plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()

A = ['speed','area','DT','eccentricity','solidity','dip_ratio','turning_angle','step_length']
unique_pairs = [('avg_trac_mag', x) for x in A]
for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    r, pval = pearsonr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
    rho, p_value = stats.spearmanr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
    plt.plot(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2],'.',color=(0,102/256,102/256))
    plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
    plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))

    sns.regplot(data=pooled_med_arpc2ko_df, x=column1, y=column2, scatter=False, ci=None, color=(0,102/256,102/256))


    plt.xlabel('{}'.format(column1))
    plt.ylabel('{}'.format(column2))

    plt.legend()

    plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()


column1 = 'area'
column2 = 'DT'
r, pval = pearsonr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
rho, p_value = stats.spearmanr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
plt.plot(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2],'.',color=(0,102/256,102/256))
plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))
sns.regplot(data=pooled_med_arpc2ko_df, x=column1, y=column2, scatter=False, ci=None, color=(0,102/256,102/256))
plt.xlabel('{}'.format(column1))
plt.ylabel('{}'.format(column2))
plt.legend()
plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
plt.clf()

column1 = 'solidity'
column2 = 'DT'
r, pval = pearsonr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
rho, p_value = stats.spearmanr(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2])
plt.plot(pooled_med_arpc2ko_df[column1],pooled_med_arpc2ko_df[column2],'.',color=(0,102/256,102/256))
plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))
sns.regplot(data=pooled_med_arpc2ko_df, x=column1, y=column2, scatter=False, ci=None, color=(0,102/256,102/256))
plt.xlabel('{}'.format(column1))
plt.ylabel('{}'.format(column2))
plt.legend()
plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
plt.clf()


#######Plot Distributions#############
save_path = './figures/histograms_median'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

save_path_wt = save_path+'/wt'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path_wt):
  os.mkdir(save_path_wt)

save_path_arpc2ko = save_path+'/arpc2ko'
#check to see if the path exists, if not make the directory
if not os.path.exists(save_path_arpc2ko):
  os.mkdir(save_path_arpc2ko)

params = ['speed','area','DT','eccentricity','solidity','dip_ratio','turning_angle','step_length','avg_trac_mag']
for column in params:
    plot_dist(column, pooled_med_wt_df, pooled_med_arpc2ko_df, 'WT', 'ARPC2KO', save_path, column + ' distribution') 
    plot_dist_onetype(column, pooled_med_wt_df, 'WT', save_path_wt, column + ' distribution')
    plot_dist_onetype(column, pooled_med_arpc2ko_df, 'ARPC2KO', save_path_arpc2ko, column + ' distribution')