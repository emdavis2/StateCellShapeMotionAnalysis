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


###################HMM States###############################################################

save_path = './figures/compare_filtarea_protrusions'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

params = ['width','length','protrusion_id']

wt_dfs = []
arpc2ko_dfs = []

pixel_size = 0.645 #microns

filt_area_val = np.arange(10, 1000, 10)
# filt_area_val = np.arange(1000, 10000, 100)

for val in filt_area_val:

    min_protr_area = val
    filtered_protrusion_df = protrusion_df[protrusion_df['protr_area']>min_protr_area]

    arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, filtered_protrusion_df)


    wt_df = pooled_wt_df

    arpc2ko_df = pooled_arpc2ko_df

    wt_dfs.append(wt_df[params].median())
    arpc2ko_dfs.append(arpc2ko_df[params].median())



colors_for_state_plot = [(242/256,140/256,40/256), (11/256,218/256,81/256), (255/256,0,255/256)] 

wt_dfs = pd.DataFrame(wt_dfs)
arpc2ko_dfs = pd.DataFrame(arpc2ko_dfs)

filt_area_val_um = filt_area_val * (pixel_size**2)

colors = [(128/256,0,0),(0,102/256,102/256)]

for column in params:
    plt.plot(filt_area_val_um, wt_dfs[column],label='WT',color=colors[0],marker='o', linestyle='-')
    plt.plot(filt_area_val_um, arpc2ko_dfs[column],label='Arpc2KO',color=colors[1],marker='o', linestyle='-')
    plt.title('{}'.format(column))
    plt.xlabel('minimum area (um)')
    plt.ylabel('{}'.format(column))
    plt.legend()
    plot_name = '{}_compare_filtval'.format(column)
    plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
    plt.clf()
