from hmmlearn.hmm import GaussianHMM
# from hmmlearn.hmm import GMMHMM
from scipy.stats import norm
import numpy as np

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

save_path = './figures/hmm_3states_transition_stats'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df = combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df)


pixel_size = 0.645 #microns

# dt = 5*3 #min per frame because 5 minute intervals between frames and we sample every third frame

wt_cells_states, wt_pooled_state_df= predict_states(wt_cells, 'avg_trac_mag')


arpc2ko_cells_states, arpc2ko_pooled_state_df = predict_states(arpc2ko_cells, 'avg_trac_mag')

dt = 15/60 #min between values since sampled every third frame with 5 min intervals between frames then convert from min to hr


def count_transitions(states):
    states = np.asarray(states)
    
    if len(states) < 2:
        return 0
    
    # Compare consecutive elements
    transitions = np.sum(states[1:] != states[:-1])
    
    return transitions

def compute_transition_rate(states, frame_interval=None):
    """
    states: 1D array-like of HMM states
    frame_interval: time per frame (e.g., minutes or seconds)
                    if None → returns transitions per frame
    """
    
    n_transitions = count_transitions(states)
    n_frames = len(states)
    
    if n_frames < 2:
        return np.nan
    
    total_time = n_frames - 1  # number of intervals
    
    if frame_interval is not None:
        total_time = total_time * frame_interval
    
    rate = n_transitions / total_time
    
    return rate

rates_wt = []
n_transitions_list_wt = []

for df in wt_cells_states:
    states = df["state"].values
    
    n_transitions = count_transitions(states)
    rate = compute_transition_rate(states, frame_interval=dt)  # e.g., 1 min/frame
    
    n_transitions_list_wt.append(n_transitions)
    rates_wt.append(rate)

rates_arpc2ko = []
n_transitions_list_arpc2ko = []

for df in arpc2ko_cells_states:
    states = df["state"].values
    
    n_transitions = count_transitions(states)
    rate = compute_transition_rate(states, frame_interval=dt)  # e.g., 1 min/frame
    
    n_transitions_list_arpc2ko.append(n_transitions)
    rates_arpc2ko.append(rate)


rates_wt = np.array(rates_wt)
mean_rate_wt = np.nanmean(rates_wt)
sem_rate_wt = np.nanstd(rates_wt) / np.sqrt(len(rates_wt))

rates_arpc2ko = np.array(rates_arpc2ko)
mean_rate_arpc2ko = np.nanmean(rates_arpc2ko)
sem_rate_arpc2ko = np.nanstd(rates_arpc2ko) / np.sqrt(len(rates_arpc2ko))

save_path = './figures/hmm_3states_transition_stats'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

#clears out sentinel file if it exists
open('{}/transition_rates.txt'.format(save_path),'w').close()
#create new sentinel file to write to
transition_stats = open('{}/transition_rates.txt'.format(save_path),'w')
file_lines = []

file_lines.append('WT: Mean transition per hour {} +/ {} SEM'.format(mean_rate_wt, sem_rate_wt) + '\n')
file_lines.append('ARPC2KO: Mean transition per hour {} +/ {} SEM'.format(mean_rate_arpc2ko, sem_rate_arpc2ko) + '\n')


#write lines to text file 
transition_stats.writelines(file_lines)
transition_stats.close()

width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['WT', 'ARPC2KO']
counts = [mean_rate_wt, mean_rate_arpc2ko]
err = [sem_rate_wt, sem_rate_arpc2ko]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, yerr=err, color=[(128/256,0,0),(0,102/256,102/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Average transition rate per hour')
plt.savefig('{}/avg_transrate_perhour.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()