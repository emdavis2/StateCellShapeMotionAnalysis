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

### want 25 pixels as my cut off length for definition of protrusion
threshold = 25

pixel_size = 0.645 #microns

dt = 5*3 #min per frame because 5 minute intervals between frames and we sample every third frame

# for df in arpc2ko_cells:
#    df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)
#    df['filtered_num_protrusions'] = df["protrusion_lengths"].apply(lambda lst: sum(x > threshold for x in lst))

pooled_arpc2ko_df = pd.concat(arpc2ko_cells, ignore_index=True)

# for df in wt_cells:
#    df['track_name'] = [(df['experiment'].iloc[0] + '_movie' + str(int(df['movie'].iloc[0])) + '_track' + str(int(df['track_id'].iloc[0])))] * len(df)
#    df['filtered_num_protrusions'] = df["protrusion_lengths"].apply(lambda lst: sum(x > threshold for x in lst))

pooled_wt_df = pd.concat(wt_cells, ignore_index=True)


wt_cells_states, wt_pooled_state_df= predict_states(wt_cells, 'avg_trac_mag')


arpc2ko_cells_states, arpc2ko_pooled_state_df = predict_states(arpc2ko_cells, 'avg_trac_mag')

dt = 15/60 #min between values since sampled every third frame with 5 min intervals between frames then convert from min to hr

low_to_high_wt = 0
high_to_low_wt = 0
low_to_med_wt = 0
med_to_low_wt = 0
med_to_high_wt = 0
high_to_med_wt = 0
no_transition_wt = 0
num_transitions_wt = []
time_spent_in_low_wt = []
time_spent_in_med_wt = []
time_spent_in_high_wt = []
for df in wt_cells_states:
    states = df['state']
    states = states.mask(states==2,5)
    name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
    if len(np.unique(states)) > 1:
        num_transitions_wt.append(np.sum(np.diff(states) != 0))
        low_to_med_wt += np.count_nonzero(np.diff(states) == 1)
        med_to_low_wt += np.count_nonzero(np.diff(states) == -1)
        med_to_high_wt += np.count_nonzero(np.diff(states) == 4)
        high_to_med_wt += np.count_nonzero(np.diff(states) == -4)
        high_to_low_wt += np.count_nonzero(np.diff(states) == -5)
        low_to_high_wt += np.count_nonzero(np.diff(states) == 5)
    else:
       no_transition_wt += 1
    
    time_spent_in_low_wt.append(len(np.where(states == 0)[0]) * dt)
    time_spent_in_med_wt.append(len(np.where(states == 1)[0]) * dt)
    time_spent_in_high_wt.append(len(np.where(states == 5)[0]) * dt)

transition_cells_wt = len(num_transitions_wt)
total_num_wt = transition_cells_wt + no_transition_wt

low_to_high_arpc2ko = 0
high_to_low_arpc2ko = 0
low_to_med_arpc2ko = 0
med_to_low_arpc2ko = 0
med_to_high_arpc2ko = 0
high_to_med_arpc2ko = 0
no_transition_arpc2ko = 0
num_transitions_arpc2ko = []
time_spent_in_low_arpc2ko = []
time_spent_in_med_arpc2ko = []
time_spent_in_high_arpc2ko = []
for df in arpc2ko_cells_states:
    states = df['state']
    states = states.mask(states==2,5)
    name = df['experiment'][0] +'_movie'+str(int(df['movie'][0])) + '_track'+str(int(df['track_id'][0]))
    if len(np.unique(states)) > 1:
        num_transitions_arpc2ko.append(np.sum(np.diff(states) != 0))
        low_to_med_arpc2ko += np.count_nonzero(np.diff(states) == 1)
        med_to_low_arpc2ko += np.count_nonzero(np.diff(states) == -1)
        med_to_high_arpc2ko += np.count_nonzero(np.diff(states) == 4)
        high_to_med_arpc2ko += np.count_nonzero(np.diff(states) == -4)
        high_to_low_arpc2ko += np.count_nonzero(np.diff(states) == -5)
        low_to_high_arpc2ko += np.count_nonzero(np.diff(states) == 5)
    else:
       no_transition_arpc2ko += 1

    time_spent_in_low_arpc2ko.append(len(np.where(states == 0)[0]) * dt)
    time_spent_in_med_arpc2ko.append(len(np.where(states == 1)[0]) * dt)
    time_spent_in_high_arpc2ko.append(len(np.where(states == 5)[0]) * dt)

transition_cells_arpc2ko = len(num_transitions_arpc2ko)
total_num_arpc2ko = transition_cells_arpc2ko + no_transition_arpc2ko


#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['No state transition', 'State transition']
counts = [no_transition_wt, transition_cells_wt]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width)
ax.bar_label(p, label_type='center')
ax.set_title('Number of cell transitions')
plt.savefig('{}/WT_transitions.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['No state transition', 'State transition']
counts = [no_transition_arpc2ko, transition_cells_arpc2ko]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width)
ax.bar_label(p, label_type='center')
ax.set_title('Number of cell transitions')
plt.savefig('{}/ARPC2KO_transitions.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['WT', 'ARPC2KO']
counts = [transition_cells_wt/total_num_wt, transition_cells_arpc2ko/total_num_arpc2ko]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, color=[(128/256,0,0),(0,102/256,102/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Fraction of cells that undergo transitions')
plt.savefig('{}/fraction_transitions.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['WT', 'ARPC2KO']
counts = [np.sum(num_transitions_wt)/total_num_wt, np.sum(num_transitions_arpc2ko)/total_num_arpc2ko]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, color=[(128/256,0,0),(0,102/256,102/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Average number of transitions per cell')
plt.savefig('{}/avg_num_transitions_per_cell.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for comparing number of cells that transition
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['WT', 'ARPC2KO']
counts = [(np.sum(num_transitions_wt)+total_num_wt)/total_num_wt, (np.sum(num_transitions_arpc2ko)+total_num_arpc2ko)/total_num_arpc2ko]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, color=[(128/256,0,0),(0,102/256,102/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Average number of segments per cell')
plt.savefig('{}/avg_num_segments_per_cell.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for average time spent in each state
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['WT', 'ARPC2KO']
counts = [np.average(time_spent_in_low_wt), np.average(time_spent_in_low_arpc2ko)]
err = [np.nanstd(time_spent_in_low_wt) / np.sqrt(len(time_spent_in_low_wt)), np.nanstd(time_spent_in_low_arpc2ko) / np.sqrt(len(time_spent_in_low_arpc2ko))]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, yerr=err, color=[(128/256,0,0),(0,102/256,102/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Average time spent in low force state')
plt.savefig('{}/avg_time_low_force_state.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['WT', 'ARPC2KO']
counts = [np.average(time_spent_in_med_wt), np.average(time_spent_in_med_arpc2ko)]
err = [np.nanstd(time_spent_in_med_wt) / np.sqrt(len(time_spent_in_med_wt)), np.nanstd(time_spent_in_med_arpc2ko) / np.sqrt(len(time_spent_in_med_arpc2ko))]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, yerr=err, color=[(128/256,0,0),(0,102/256,102/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Average time spent in med force state')
plt.savefig('{}/avg_time_med_force_state.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#Plot bar charts for average time spent in each state
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['WT', 'ARPC2KO']
counts = [np.average(time_spent_in_high_wt), np.average(time_spent_in_high_arpc2ko)]
err = [np.nanstd(time_spent_in_high_wt) / np.sqrt(len(time_spent_in_high_wt)), np.nanstd(time_spent_in_high_arpc2ko) / np.sqrt(len(time_spent_in_high_arpc2ko))]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, yerr=err, color=[(128/256,0,0),(0,102/256,102/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Average time spent in high force state')
plt.savefig('{}/avg_time_high_force_state.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()