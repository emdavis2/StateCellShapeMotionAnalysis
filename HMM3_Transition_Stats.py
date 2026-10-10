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


protrusion_df = pd.read_pickle(data_dir + "/protrusion_force.pkl")

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

#Plot bar charts for average time spent in each state
width = 0.6  # the width of the bars: can also be len(x) sequence
categories = ['Low', 'Med', 'High']
counts = [np.average(time_spent_in_low_wt), np.average(time_spent_in_med_wt), np.average(time_spent_in_high_wt)]
err = [np.nanstd(time_spent_in_low_wt) / np.sqrt(len(time_spent_in_low_wt)), np.nanstd(time_spent_in_med_wt) / np.sqrt(len(time_spent_in_med_wt)), np.nanstd(time_spent_in_high_wt) / np.sqrt(len(time_spent_in_high_wt))]
fig, ax = plt.subplots()
p = ax.bar(categories, counts, width, yerr=err, color=[(242/256,140/256,40/256), (11/256,218/256,81/256), (255/256,0,255/256)])
ax.bar_label(p, label_type='center')
ax.set_title('Average time spent in each force state')
plt.savefig('{}/avg_time_each_force_state.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

#########################VIOLIN PLOTS###############################

values = np.concatenate([time_spent_in_low_wt, time_spent_in_low_arpc2ko])
labels = ['WT'] * len(time_spent_in_low_wt) + ['ARPC2KO'] * len(time_spent_in_low_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = time_spent_in_low_wt
data2 = time_spent_in_low_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Time spent in low force state (hr)")
plt.xticks(rotation=90)
plot_name = '{}_stripplot'.format('time_low_force_vp')
plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()


values = np.concatenate([time_spent_in_med_wt, time_spent_in_med_arpc2ko])
labels = ['WT'] * len(time_spent_in_med_wt) + ['ARPC2KO'] * len(time_spent_in_med_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = time_spent_in_med_wt
data2 = time_spent_in_med_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Time spent in med force state (hr)")
plt.xticks(rotation=90)
plot_name = '{}_stripplot'.format('time_med_force_vp')
plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

values = np.concatenate([time_spent_in_high_wt, time_spent_in_high_arpc2ko])
labels = ['WT'] * len(time_spent_in_high_wt) + ['ARPC2KO'] * len(time_spent_in_high_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = time_spent_in_high_wt
data2 = time_spent_in_high_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Time spent in high force state (hr)")
plt.xticks(rotation=90)
plot_name = '{}_stripplot'.format('time_high_force_vp')
plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()




values = np.concatenate([time_spent_in_low_wt, time_spent_in_med_wt, time_spent_in_high_wt])
labels = ['Low'] * len(time_spent_in_low_wt) + ['Med'] * len(time_spent_in_med_wt) + ['High'] * len(time_spent_in_high_wt)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(242/256,140/256,40/256), (11/256,218/256,81/256), (255/256,0,255/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = time_spent_in_low_wt
data2 = time_spent_in_med_wt
data3 = time_spent_in_high_wt
U1, p12 = mannwhitneyu(data1, data2)
U2, p13 = mannwhitneyu(data1, data3)
U2, p23 = mannwhitneyu(data2, data3)
plt.plot([],[], ' ', label ="p12 val: {}".format(np.round(p12,4)))
plt.plot([],[], ' ', label ="p13 val: {}".format(np.round(p13,4)))
plt.plot([],[], ' ', label ="p23 val: {}".format(np.round(p23,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Time spent in each force state (hr)")
plt.xticks(rotation=90)
plot_name = '{}_stripplot'.format('time_each_force_vp')
plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()



######################Calculate Rates Between States#######################

def count_transitions_in_state(states):
    states = np.asarray(states)
    
    if len(states) < 2:
        return 0
    
    # Compare consecutive elements
    transitions_low_med = np.sum(np.count_nonzero(np.diff(states) == 1))
    transitions_med_low = np.sum(np.count_nonzero(np.diff(states) == -1))
    transitions_low_high = np.sum(np.count_nonzero(np.diff(states) == 5))
    transitions_high_low = np.sum(np.count_nonzero(np.diff(states) == -5))
    transitions_med_high = np.sum(np.count_nonzero(np.diff(states) == 4))
    transitions_high_med = np.sum(np.count_nonzero(np.diff(states) == -4))
    
    
    return transitions_low_med, transitions_med_low, transitions_low_high, transitions_high_low, transitions_med_high, transitions_high_med


def compute_transition_rate(states, frame_interval=None):
    """
    states: 1D array-like of HMM states
    frame_interval: time per frame (e.g., minutes or seconds)
                    if None → returns transitions per frame
    """
    
    transitions_low_med, transitions_med_low, transitions_low_high, transitions_high_low, transitions_med_high, transitions_high_med = count_transitions_in_state(states)
    n_frames = len(states)
    
    if n_frames < 2:
        return np.nan
    
    total_time = n_frames - 1  # number of intervals
    
    if frame_interval is not None:
        total_time = total_time * frame_interval
    
    rate_low_med = transitions_low_med / total_time
    rate_med_low = transitions_med_low / total_time
    rate_low_high = transitions_low_high / total_time
    rate_high_low = transitions_high_low / total_time
    rate_med_high = transitions_med_high / total_time
    rate_high_med = transitions_high_med / total_time
    
    return rate_low_med, rate_med_low, rate_low_high, rate_high_low, rate_med_high, rate_high_med

rates_low_med_wt = []
rates_med_low_wt = []
rates_low_high_wt = []
rates_high_low_wt = []
rates_med_high_wt = []
rates_high_med_wt = []

for df in wt_cells_states:
    states = df['state']
    states = states.mask(states==2,5)
    
    rate_low_med, rate_med_low, rate_low_high, rate_high_low, rate_med_high, rate_high_med = compute_transition_rate(states, frame_interval=dt)  # e.g., 1 min/frame
    
    rates_low_med_wt.append(rate_low_med)
    rates_med_low_wt.append(rate_med_low)
    rates_low_high_wt.append(rate_low_high)
    rates_high_low_wt.append(rate_high_low)
    rates_med_high_wt.append(rate_med_high)
    rates_high_med_wt.append(rate_high_med)

rates_low_med_arpc2ko = []
rates_med_low_arpc2ko = []
rates_low_high_arpc2ko = []
rates_high_low_arpc2ko = []
rates_med_high_arpc2ko = []
rates_high_med_arpc2ko = []

for df in arpc2ko_cells_states:
    states = df["state"]
    states = states.mask(states==2,5)
    
    rate_low_med, rate_med_low, rate_low_high, rate_high_low, rate_med_high, rate_high_med = compute_transition_rate(states, frame_interval=dt)  # e.g., 1 min/frame
    
    rates_low_med_arpc2ko.append(rate_low_med)
    rates_med_low_arpc2ko.append(rate_med_low)
    rates_low_high_arpc2ko.append(rate_low_high)
    rates_high_low_arpc2ko.append(rate_high_low)
    rates_med_high_arpc2ko.append(rate_med_high)
    rates_high_med_arpc2ko.append(rate_high_med)

########################

values = np.concatenate([rates_low_med_wt, rates_low_med_arpc2ko])
labels = ['WT'] * len(rates_low_med_wt) + ['ARPC2KO'] * len(rates_low_med_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_low_med_wt
data2 = rates_low_med_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from low to med per hour")
plt.xticks(rotation=90)

plot_name = '{}_stripplot'.format('num_trans_low_med_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################

values = np.concatenate([rates_med_low_wt, rates_med_low_arpc2ko])
labels = ['WT'] * len(rates_med_low_wt) + ['ARPC2KO'] * len(rates_med_low_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_med_low_wt
data2 = rates_med_low_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from med to low per hour")
plt.xticks(rotation=90)

plot_name = '{}_stripplot'.format('num_trans_med_low_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################

values = np.concatenate([rates_low_high_wt, rates_low_high_arpc2ko])
labels = ['WT'] * len(rates_low_high_wt) + ['ARPC2KO'] * len(rates_low_high_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_low_high_wt
data2 = rates_low_high_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from low to high per hour")
plt.xticks(rotation=90)

plot_name = '{}_stripplot'.format('num_trans_low_high_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################

values = np.concatenate([rates_high_low_wt, rates_high_low_arpc2ko])
labels = ['WT'] * len(rates_high_low_wt) + ['ARPC2KO'] * len(rates_high_low_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_high_low_wt
data2 = rates_high_low_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from high to low per hour")
plt.xticks(rotation=90)

plot_name = '{}_stripplot'.format('num_trans_high_low_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################

values = np.concatenate([rates_med_high_wt, rates_med_high_arpc2ko])
labels = ['WT'] * len(rates_med_high_wt) + ['ARPC2KO'] * len(rates_med_high_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_med_high_wt
data2 = rates_med_high_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from med to high per hour")
plt.xticks(rotation=90)

plot_name = '{}_stripplot'.format('num_trans_med_high_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################

values = np.concatenate([rates_high_med_wt, rates_high_med_arpc2ko])
labels = ['WT'] * len(rates_high_med_wt) + ['ARPC2KO'] * len(rates_high_med_arpc2ko)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_high_med_wt
data2 = rates_high_med_arpc2ko
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from high to med per hour")
plt.xticks(rotation=90)

plot_name = '{}_stripplot'.format('num_trans_high_med_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

###################################################################################


# ---- Compute means and errors ----
means_wt = [
    round(np.nanmean(time_spent_in_low_wt),1),
    round(np.nanmean(time_spent_in_med_wt),1),
    round(np.nanmean(time_spent_in_high_wt),1)
]

means_ko = [
    round(np.nanmean(time_spent_in_low_arpc2ko),1),
    round(np.nanmean(time_spent_in_med_arpc2ko),1),
    round(np.nanmean(time_spent_in_high_arpc2ko),1)
]

err_wt = [
    np.nanstd(time_spent_in_low_wt) / np.sqrt(len(time_spent_in_low_wt)),
    np.nanstd(time_spent_in_med_wt) / np.sqrt(len(time_spent_in_med_wt)),
    np.nanstd(time_spent_in_high_wt) / np.sqrt(len(time_spent_in_high_wt))
]

err_ko = [
    np.nanstd(time_spent_in_low_arpc2ko) / np.sqrt(len(time_spent_in_low_arpc2ko)),
    np.nanstd(time_spent_in_med_arpc2ko) / np.sqrt(len(time_spent_in_med_arpc2ko)),
    np.nanstd(time_spent_in_high_arpc2ko) / np.sqrt(len(time_spent_in_high_arpc2ko))
]

# ---- X positions ----
x = np.arange(3)  # 3 force states: low, med, high
width = 0.35      # narrower so bars sit side-by-side

# ---- Plot ----
fig, ax = plt.subplots()

bars_wt = ax.bar(x - width/2, means_wt, width, yerr=err_wt,
                 label='WT', color=(128/256, 0, 0))

bars_ko = ax.bar(x + width/2, means_ko, width, yerr=err_ko,
                 label='ARPC2KO', color=(0, 102/256, 102/256))

# ---- Labels & formatting ----
ax.set_xticks(x)
ax.set_xticklabels(['Low', 'Medium', 'High'])
ax.set_xlabel('Force State')
ax.set_ylabel('Average Time (hr)')
ax.set_title('Average Time Spent in Each Force State')
ax.legend()

ax.bar_label(bars_wt, padding=3)
ax.bar_label(bars_ko, padding=3)

plt.savefig(f'{save_path}/avg_time_all_force_states.pdf',
            bbox_inches='tight', format='pdf')
plt.close()



########################
      #Non zero#
########################
rates_low_med_wt_nz = [x for x in rates_low_med_wt if x != 0]
rates_low_med_arpc2ko_nz = [x for x in rates_low_med_arpc2ko if x != 0]

values = np.concatenate([rates_low_med_wt_nz, rates_low_med_arpc2ko_nz])
labels = ['WT'] * len(rates_low_med_wt_nz) + ['ARPC2KO'] * len(rates_low_med_arpc2ko_nz)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_low_med_wt_nz
data2 = rates_low_med_arpc2ko_nz
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from low to med per hour")
plt.xticks(rotation=90)

plot_name = 'nz_{}_stripplot'.format('num_trans_low_med_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################

rates_med_low_wt_nz = [x for x in rates_med_low_wt if x != 0]
rates_med_low_arpc2ko_nz = [x for x in rates_med_low_arpc2ko if x != 0]

values = np.concatenate([rates_med_low_wt_nz, rates_med_low_arpc2ko_nz])
labels = ['WT'] * len(rates_med_low_wt_nz) + ['ARPC2KO'] * len(rates_med_low_arpc2ko_nz)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_med_low_wt_nz
data2 = rates_med_low_arpc2ko_nz
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from med to low per hour")
plt.xticks(rotation=90)

plot_name = 'nz_{}_stripplot'.format('num_trans_med_low_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################
# rates_low_high_wt_nz = [x for x in rates_low_high_wt if x != 0]
# rates_low_high_arpc2ko_nz = [x for x in rates_low_high_arpc2ko if x != 0]

# values = np.concatenate([rates_low_high_wt_nz, rates_low_high_arpc2ko_nz])
# labels = ['WT'] * len(rates_low_high_wt_nz) + ['ARPC2KO'] * len(rates_low_high_arpc2ko_nz)

# sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
# sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
# data1 = rates_low_high_wt_nz
# data2 = rates_low_high_arpc2ko_nz
# U1, p = mannwhitneyu(data1, data2)
# plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
# plt.legend()
# plt.xlabel("Type")
# plt.ylabel("Number of transitions from low to high per hour")
# plt.xticks(rotation=90)

# plot_name = 'nz_{}_stripplot'.format('num_trans_low_high_per_hour_vp')

# plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
# plt.clf()

# ########################
# rates_high_low_wt_nz = [x for x in rates_high_low_wt if x != 0]
# rates_high_low_arpc2ko_nz = [x for x in rates_high_low_arpc2ko if x != 0]

# values = np.concatenate([rates_high_low_wt_nz, rates_high_low_arpc2ko_nz])
# labels = ['WT'] * len(rates_high_low_wt_nz) + ['ARPC2KO'] * len(rates_high_low_arpc2ko_nz)

# sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
# sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
# data1 = rates_high_low_wt_nz
# data2 = rates_high_low_arpc2ko_nz
# U1, p = mannwhitneyu(data1, data2)
# plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
# plt.legend()
# plt.xlabel("Type")
# plt.ylabel("Number of transitions from high to low per hour")
# plt.xticks(rotation=90)

# plot_name = 'nz_{}_stripplot'.format('num_trans_high_low_per_hour_vp')

# plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
# plt.clf()

########################
rates_med_high_wt_nz = [x for x in rates_med_high_wt if x != 0]
rates_med_high_arpc2ko_nz = [x for x in rates_med_high_arpc2ko if x != 0]

values = np.concatenate([rates_med_high_wt_nz, rates_med_high_arpc2ko_nz])
labels = ['WT'] * len(rates_med_high_wt_nz) + ['ARPC2KO'] * len(rates_med_high_arpc2ko_nz)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_med_high_wt_nz
data2 = rates_med_high_arpc2ko_nz
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from med to high per hour")
plt.xticks(rotation=90)

plot_name = 'nz_{}_stripplot'.format('num_trans_med_high_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

########################
rates_high_med_wt_nz = [x for x in rates_high_med_wt if x != 0]
rates_high_med_arpc2ko_nz = [x for x in rates_high_med_arpc2ko if x != 0]

values = np.concatenate([rates_high_med_wt_nz, rates_high_med_arpc2ko_nz])
labels = ['WT'] * len(rates_high_med_wt_nz) + ['ARPC2KO'] * len(rates_high_med_arpc2ko_nz)

sns.violinplot(x=labels, y=values, hue=labels, palette=[(128/256,0,0),(0,102/256,102/256)],alpha=0.8, inner='quart', legend=False)
sns.swarmplot(x=labels, y=values, color=(243/256,243/256,243/256),legend=False)
data1 = rates_high_med_wt_nz
data2 = rates_high_med_arpc2ko_nz
U1, p = mannwhitneyu(data1, data2)
plt.plot([],[], ' ', label ="p val: {}".format(np.round(p,4)))
plt.legend()
plt.xlabel("Type")
plt.ylabel("Number of transitions from high to med per hour")
plt.xticks(rotation=90)

plot_name = 'nz_{}_stripplot'.format('num_trans_high_med_per_hour_vp')

plt.savefig('{}/{}.pdf'.format(save_path, plot_name),bbox_inches='tight',format='pdf')
plt.clf()

###################################################################################
