import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.io import imread, imsave, imshow

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 # Also set for PostScript exports
mpl.rcParams['font.family'] = 'arial'

# function to get center of cell using binary mask
# uses the approximate medoid method to calculate cell center
def get_center(cell_mask):
    y, x = np.nonzero(cell_mask)
    
    ym_temp, xm_temp = np.median(y), np.median(x)
    imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
    ym, xm = y[imin], x[imin]
    
    return (ym, xm)

def plot_1d_hmm_state_trace_and_probs(avg_trac_mag, hard_states, probs, cell_name, timepoints, save_path):
    n_states = 2
    # timepoints = np.arange(len(hard_states))

    states = ['low force', 'high force']

    probs_T = probs.T  # shape: (n_states, n_timepoints)

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})

    axes[0].plot(timepoints, avg_trac_mag, lw=2)
    axes[0].set_ylabel("Average traction magnitude (Pa)")
    axes[0].set_title(f"Average traction magnitude — {cell_name}")
    axes[0].grid(True)

    # --- Hard state trace ---
    axes[1].plot(timepoints, hard_states, drawstyle="steps-mid", lw=2)
    axes[1].set_ylabel("State")
    axes[1].set_yticks(range(n_states))
    axes[1].set_title(f"HMM State Trace — {cell_name}")
    axes[1].grid(True)

    # --- Stacked soft probabilities ---
    axes[2].stackplot(
        timepoints,
        probs_T,
        labels=[f"State {states[i]}" for i in range(n_states)],
        alpha=0.85
    )
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("P(State | Observation)")
    axes[2].set_title(f"Soft State Probabilities — {cell_name}")
    axes[2].legend(loc="upper right", bbox_to_anchor=(1.12, 1.0))
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('{}/1d_hmm_state_trace.pdf'.format(save_path),bbox_inches='tight',format='pdf')
    plt.clf()

def colored_line(ax, x, y, c, cmap="viridis", lw=2):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segs, cmap=cmap,
                        norm=Normalize(c.min(), c.max()))
    lc.set_array(c[:-1])
    lc.set_linewidth(lw)

    ax.add_collection(lc)
    ax.autoscale()
    return lc


def plot_1d_hmm_state_trace_and_probs_coloredby_avgtracmag(avg_trac_mag, hard_states, probs, cell_name, timepoints, save_path):
    n_states = 2
    # timepoints = np.arange(len(hard_states))

    states = ['low force', 'high force']

    probs_T = probs.T  # shape: (n_states, n_timepoints)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False, gridspec_kw={"height_ratios": [1, 1]})

    # --- Hard state trace ---
    lc = colored_line(axes[0], timepoints, hard_states, avg_trac_mag)
    fig.colorbar(lc, ax=axes[0], label="Average traction magnitude (Pa)")
    axes[0].set_ylabel("State")
    axes[0].set_yticks(range(n_states))
    axes[0].set_title(f"HMM State Trace — {cell_name}")
    axes[0].grid(True)

    # --- Stacked soft probabilities ---
    axes[1].stackplot(
        timepoints,
        probs_T,
        labels=[f"State {states[i]}" for i in range(n_states)],
        alpha=0.85
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("P(State | Observation)")
    axes[1].set_title(f"Soft State Probabilities — {cell_name}")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.12, 1.0))
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('{}/1d_hmm_state_trace_coloredby_avgtracmag.pdf'.format(save_path),bbox_inches='tight',format='pdf')
    plt.clf()

def plot_1d_hmm_state_trace_and_probs_coloredby_state(avg_trac_mag, hard_states, probs0, probs1, probs2, cell_name, timepoints, save_path):
    n_states = 3
    # timepoints = np.arange(len(hard_states))

    states = ['low force', 'medium force', 'high force']


    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False, gridspec_kw={"height_ratios": [1, 1]})

    # --- Avg trac mag colored by state ---
    lc = colored_line(axes[0], timepoints, avg_trac_mag, hard_states)
    fig.colorbar(lc, ax=axes[0], label="State")
    axes[0].set_ylabel("Average traction magnitude (Pa)")
    axes[0].set_title(f"Average traction magnitude — {cell_name}")
    axes[0].grid(True)

    # --- Stacked soft probabilities ---
    axes[1].plot(
        timepoints,
        probs0,
        label=f"State 0",
    )
    axes[1].plot(
        timepoints,
        probs1,
        label=f"State 1",
    )
    axes[1].plot(
        timepoints,
        probs2,
        label=f"State 2",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("P(State | Observation)")
    axes[1].set_title(f"Soft State Probabilities — {cell_name}")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.12, 1.0))
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('{}/1d_hmm_state_trace_coloredby_state.pdf'.format(save_path),bbox_inches='tight',format='pdf')
    plt.clf()


def plot_state_track(base_path, hard_states, framenum, save_path):
    mask_path = base_path + '/masks'
    mask_files = os.listdir(mask_path)
    mask_files = [s for s in mask_files if s.startswith('._')==False]
    mask_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    x_centers = []
    y_centers = []
    for file in mask_files:
        mask = imread(mask_path + '/' + file)
        
        if len(np.flatnonzero(mask)) > 0:

            # get position of center of cell
            centroid = get_center(mask) 
            x_center = (centroid[1])
            y_center = (centroid[0])

            x_centers.append(x_center)
            y_centers.append(y_center)
        
    data = {'x': x_centers, 'y': y_centers}

    plot_df = pd.DataFrame(data)

    plot_df['state'] = pd.Series(hard_states.to_numpy(), index=framenum)

    plot_df['x_smooth'] = plot_df['x'].rolling(3).mean()[3-1::3]
    plot_df['y_smooth'] = plot_df['y'].rolling(3).mean()[3-1::3]

    plot_df['x_smooth'].iat[0] =  plot_df['x'].iloc[0]
    plot_df['x_smooth'].iat[-1] =  plot_df['x'].iloc[-1]

    plot_df['y_smooth'].iat[0] =  plot_df['y'].iloc[0]
    plot_df['y_smooth'].iat[-1] =  plot_df['y'].iloc[-1]

    plot_df['x_smooth'] = plot_df['x_smooth'].interpolate()
    plot_df['y_smooth'] = plot_df['y_smooth'].interpolate()

    plot_df['state'] = plot_df['state'].ffill().bfill()

    img_path = base_path + '/gfp'
    img_files = os.listdir(img_path)
    img_files = [s for s in img_files if s.startswith('._')==False]
    img_files = [idx for idx in img_files if idx.endswith('.tif')]
    img_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


    colors_for_state = [(242/256,140/256,40/256), (11/256,218/256,81/256), (255/256,0,255/256)] 
    color_time = np.array(colors_for_state)[plot_df['state'].array.astype(int)]
    for t in range(len(img_files)):
        img = imread(img_path + '/' + img_files[t], as_gray=True)
        plt.scatter(plot_df['x_smooth'].iloc[:t],plot_df['y_smooth'].iloc[:t],c=color_time[:t],s=0.5)
        # plt.plot(plot_df['x_smooth'].iloc[:t],plot_df['y_smooth'].iloc[:t],'orange',linewidth=0.5)
        plt.imshow(img, origin='lower')
        # plt.savefig(save_path + '/t_{}.tiff'.format(t+1),bbox_inches='tight',dpi=200) 
        plt.savefig(save_path + '/t_{}.svg'.format(t+1),bbox_inches='tight',format='svg')
        plt.clf()