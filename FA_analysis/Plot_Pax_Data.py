import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
import pickle

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

##################################################################

save_path = './figures/corr_plots_Pax'

#check to see if the path exists, if not make the directory
if not os.path.exists(save_path):
  os.mkdir(save_path)

# 'fa_area_avg','fa_count_total' refer to FAs only in protrusion region
A = ['fa_avg_area','fa_med_area','num_fa','avg_trac_mag','traction_mag','faai', 'dipole_ratio', 'base_fa_count', 'base_fa_avg', 'mid_fa_count', 'mid_fa_avg', 'tip_fa_count', 'tip_fa_avg', 'base_force_avg', 'mid_force_avg',
       'tip_force_avg', 'fa_area_avg','fa_count_total', 'protr_area','area', 'solidity', 'eccentricity', 'fa_density', 'protr_fa_density','total_force', 'total_force_avg', 'fa_mean_dist', 'fa_med_dist', "body_force_avg","body_fa_avg","body_fa_count",'body_fa_density']
B = ['fa_avg_area','fa_med_area','num_fa','avg_trac_mag','traction_mag','faai', 'dipole_ratio', 'base_fa_count', 'base_fa_avg', 'mid_fa_count', 'mid_fa_avg', 'tip_fa_count', 'tip_fa_avg', 'base_force_avg', 'mid_force_avg',
       'tip_force_avg', 'fa_area_avg','fa_count_total', 'protr_area', 'area', 'solidity', 'eccentricity', 'fa_density', 'protr_fa_density', 'total_force', 'total_force_avg', 'fa_mean_dist', 'fa_med_dist', "body_force_avg","body_fa_avg","body_fa_count",'body_fa_density']
unique_pairs = list({tuple(sorted((a, b))) for a in A for b in B})

for pair in unique_pairs:
    column1 = pair[0]
    column2 = pair[1]
    x = final_df[column1]
    y = final_df[column2]
    # Create a combined mask where BOTH elements are valid numbers
    mask = np.isfinite(x) & np.isfinite(y)

    r, pval = pearsonr(x[mask],y[mask])
    # rho, p_value = stats.spearmanr(final_df[column1].fillna(0),final_df[column2].fillna(0))
    plt.plot(x[mask],y[mask],'.',color='maroon')
    plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
    plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))

    sns.regplot(data=final_df, x=column1, y=column2, scatter=False, ci=None, color="maroon")

    plt.xlabel('{}'.format(column1))
    plt.ylabel('{}'.format(column2))

    plt.legend()

    plt.savefig('{}/{}_{}_corrplot.pdf'.format(save_path, column1, column2),bbox_inches='tight',format='pdf')
    plt.clf()




x = 1/final_df['dipole_ratio']
y = final_df['eccentricity']
# Create a combined mask where BOTH elements are valid numbers
mask = np.isfinite(x) & np.isfinite(y)

r, pval = pearsonr(x[mask],y[mask])
# rho, p_value = stats.spearmanr(final_df[column1].fillna(0),final_df[column2].fillna(0))
plt.plot(x[mask],y[mask],'.',color='maroon')
plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))

sns.regplot(data=final_df, x=column1, y=column2, scatter=False, ci=None, color="maroon")

plt.xlabel('{}'.format('dipole ratio (major/minor)'))
plt.ylabel('{}'.format('eccentricity'))

plt.legend()

plt.savefig('{}/1overdipoleratio_ecc_corrplot.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()

x = 1/final_df['dipole_ratio']
y = final_df['faai']
# Create a combined mask where BOTH elements are valid numbers
mask = np.isfinite(x) & np.isfinite(y)

r, pval = pearsonr(x[mask],y[mask])
# rho, p_value = stats.spearmanr(final_df[column1].fillna(0),final_df[column2].fillna(0))
plt.plot(x[mask],y[mask],'.',color='maroon')
plt.plot([],[], ' ', label='r {}'.format(np.round(r,2)))
plt.plot([],[], ' ', label='p value {}'.format(np.round(pval,5)))

sns.regplot(data=final_df, x=column1, y=column2, scatter=False, ci=None, color="maroon")

plt.xlabel('{}'.format('dipole ratio (major/minor)'))
plt.ylabel('{}'.format('faai'))

plt.legend()

plt.savefig('{}/1overdipoleratio_faai_corrplot.pdf'.format(save_path),bbox_inches='tight',format='pdf')
plt.clf()