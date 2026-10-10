import numpy as np
import pandas as pd
import os
from skimage.measure import label, regionprops
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import pickle
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops

pixel_size = .108 #um
pixel_area = pixel_size**2

save_path = '/Volumes/EMD_Drive'

spatial_fa_df = {}

dir1 = '/Volumes/EMD_Drive/faas_results_minA20'
dir2 = '/Volumes/EMD_Drive/faas_results_minA20crop'

mask_dir1 = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/masks'
mask_dir2 = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/crop/masks'

cell_folders = [f"{dir1+'/'}{file}" for file in os.listdir(dir1)] + [f"{dir2+'/'}{file}" for file in os.listdir(dir2)]

for fold in cell_folders:
    name = fold.split('/')[-1]  
    exp_fold = os.listdir(fold)[0]
    fa_mask_path = fold + '/' + exp_fold + '/individual_pictures/00001/adhesions_binary.png'
    if not os.path.isfile(fa_mask_path):
        print(f"{fold}: no fa mask")
        df = pd.DataFrame({'fa_mean_dist': [np.nan], 'fa_med_dist': [np.nan]})
        spatial_fa_df[fold.split('/')[-1]] = df
        continue

    if 'crop' in fold:
        mask_path = mask_dir2 + '/' + name + '.tif'
    else:
        mask_path = mask_dir1 + '/' + name + '.tif'

    fa_mask = imread(fa_mask_path)
    cell_mask = imread(mask_path)

    # Distance map (inside cell)
    distance_map = distance_transform_edt(cell_mask)

    # Normalize distance map
    distance_map_norm = distance_map / np.max(distance_map)

    # Label adhesions
    labeled_fa = label(fa_mask)

    distances = []

    for prop in regionprops(labeled_fa):
        y, x = prop.centroid
        y, x = int(y), int(x)
        
        distances.append(distance_map_norm[y, x])

    # Metrics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)

    df = pd.DataFrame({'fa_mean_dist': [mean_dist * pixel_size], 'fa_med_dist': [median_dist * pixel_size]})
    
    spatial_fa_df[fold.split('/')[-1]] = df

with open(save_path+'/spatial_fa_df.pkl', 'wb') as handle:
  pickle.dump(spatial_fa_df, handle, protocol=2)