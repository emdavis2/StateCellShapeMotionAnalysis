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

pixel_size = .108 #um
pixel_area = pixel_size**2

dir1 = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/masks'
dir2 = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/crop/masks'

save_path = '/Volumes/EMD_Drive'

shape_df = {}

cell_masks = [f"{dir1+'/'}{file}" for file in os.listdir(dir1)] + [f"{dir2+'/'}{file}" for file in os.listdir(dir2)]


for file in cell_masks:
    mask = imread(file)


    region = regionprops(mask)[0]
    area = region.area * pixel_area
    solidity = region.solidity
    eccentricity = region.eccentricity


    df = pd.DataFrame({'area': [np.average(area)], 'solidity': [np.median(solidity)], 'eccentricity': [eccentricity]})

    shape_df[(file.split('/')[-1])[:-4]] = df

with open(save_path+'/shape_df.pkl', 'wb') as handle:
  pickle.dump(shape_df, handle, protocol=2)