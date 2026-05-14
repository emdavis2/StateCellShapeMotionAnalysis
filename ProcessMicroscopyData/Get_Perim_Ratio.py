import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import opening, disk, binary_dilation, convex_hull_image
from skimage.measure import perimeter_crofton
from skimage.io import imread, imsave, imshow
from skimage.measure import label, regionprops
import os
import pandas as pd
import pickle
from skimage.filters import gaussian
from skimage.morphology import closing, disk

main_dir = '/proj/telston_lab/projects/data/DataForTFM/TFM_Data_02242025'

# #make directory to save skeletons
# fig_savepath = main_dir + '/skeleton_figs'
# if not os.path.exists(fig_savepath):
#     os.mkdir(fig_savepath)

all_results = []

exp_list = os.listdir(main_dir)
for exp in exp_list:
    if not exp.startswith(".") and '2025_01_30_Jr20Overnight' not in exp and 'Duo_Seg' not in exp and 'skeleton_figs_mymethodskeleton' not in exp:
        treatment_list = os.listdir(main_dir+'/'+exp)
        for treatment in treatment_list:
            if not treatment.startswith("."):
                track_list = os.listdir(main_dir+'/'+exp+'/'+treatment)
                for track in track_list:
                    if not track.startswith("."):
                        basepath = main_dir+'/'+exp+'/'+treatment+'/'+track
                        mask_dir = basepath + '/masks'

                        
                        date = exp.split('_')
                        date = date[0]+date[1]+date[2]
                        
                        track_name = date + "_" + treatment + "_" + track


                        t_list = os.listdir(mask_dir)
                        for t_file in t_list:
                            if not t_file.startswith(".") and t_file.endswith('.tif'):
                                mask = imread(mask_dir+'/'+t_file).astype(bool)

                                blurred = gaussian(mask.astype(float), sigma=2)
                                smoothed = blurred > 0.5
                                smoothed = closing(smoothed, disk(10))

                                mask = smoothed
                                
                                convex_hull = convex_hull_image(mask)

                                perimeter_convexhull = perimeter_crofton(convex_hull)
                                perimeter_cell = perimeter_crofton(mask)

                                ratio = perimeter_convexhull/perimeter_cell

                                results = {"name": track_name+'_'+t_file, "perim_ratio": ratio, 'perim': perimeter_cell}
                                


                                all_results.append(results)
                                
perimratio_df = pd.DataFrame(all_results)

savepath = '/proj/telston_lab/projects/data/DataForTFM/'

perimratio_df.to_pickle(savepath+'perim_ratio_crof.pkl')
perimratio_df.to_csv(savepath+'perim_ratio_crof.csv', index=False)