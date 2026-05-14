import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.measure import regionprops, regionprops_table, find_contours
import os
from scipy import ndimage
from numpy import linalg as LA
import math

#### Note: the expected format of the data for this code is main_folder -> experiment_date -> treatment (ARPC2KO/WT) -> track -> masks/tractions_csv
main_dir = '/proj/telston_lab/projects/data/DataForTFM/40x_TFM_Data_12042025'


def not_empty(arr):
    return any(any(row) for row in arr)


exp_list = os.listdir(main_dir)
for exp in exp_list:
    if not exp.startswith(".") and not exp.startswith('2024_09_06') and not exp.startswith('2024_09_27') and not exp.startswith('2025_01_30'): ###CHANGE THIS
        treatment_list = os.listdir(main_dir+'/'+exp)
        for treatment in treatment_list:
            if not treatment.startswith("."):
                track_list = os.listdir(main_dir+'/'+exp+'/'+treatment)
                for track in track_list:
                    if not track.startswith("."):
                        basepath = main_dir+'/'+exp+'/'+treatment+'/'+track
                        track_maskfold = basepath + '/masks' #path to mask directory for track of interest in fov ref frame

                        #corresponding mask for each time
                        mask_files = os.listdir(track_maskfold)
                        mask_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

                        major_ellipse_ang = []
                        minor_ellipse_ang = []
                        major_ellipse_len = []
                        minor_ellipse_len = []

                        for count in range(len(mask_files)):
                            # get mask for this time point
                            t_mask = imread(track_maskfold + '/' + mask_files[count])
                            
                            if not_empty(t_mask):

                                props = regionprops(t_mask)

                                # get coordinates of cell center
                                y0, x0 = props[0].centroid

                                # get angle from y axis to major axis of ellipse that best fits cell
                                orientation = props[0].orientation

                                # transform so angle is relative to x-axis
                                major_axis_ang = np.sign(orientation) * ((np.pi/2) - np.abs(orientation))

                                # orientation is also angle from x-axis to minor axis
                                minor_axis_ang = orientation

                                major_ellipse_ang.append(major_axis_ang)
                                minor_ellipse_ang.append(minor_axis_ang)

                                minor_ellipse_len.append(props[0].axis_minor_length)
                                major_ellipse_len.append(props[0].axis_major_length)

                        
                        ellipse_savepath_base = basepath + '/ellipse_analysis'

                        if not os.path.exists(ellipse_savepath_base):
                            os.mkdir(ellipse_savepath_base)
                            
                        ellipse_savepath_majorangle = ellipse_savepath_base + '/major_ellipse_angle'

                        if not os.path.exists(ellipse_savepath_majorangle):
                            os.mkdir(ellipse_savepath_majorangle)
                            
                        ellipse_savepath_minorangle = ellipse_savepath_base + '/minor_ellipse_angle'

                        if not os.path.exists(ellipse_savepath_minorangle):
                            os.mkdir(ellipse_savepath_minorangle)
                            
                        ellipse_savepath_majorlen = ellipse_savepath_base + '/major_ellipse_len'

                        if not os.path.exists(ellipse_savepath_majorlen):
                            os.mkdir(ellipse_savepath_majorlen)
                            
                        ellipse_savepath_minorlen = ellipse_savepath_base + '/minor_ellipse_len'

                        if not os.path.exists(ellipse_savepath_minorlen):
                            os.mkdir(ellipse_savepath_minorlen)
                            
                        np.savetxt(ellipse_savepath_majorangle+'/{}_majorellipseangle.csv'.format(track), major_ellipse_ang, delimiter =", ", fmt ='% s')
                        np.savetxt(ellipse_savepath_minorangle+'/{}_minorellipseangle.csv'.format(track), minor_ellipse_ang, delimiter =", ", fmt ='% s')
                        np.savetxt(ellipse_savepath_majorlen+'/{}_majorellipselen.csv'.format(track), major_ellipse_len, delimiter =", ", fmt ='% s')
                        np.savetxt(ellipse_savepath_minorlen+'/{}_minorellipselen.csv'.format(track), minor_ellipse_len, delimiter =", ", fmt ='% s')
