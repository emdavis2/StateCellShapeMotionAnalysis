import numpy as np
import pandas as pd
import os
import ast
import re

#### Function to load in the force dipole and quadrapole information as well as ellipse fit information per track
#### Note: the expected format of the data for this code is main_folder -> experiment_date -> treatment (ARPC2KO/WT) -> track -> masks/tractions_csv
#### Output is a dictionary where the keys are each individual track and the entries are dictionaries whose keys denote the parameters of interest
def force_ellipse_dict(main_dir):
    dipole_dict = {}
    ellipse_dict = {}
    quad_dict = {}
    exp_list = os.listdir(main_dir)
    for exp in exp_list:
        if not exp.startswith("."): 
            treatment_list = os.listdir(main_dir+'/'+exp)
            for treatment in treatment_list:
                if not treatment.startswith("."):
                    track_list = os.listdir(main_dir+'/'+exp+'/'+treatment)
                    for track in track_list:
                        if not track.startswith("."):
                            basepath = main_dir+'/'+exp+'/'+treatment+'/'+track
                            dipole_dir = basepath + '/dipole_analysis'
                            ellipse_dir = basepath + '/ellipse_analysis'
                            quadrapole_dir = basepath + '/quadrapole_analysis'
                            major_dipoleang_dir = dipole_dir + '/major_dipole_angle'
                            minor_dipoleang_dir = dipole_dir + '/minor_dipole_angle'
                            major_dipoleev_dir = dipole_dir + '/major_dipole_eigenval'
                            minor_dipoleev_dir = dipole_dir + '/minor_dipole_eigenval'
    
                            total_forcemag_dir = dipole_dir + '/total_force_mag'
                            avg_tracmag_dir = dipole_dir + '/avg_trac_mag'
                            
                            major_ellipseang_dir = ellipse_dir + '/major_ellipse_angle'
                            minor_ellipseang_dir = ellipse_dir + '/minor_ellipse_angle'
                            major_ellipselen_dir = ellipse_dir + '/major_ellipse_len'
                            minor_ellipselen_dir = ellipse_dir + '/minor_ellipse_len'
                            
                            date = exp.split('_')
                            date = date[0]+date[1]+date[2]
                            
                            track_name = date + "_" + treatment + "_" + track
                            dipole_dict[track_name] =  {}
                            ellipse_dict[track_name] =  {}
                            quad_dict[track_name] = {}
                            
                            quad_dict[track_name]['MxxTx'] = np.genfromtxt(quadrapole_dir+'/'+'{}_MxxTx.csv'.format(track), delimiter=",")
                            quad_dict[track_name]['MxxTy'] = np.genfromtxt(quadrapole_dir+'/'+'{}_MxxTy.csv'.format(track), delimiter=",")
                            quad_dict[track_name]['MxyTx'] = np.genfromtxt(quadrapole_dir+'/'+'{}_MxyTx.csv'.format(track), delimiter=",")
                            quad_dict[track_name]['MxyTy'] = np.genfromtxt(quadrapole_dir+'/'+'{}_MxyTy.csv'.format(track), delimiter=",")
                            quad_dict[track_name]['MyyTx'] = np.genfromtxt(quadrapole_dir+'/'+'{}_MyyTx.csv'.format(track), delimiter=",")
                            quad_dict[track_name]['MyyTy'] = np.genfromtxt(quadrapole_dir+'/'+'{}_MyyTy.csv'.format(track), delimiter=",")
                            
                            for file in os.listdir(major_dipoleang_dir):
                                dipole_dict[track_name]['major_axis_ang'] = np.genfromtxt(major_dipoleang_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(minor_dipoleang_dir):
                                dipole_dict[track_name]['minor_axis_ang'] = np.genfromtxt(minor_dipoleang_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(major_dipoleev_dir):
                                dipole_dict[track_name]['major_axis_eigenval'] = np.genfromtxt(major_dipoleev_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(minor_dipoleev_dir):
                                dipole_dict[track_name]['minor_axis_eigenval'] = np.genfromtxt(minor_dipoleev_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(total_forcemag_dir):
                                dipole_dict[track_name]['total_force_mag'] = np.genfromtxt(total_forcemag_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(avg_tracmag_dir):
                                dipole_dict[track_name]['avg_trac_mag'] = np.genfromtxt(avg_tracmag_dir+'/'+file, delimiter=",")
    
    
                            for file in os.listdir(major_ellipseang_dir):
                                ellipse_dict[track_name]['major_axis_ang'] = np.genfromtxt(major_ellipseang_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(minor_ellipseang_dir):
                                ellipse_dict[track_name]['minor_axis_ang'] = np.genfromtxt(minor_ellipseang_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(major_ellipselen_dir):
                                ellipse_dict[track_name]['major_axis_len'] = np.genfromtxt(major_ellipselen_dir+'/'+file, delimiter=",")
    
                            for file in os.listdir(minor_ellipselen_dir):
                                ellipse_dict[track_name]['minor_axis_len'] = np.genfromtxt(minor_ellipselen_dir+'/'+file, delimiter=",")

    return dipole_dict, ellipse_dict, quad_dict


##############################################################################################################################################

#### Input path to tracksgeo pkl file
#### Returns a dicitonary where the keys are individual tracks and the entries are shape and motion information
def load_tracksgeo(tracksgeo_path, dipole_dict):
    open_tracksgeo = pd.read_pickle(tracksgeo_path)
    
    tracksgeo_dict = {}
    
    refined_keylist = ['soft_gel_WT', 'soft_gel_ARPC2KO']
    
    for key in refined_keylist:
        treatment = key.split('_')[2]
        for df in open_tracksgeo[key]:
            movie = 'movie'+str(int(df['movie'][0]))+'_' 
            track = 'track'+str(int(df['track_id'][0]))
            date = str(df['experiment'][0]).split('_')[0]
            full_name = date + '_' + treatment + '_' + movie + track
            if full_name in list(dipole_dict.keys()):
                tracksgeo_dict[full_name] = df
    return tracksgeo_dict




##############################################################################################################################################


