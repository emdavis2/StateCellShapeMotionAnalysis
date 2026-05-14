import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.io import imread, imsave, imshow
from skimage.util import img_as_uint
from skimage import data, filters, measure, morphology, util
from skimage import color
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.measure import  regionprops, regionprops_table
from pathlib import Path
from enum import Enum
import tifffile
import pandas as pd
import cv2
import ntpath
import os
import time
import pickle
import re
import copy
import scipy
import builtins
import shutil
import contextlib
import zipfile
from itertools import starmap
from functools import partial
import random
import csv
from scipy import ndimage
from scipy.stats import skew
import numpy as np
from ipywidgets import interact, widgets, Layout
from IPython.utils.io import capture_output
import matplotlib.pyplot as plt
from typing import Dict, DefaultDict, Tuple, List, Union

import itertools
import math

#Experiment Name (will be incorporated into output file and folder names)
experiment = "20240927_Jr20ARPC2KO" ###MODIFY WITH NAME THAT REFLECTS DATA BEING ANALYZED

#Analysis folder: all output analysis data will be output to {analysis output folder}/{experiment name}
analysis_output_folder = "/proj/telston_lab/projects/data/SegmentationAnalysis/"+experiment ###Shouldn't need to modify unless saving output to a different location than telston_lab space on Longleaf

if not os.path.exists(analysis_output_folder):
  os.mkdir(analysis_output_folder)

folderimages = '/proj/telston_lab/projects/data/rf_2024_09_27_Jr20Overnight/ARPC2KO/dc_gfp' ###MODIFY WITH PATH TO UNSTACKED TIFF IMAGES TO TRACK
foldermasks = '/proj/telston_lab/projects/data/rf_2024_09_27_Jr20Overnight/ARPC2KO/dc_masks' ###MODIFY WITH PATH TO CELL MEMBRANE MASKS

# if reference images are present for traction force data, specify the path to those images
bead_ref_dir = '/proj/telston_lab/projects/data/rf_2024_09_27_Jr20Overnight/ARPC2KO/bead_ref'

# path to where bead images are located 
bead_dir = '/proj/telston_lab/projects/data/rf_2024_09_27_Jr20Overnight/ARPC2KO/dc_beads'

#################################################################################################################################

### The names/locations of various parameter and analysis output files within the segmentation analysis folder
### All input paths are relative to the analysis output folder defined in the previous cell

movies_folder = analysis_output_folder + "/OutputMovies" #@param

# Analysis Parameters
parameters_folder = analysis_output_folder+"/AnalysisParameters" #@param
# if not os.path.exists(parameters_folder):
#   os.mkdir(parameters_folder)
cell_reading_params_path = parameters_folder+"/reading_cells_parameters.pkl" #@param 
track_params_path = parameters_folder+"/tracking_parameters.pkl" #@param
qc_params_path = parameters_folder+"/track_qc_output.pkl" #@param
#make sure no dependency chains
del parameters_folder

# Analysis Outputs
labeled_cellmasks_path = analysis_output_folder+"/labeledmasks.zip" #@param
#labeled_nucmasks_path = analysis_output_folder+"/labelednucs.zip" #@param
cell_features_path = analysis_output_folder+"/cell_features.csv" #@param
raw_tracks_path = analysis_output_folder+"/tracks.pkl" #@param
qc_tracks_path = analysis_output_folder+"/qc_tracks.pkl" #@param

#################################################################################################################################

folder_cellmasks_labeled = analysis_output_folder+'/'+Path(labeled_cellmasks_path).stem

imagenames=[f for f in os.listdir(foldermasks) if f.endswith('.tif')]
#Get list of movies
movies = [int(re.findall(r"s(\d+).", imagenames[i])[0]) for i in range(len(imagenames)) ]
#get set of unique elements
movies = list(set(movies))
#sort
movies.sort()
#Get frame numbers and show the largest number
frames = {m:[] for m in movies};
for name in imagenames:
  match = re.findall(r"s(\d+)_t(\d+).", name)[0];
  frames[int(match[0])].append(int(match[1]));
for m in movies:
  frames[m].sort();

#################################################################################################################################

#function to read a mask corresponding to a given movie (implicit parameter), and frame (explicit parameter iframe)
#get_mask = lambda iframe: imread( masks_folder/basename+'_s'+str(int(movie))+'_t'+str(int(frame[iframe]))+'.TIF')
def get_mask(iframe):
  filename = basename+'_s'+str(int(movie))+'_t'+str(int(frame[iframe]))+'.TIF'
  return imread(folder_cellmasks_labeled+'/'+filename)

#function that returns the mask of a cell centered. It takes the cell label in the mask, and the movie
#as implicit parameters determined by the track information (as used later). 
def get_centered_cell(iframe):
  mask=get_mask(iframe)
  #erase all objects with different label as cell
  mask[mask != label[iframe] ] =0
  #set cell positions as 1
  mask[mask > 0 ] = 1
  props = measure.regionprops(mask)
  #center image
  centroid=props[0].centroid
  N=mask.shape
  centered=mask[np.ix_((np.arange(N[0]) + int(centroid[0]) - int(N[0]/2)) % N[0] , (np.arange(N[1]) + int(centroid[1]) - int(N[1]/2)) % N[1])]
  return centered

#function that returns the mask of a cell. It takes the cell label in the mask, and the movie
#as implicit parameters determined by the track information (as used later).
def get_cell(iframe):
  mask=get_mask(iframe)
  mask_all = get_mask(iframe) 
  #erase all objects with different label as cell
  mask[mask != label[iframe] ] =0
  #set cell positions as 1
  mask[mask > 0 ] = 1

  return mask, mask_all

rotation_matrix = lambda angle: np.asarray([[np.cos(angle) , -np.sin(angle)],[np.sin(angle), np.cos(angle)]] )


def skew_from_hist(hist):
  value = np.asarray(range(len(hist)))
  mean = np.sum(hist*value)/np.sum(hist)
  m2 = np.sum(hist*(value-mean)**2)/np.sum(hist)
  m3 = np.sum(hist*(value-mean)**3)/np.sum(hist)
  return  m3/m2**(3/2) 

#get corresonding bead image for track
def get_img(iframe):
    filename = basename_beads+'_s'+str(int(movie))+'_t'+"{:01d}".format(int(frame[iframe]))+'.tif'
    img = imread(bead_dir+'/'+filename)#[:,:,0]
    return img # change so image is same size as mask if needed

#get corresonding gfp image for track
def get_gfpimg(iframe):
    filename = basename_beads+'_s'+str(int(movie))+'_t'+"{:01d}".format(int(frame[iframe]))+'.tif'
    img = imread(folderimages+'/s'+str(int(movie))+'/'+filename)#[:,:,0]
    return img # change so image is same size as mask if needed

#get corresponding reference bead image for track
def get_refimg(movie):
    # ## if no reference frames are specified in the dataset,then use the last frame as the reference frame
    # last_frame = frames[int(movie)][-1]
    # filename = basename+'_s'+str(int(movie))+'_t'+str(int(last_frame))+'.tif'
    # img = imread(folderimages+'/beads/'+filename)[:,:,0]
    # return resize(img,(img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True, preserve_range=True).astype('uint16') # change so image is same size as mask if needed
    
    ## if reference frames are specified in the dataset, then use that
    filename = basename_refbeads + '_s{}.tif'.format(movie)
    img = imread(bead_ref_dir+'/'+filename)
    return img # change so image is same size as mask if needed


#determine what the minimum and maximum values to crop the image to so that it encompasses the entire track
def get_bounds(bounds):
    first_frame = 0
    
    mask, mask_all = get_cell(first_frame)

    min_y = 0
    min_x = 0
    max_y = np.shape(mask)[0]
    max_x = np.shape(mask)[1]

    low_x_b = min(center_x) - bounds
    high_x_b = max(center_x) + bounds
    low_y_b = min(center_y) - bounds
    high_y_b = max(center_y) + bounds

    lb_x = max([min_x, low_x_b])
    ub_x = min([max_x, high_x_b])

    lb_y = max([min_y, low_y_b])
    ub_y = min([max_y, high_y_b])
    
    return lb_x, ub_x, lb_y, ub_y

#crop images so they are cropped around the cell's entire track
def crop_img(iframe):
    
    img = get_img(iframe) #get bead image
    gfp = get_gfpimg(iframe)
    mask, mask_all = get_cell(iframe) #get mask for track
    ref_img = get_refimg(movie) #get last bead frame of movie (not track) for reference
    
    cropgfp = gfp[lb_y:ub_y, lb_x:ub_x] 
    
    cropimg = img[lb_y:ub_y, lb_x:ub_x] 

    cropmask = mask[lb_y:ub_y, lb_x:ub_x] 

    cropmask_all = mask_all[lb_y:ub_y, lb_x:ub_x] 

    ref_cropimg = ref_img[lb_y:ub_y, lb_x:ub_x] 
    
    return np.asarray(cropgfp), np.asarray(cropimg), np.asarray(cropmask), np.asarray(ref_cropimg), np.asarray(cropmask_all)


masks_folder = foldermasks

#read pickled tracks
#unpickle tracks  
with open(qc_tracks_path, 'rb') as handle:
  tracks_noshape = pickle.load(handle, encoding='latin1') 


# Creates directory in analysis_output_folder to save FOV arouund cell for each track as individual tiff images
# create folder to store individual movies from tracks
if not os.path.exists(analysis_output_folder+'/beads_movies'):
  os.mkdir(analysis_output_folder+'/beads_movies');


###Loop to add features to each timepoint of all the tracks
#Add morphological and morphodynamical features to each timepoint (frame) in the tracks

tracks = copy.copy(tracks_noshape)

for movie in tracks:
  for itrack in tracks[movie]:
    tracklength = len(tracks[movie][itrack])
    frame = list(tracks[movie][itrack]['frame'])
    label = list(tracks[movie][itrack]['label'])
    center_x = list(tracks[movie][itrack]['approximate-medoidx'])
    center_y = list(tracks[movie][itrack]['approximate-medoidy'])
    masks=os.listdir(masks_folder)
    #implicit argument to read cells
    basename=re.findall(r"(.+)_s",masks[0])[0] 
    bead_imgs=os.listdir(bead_dir)
    basename_beads=re.findall(r"(.+)_s",bead_imgs[0])[0]
    refbead_imgs=os.listdir(bead_ref_dir)
    basename_refbeads=re.findall(r"(.+)_s",refbead_imgs[0])[0]

    
    # Create directory in beads_movies directory
    # if not os.path.exists(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)):
    #     os.mkdir(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack))
    # # Create subdirectory within just created directory to save reference frames of region of interest     
    # if not os.path.exists(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/reference'):
    #     os.mkdir(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/reference')
    # # Create subdirectory within just created directory to save cropped masks of cell in track     
    # if not os.path.exists(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/masks'):
    #     os.mkdir(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/masks')
        
    # if not os.path.exists(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/gfp'):
    #     os.mkdir(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/gfp')

    # Create subdirectory within just created directory to save cropped masks of all cells in track     
    if not os.path.exists(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/masks_all'):
        os.mkdir(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/masks_all')  
    
    # find upper and lower bounds in both x and y when cropping images
    bounds = 300
    lb_x, ub_x, lb_y, ub_y = get_bounds(bounds)
    
    #GET SCIKIT-IMAGE CELL METRICS AND SAVE CELL MASK
    for iframe in range(tracklength):
    
      # Crop the images of beads to area around cell (cell_mov), cropped mask (cell_mask), and corresponding region in reference frame and/or last frame (ref_mov)
      cell_gfp, cell_mov, cell_mask, ref_mov, cell_mask_all = crop_img(iframe)
    
      frame_str = frame[iframe]
        
      #rescale images and convert from 32 bit to 16 bit
      cell_mov = cell_mov - cell_mov.min()
      cell_mov = cell_mov / cell_mov.max()
      cell_mov = img_as_uint(cell_mov)
        
      ref_mov = ref_mov - ref_mov.min()
      ref_mov = ref_mov / ref_mov.max()
      ref_mov = img_as_uint(ref_mov)
        
      # Save the bead images, masks, and reference bead images in the beads_movie appropriate sub-directories
    #   imsave(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/t'+str(frame_str)+'.tif',cell_mov,check_contrast=False)
    #   imsave(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/masks/t'+str(frame_str)+'.tif',cell_mask,check_contrast=False)
    #   imsave(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/gfp/t'+str(frame_str)+'.tif',cell_gfp,check_contrast=False)
    # imsave(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/reference/ref_mov'+str(movie)+'.tif',ref_mov,check_contrast=False)
      imsave(analysis_output_folder+'/beads_movies/movie'+str(movie)+'_track'+str(itrack)+'/masks_all/t'+str(frame_str)+'.tif',cell_mask_all,check_contrast=False)