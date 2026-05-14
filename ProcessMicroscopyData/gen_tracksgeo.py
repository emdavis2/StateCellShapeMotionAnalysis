import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.io import imread, imsave, imshow
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
  #erase all objects with different label as cell
  mask[mask != label[iframe] ] =0
  #set cell positions as 1
  mask[mask > 0 ] = 1

  return mask

rotation_matrix = lambda angle: np.asarray([[np.cos(angle) , -np.sin(angle)],[np.sin(angle), np.cos(angle)]] )


def skew_from_hist(hist):
  value = np.asarray(range(len(hist)))
  mean = np.sum(hist*value)/np.sum(hist)
  m2 = np.sum(hist*(value-mean)**2)/np.sum(hist)
  m3 = np.sum(hist*(value-mean)**3)/np.sum(hist)
  return  m3/m2**(3/2) 

#get corresonding bead image for track
def get_img(iframe):
    filename = basename_beads+'_s'+str(int(movie))+'_t'+"{:03d}".format(int(frame[iframe]))+'.tif'
    img = imread(bead_dir+'/'+filename)#[:,:,0]
    return img # change so image is same size as mask if needed

#get corresponding reference bead image for track
def get_refimg(movie):
    # ## if no reference frames are specified in the dataset,then use the last frame as the reference frame
    # last_frame = frames[int(movie)][-1]
    # filename = basename+'_s'+str(int(movie))+'_t'+str(int(last_frame))+'.tif'
    # img = imread(folderimages+'/beads/'+filename)[:,:,0]
    # return resize(img,(img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True, preserve_range=True).astype('uint16') # change so image is same size as mask if needed
    
    ## if reference frames are specified in the dataset, then use that
    filename = basename_refbeads + '_s{}_ref.tif'.format(movie)
    img = imread(bead_ref_dir+'/'+filename)
    return img # change so image is same size as mask if needed

#crop images so they are centered on cell of track
def crop_img(iframe):
    
    img = get_img(iframe) #get bead image
    mask = get_cell(iframe) #get mask for track
    ref_img = get_refimg(movie) #get last bead frame of movie (not track) for reference
    
    cx = center_x[iframe] # x coordinate of cell center
    cy = center_y[iframe] # y coordinate of cell center
    
    bounds = 200 #change 200 to something else if you want a bigger or smaller field of view in crop   
    
    min_y = 0
    min_x = 0
    max_y = np.shape(mask)[0]
    max_x = np.shape(mask)[1]

    low_y_b = cy-bounds
    high_y_b = cy+bounds
    low_x_b = cx-bounds
    high_x_b = cx+bounds

    lb_x = max([min_x, low_x_b])
    ub_x = min([max_x, high_x_b])

    lb_y = max([min_y, low_y_b])
    ub_y = min([max_y, high_y_b])
                                                      
    cropimg = img[lb_y:ub_y, lb_x:ub_x] 

    cropmask = mask[lb_y:ub_y, lb_x:ub_x] 

    ref_cropimg = ref_img[lb_y:ub_y, lb_x:ub_x] 
    
    return np.asarray(cropimg), np.asarray(cropmask), np.asarray(ref_cropimg)


#Experiment Name (will be incorporated into output file and folder names)
experiment = "20240918_Jr20WT" ###MODIFY WITH NAME THAT REFLECTS DATA BEING ANALYZED

#Analysis folder: all output analysis data will be output to {analysis output folder}/{experiment name}
analysis_output_folder = "/proj/telston_lab/projects/data/SegmentationAnalysis/"+experiment ###Shouldn't need to modify unless saving output to a different location than telston_lab space on Longleaf

if not os.path.exists(analysis_output_folder):
  os.mkdir(analysis_output_folder)

folderimages = '/proj/telston_lab/projects/data/rf_2024_09_18_Jr20Overnight10x/WT/cp_gfp' ###MODIFY WITH PATH TO UNSTACKED TIFF IMAGES TO TRACK
foldermasks = '/proj/telston_lab/projects/data/rf_2024_09_18_Jr20Overnight10x/WT/masks' ###MODIFY WITH PATH TO CELL MEMBRANE MASKS

#################################################################################################################################

### The names/locations of various parameter and analysis output files within the segmentation analysis folder
### All input paths are relative to the analysis output folder defined in the previous cell

movies_folder = analysis_output_folder + "/OutputMovies" #@param

# Analysis Parameters
parameters_folder = analysis_output_folder+"/AnalysisParameters" #@param
if not os.path.exists(parameters_folder):
  os.mkdir(parameters_folder)
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

#label of substrate or region on gel where experiment was taken

region = 'soft_gel' #MODIFY THIS VARIABLE WITH DESCRIPTION OF SUBSTRATE OF DATA (CURRENTLY 'glass', 'stiff_gel', OR 'soft_gel')

#################################################################################################################################

masks_folder = foldermasks

#read pickled tracks
#unpickle tracks  
with open(qc_tracks_path, 'rb') as handle:
  tracks_noshape = pickle.load(handle, encoding='latin1') 


###Loop to add features to each timepoint of all the tracks
#Add morphological and morphodynamical features to each timepoint (frame) in the tracks

tracks = copy.copy(tracks_noshape)

for movie in tracks:
  for itrack in tracks[movie]:
    tracklength = len(tracks[movie][itrack])
    #store experiment and id_track, to retrieve track_masks from track
    tracks[movie][itrack]["experiment"]=experiment
    tracks[movie][itrack]["region"]=region
    tracks[movie][itrack]["track_id"]=itrack
    #read data to retrieve corresponding labeled mask cell
    #movie = tracks[itrack]['movie'].iloc[0]
    frame = list(tracks[movie][itrack]['frame'])
    label = list(tracks[movie][itrack]['label'])
    masks=os.listdir(masks_folder)
    #implicit argument to read cells
    basename=re.findall(r"(.+)_s",masks[0])[0] 

    #get polarization angle, skew, protrusion and retraction angle,norm area, radii 

    median_centroidy = []
    median_centroidx = []
    protrusion_angles=[]
    mean_protrusion_angles=[]
    protrusion_norm_radii =[]
    protrusion_norm_areas=[]
    retraction_angles=[]
    mean_retraction_angles=[]
    retraction_norm_areas=[]
    retraction_norm_radii=[]
    mean_retraction_norm_radii=[]
    mean_protrusion_norm_radii =[]

    cell_angles = []
    cell_skews = []

    #select scikit-image shape metrics
    shape_metrics = ['area','convex_area','eccentricity','orientation','perimeter','equivalent_diameter','solidity','extent','major_axis_length','minor_axis_length','centroid']

    track_shape_metrics = pd.DataFrame()

    #GET SCIKIT-IMAGE CELL METRICS AND SAVE CELL MASK
    for iframe in range(tracklength):
      cell = get_cell(iframe) 
      if np.any(cell) == False:
        cell_shape_metrics = measure.regionprops_table(cell, properties = shape_metrics)        
        cell_shape_metrics=pd.DataFrame(cell_shape_metrics)
        row = np.empty(len(shape_metrics)+1)
        row[:] = np.nan
        #cell_shape_metrics = cell_shape_metrics.append(pd.Series(row, index=cell_shape_metrics.columns),ignore_index=True)
        cell_shape_metrics = pd.concat([cell_shape_metrics, pd.Series(row, index=cell_shape_metrics.columns)], ignore_index=True)
        #append cell shape metrics to track shape metrics
        #track_shape_metrics = track_shape_metrics.append(cell_shape_metrics, ignore_index=True)
        track_shape_metrics = pd.concat([track_shape_metrics, cell_shape_metrics], ignore_index=True)
        #get median calculated centroid
        celly = np.array(np.nan)
        cellx = np.array(np.nan)
        median_centroidy.append( np.median(celly) ) 
        median_centroidx.append( np.median(cellx) )
      else:
        cell_shape_metrics = measure.regionprops_table(cell, properties = shape_metrics)        
        cell_shape_metrics=pd.DataFrame(cell_shape_metrics)
        #append cell shape metrics to track shape metrics
        #track_shape_metrics = track_shape_metrics.append(cell_shape_metrics, ignore_index=True)
        track_shape_metrics = pd.concat([track_shape_metrics, cell_shape_metrics], ignore_index=True)
        #get median calculated centroid
        celly,cellx = np.where(cell)
        median_centroidy.append( np.median(celly) ) 
        median_centroidx.append( np.median(cellx) )



    for iframe in range(tracklength-1):

      #GET CELL POLARIZATION ANGLE 
      cell = get_cell(iframe)
      if np.any(cell) == False:
        cell_angles.append( np.nan )
        cell_skews.append(np.nan)
      else:
        cell_centered = get_centered_cell(iframe)
        props = regionprops(cell_centered)
        #major axis angle with respect to y (0 axis : rows) counter-clockwise -pi/2 , pi/2  
        angle_y = props[0].orientation
        #rotate cell so that major axis is aligned with the y axis
        cell_y = rotate(cell_centered,-angle_y*180/np.pi,order=0, preserve_range=True)
        #project cell on the x (1) axis
        proj_x = np.sum(cell_y,1)      
        #get cell polarization vector in a regular coordinate system
        #skew sign corresponds to the tail of the distribution, cell polarization is  
        #defined here as -skew. Because image y-axis is inverted, in a regular coordinate
        #system, cell polarization vector is defined as -(-skew) = skew:
        skew_cell = skew_from_hist(proj_x)
        celly_polarization = [0 , skew_cell ]      
        #rotate back
        polarization = np.dot( rotation_matrix(angle_y) , celly_polarization )
        cell_angles.append( np.arctan2(polarization[1],polarization[0]) )
        cell_skews.append(abs(skew_cell)) 


      #PROTRUSION AND RETRACTION VECTORS : angle, norm_areas, radii (calculated with medians)
      if np.any(cell) == False or np.any(get_cell(iframe+1)) == False:
        protrusion_angles.append( np.nan )
        protrusion_norm_areas.append(np.nan)
        #effective radius (area/pi)^0.5
        protrusion_norm_radii.append( np.nan )
        retraction_angles.append(np.nan)
        retraction_norm_areas.append( np.nan )
        retraction_norm_radii.append( np.nan )
        mean_protrusion_angles.append(np.nan)
        mean_retraction_angles.append(np.nan )
        mean_protrusion_norm_radii.append( np.nan )
        mean_retraction_norm_radii.append( np.nan )

        med_centroids = pd.DataFrame({'median_centroidx':[np.nan], 'median_centroidy':[np.nan]})

      else:
        difference  = get_centered_cell(iframe+1) - get_centered_cell(iframe)
        #get centroids of cell(iframe), protrusion and retraction
        y,x =np.where(get_centered_cell(iframe))
        centroidy, centroidx = np.median(y), np.median(x)
        yp,xp = np.where(difference==1)
        protrusion_y, protrusion_x  = np.median(yp) , np.median(xp)
        yr,xr = np.where(difference==-1)
        retraction_y, retraction_x  = np.median(yr) , np.median(xr) 
        #get protr and retr angle in a regular coordinate system: [ximage , - yimage]
        #and norm_areas
        protrusion_angles.append( np.arctan2( -(protrusion_y - centroidy) , protrusion_x - centroidx) )
        protrusion_norm_areas.append(len(yp)/ len(y))
        #effective radius (area/pi)^0.5
        protrusion_norm_radii.append( ((protrusion_x-centroidx)**2+(protrusion_y-centroidy)**2)**0.5/(len(y)/np.pi)**0.5 )
        retraction_angles.append(np.arctan2( -(retraction_y - centroidy), retraction_x - centroidx)  )
        retraction_norm_areas.append( len(yr)/len(y) )
        retraction_norm_radii.append( ((retraction_x-centroidx)**2+(retraction_y-centroidy)**2)**0.5/(len(y)/np.pi)**0.5 )


        #PROTRUSION AND RETRACTION angles, radii, calculated with means
        #get centroids of cell(iframe), protrusion and retraction
        mean_centroidy, mean_centroidx = np.mean(y), np.mean(x)
        mean_protrusion_y, mean_protrusion_x  = np.mean(yp) , np.mean(xp)
        mean_retraction_y, mean_retraction_x  = np.mean(yr) , np.mean(xr) 
        #get protr and retr angle in a regular coordinate system: [ximage , - yimage]
        #and norm_areas
        mean_protrusion_angles.append(np.arctan2( -(mean_protrusion_y - mean_centroidy), mean_protrusion_x - mean_centroidx) )
        mean_retraction_angles.append(np.arctan2( -(mean_retraction_y - mean_centroidy), mean_retraction_x - mean_centroidx) )
        mean_protrusion_norm_radii.append( ((mean_protrusion_x-mean_centroidx)**2+(mean_protrusion_y-mean_centroidy)**2)**0.5/(len(y)/np.pi)**0.5 )
        mean_retraction_norm_radii.append( ((mean_retraction_x-mean_centroidx)**2+(mean_retraction_y-mean_centroidy)**2)**0.5/(len(y)/np.pi)**0.5 )


        med_centroids = pd.DataFrame({'median_centroidx':median_centroidx, 'median_centroidy':median_centroidy})

    shape_features= pd.DataFrame({'polarity_angle':cell_angles, 'abs-skew':cell_skews, 
                              'protr_angle':protrusion_angles, 'mean_protr_angle':mean_protrusion_angles, 'protr_norm_area':protrusion_norm_areas, 
                          'retr_angle': retraction_angles, 'mean_retr_angle': mean_retraction_angles, 'retr_norm_area': retraction_norm_areas,
                          'protr_norm_radii':protrusion_norm_radii, 'mean_protr_norm_radii':mean_protrusion_norm_radii,
                          'retr_norm_radii':retraction_norm_radii, 'mean_retr_norm_radii':mean_retraction_norm_radii, })

    tracks[movie][itrack] = pd.concat([tracks[movie][itrack].reset_index(drop=True), track_shape_metrics.reset_index(drop=True)], axis = 1 )

    tracks[movie][itrack] = pd.concat([tracks[movie][itrack].reset_index(drop=True), med_centroids.reset_index(drop=True)], axis = 1 )

    tracks[movie][itrack] = pd.concat([tracks[movie][itrack].reset_index(drop=True), shape_features.reset_index(drop=True)], axis = 1 )

###Save updated tracks

with open(analysis_output_folder+'/tracks_shape.pkl', 'wb') as handle:
    pickle.dump(tracks, handle, protocol=2)