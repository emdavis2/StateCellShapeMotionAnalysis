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
from fastprogress.fastprogress import progress_bar,master_bar
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
import matplotlib.pyplot as plt
from typing import Dict, DefaultDict, Tuple, List, Union

import itertools
import math

import mediapy as media

#Get from Drive the folder libraries
from libraries.filter_cells_fns import remove_multiple_nuclei_cells, remove_large_objects, remove_touching_edge
from libraries.centers import get_centers, fill_label_holes, normalize
from libraries.qc_functions import apply_qc
from libraries.centroidtracker import CentroidTracker

media.set_ffmpeg('/nas/longleaf/home/emae/anaconda3/pkgs/ffmpeg-4.2.2-h20bf706_0/bin/ffmpeg')

#######################################################################################################################

### INPUT PATHS AND SETTINGS ###

# EXPERIMENTAL PARAMETERS

#Experiment Name (will be incorporated into output file and folder names)
experiment = "20250130_Jr20ARPC2KO" ###MODIFY WITH NAME THAT REFLECTS DATA BEING ANALYZED

#Analysis folder: all output analysis data will be output to {analysis output folder}/{experiment name}
analysis_output_folder = "/proj/telston_lab/projects/data/SegmentationAnalysis/"+experiment ###Shouldn't need to modify unless saving output to a different location than telston_lab space on Longleaf

if not os.path.exists(analysis_output_folder):
  os.mkdir(analysis_output_folder)

folderimages = '/proj/telston_lab/projects/data/rf_2025_01_30_Jr20Overnight10x/ARPC2KO/dc_gfp' ###MODIFY WITH PATH TO UNSTACKED TIFF IMAGES TO TRACK
foldermasks = '/proj/telston_lab/projects/data/rf_2025_01_30_Jr20Overnight10x/ARPC2KO/dc_masks' ###MODIFY WITH PATH TO CELL MEMBRANE MASKS
#foldernucmasks = '/proj/telston_lab/projects/data/reformatted/2023_08_26_JR20RandomMigrationPDMS_soft/nucleus_masks' ###MODIFY WITH PATH TO NUCLEUS MASKS

#######################################################################################################################

### ANALYSIS OUTPUT FILENAMES ###

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

### CELL FILTERING ###
def getcells(filecell:Union[Union[str, bytes, os.PathLike],np.ndarray],parameters,return_metrics):
  #membrane
  maskMem:np.ndarray=imread(filecell) if not isinstance(filecell,np.ndarray) else filecell;
#   maskMem[maskMem>0]=1
#   #fill holes
#   maskMem=ndimage.binary_fill_holes(maskMem).astype(int);

#   #label different objectes in masks
#   maskMem,numMem = measure.label(maskMem,return_num=True)

  numMem = len(np.unique(maskMem))

  if numMem < 255:
    maskMem = maskMem.astype('uint8');
  else:
    maskMem = maskMem.astype('uint16');
  
  #FILTERS
  if parameters['remove_cells_touching_edge'] == True:
    maskMem=remove_touching_edge(maskMem)
  
  if parameters['filter_cell_size'] == True:
    maskMem = morphology.remove_small_objects(maskMem, parameters['minareacell'])       
    maskMem = remove_large_objects(maskMem, parameters['maxareacell'])      

  if (return_metrics):
    #if there are cells get metrics
    ids=list(range(1,numMem+1));
    #remove 0 (background) from ids
    # ids.remove(0)
    if len(ids) > 0:
      cellsmetrics = measure.regionprops_table(maskMem, properties=('label','area'))
      cellsmetrics=pd.DataFrame(cellsmetrics)
      if (len(cellsmetrics['label']) > 0 and len(cellsmetrics['area']) > 0):

        #GET CENTERS
        #get labels
        labels=cellsmetrics['label']    
        #Because 'label' was copied from the table, after computing the centers 
        #and concatenating them to the table they should be in the right order
        
        centers=get_centers(maskMem,'approximate-medoid',labels, False)
        #add centers to cell properties
        appmedoid=pd.DataFrame(data=np.asarray(centers),columns=['approximate-medoidx','approximate-medoidy'])
        cellsmetrics=pd.concat([cellsmetrics,appmedoid],axis=1)
        
        centers=get_centers(maskMem,'centroid',labels, False)
        #add centers to cell properties
        centroid=pd.DataFrame(data=np.asarray(centers),columns=['centroidx','centroidy'])
        cellsmetrics=pd.concat([cellsmetrics,centroid],axis=1)
      
        


    else:
      cellsmetrics=pd.DataFrame();
    return cellsmetrics, maskMem
  else:
    return maskMem


### INPUT MOVIE READING PARAMETERS

## Identify movies and frames in input image folder
imagenames=[f for f in os.listdir(foldermasks) if f.endswith('.tif')]
#Get list of movies
movies = [int(re.findall(r"s(\d+).", imagenames[i])[0]) for i in range(len(imagenames)) ]
#get set of unique elements
movies = list(set(movies))
#sort
movies.sort()
print("movies detected in experiment:",movies);
#Get frame numbers and show the largest number
frames = {m:[] for m in movies};
for name in imagenames:
  match = re.findall(r"s(\d+)_t(\d+).", name)[0];
  # print(match);
  frames[int(match[0])].append(int(match[1]));
for m in movies:
  frames[m].sort();
# nframes={m:max(f) for m,f in frames.items()}
# print("frames detected per movie:",nframes);
print(frames);
# imagenames[0]
basename= re.findall(r"(.*)_s",imagenames[0])[0]
print("image basename:",basename);


### INPUT AND SAVE CELL FILTERING PARAMETERS ###

#### Cell location parameters

remove_cells_touching_edge = True 

remove_multi_nuclei_cells = False 


#### Set size bounds (number of pixels)

filter_cell_size = True 


#minarea=600
#maxarea=3200
#nucminarea=21*pixelsize**2

minareacell=60 #MIGHT NEED TO MODIFY 
maxareacell=120000 #MIGHT NEED TO MODIFY

filter_nuc_size = False
minareanuc=5 #MIGHT NEED TO MODIFY

reading_cells_parameters = {
    'experiment':experiment, 
    'basename':basename,
    'movies':movies,
    'frames':frames,
    'remove_cells_touching_edge':remove_cells_touching_edge,
    'remove_multi_nuclei_cells':remove_multi_nuclei_cells,
    'filter_cell_size':filter_cell_size,
    'minareacell':minareacell,
    'maxareacell':maxareacell,
    'filter_nuc_size':filter_nuc_size,
    'minareanuc':minareanuc,
    # 'img_min_intensity':minintensity,
    # 'img_max_intensity':maxintensity
              }


with open(cell_reading_params_path, 'wb') as handle:
    pickle.dump(reading_cells_parameters, handle)
    
    
### GET CELL FEATURES AND LABELLED MASKS ###

folder_incell = foldermasks
#folder_innuc = foldernucmasks

#get cell_nuc_masks names
masknames=[f for f in os.listdir(folder_incell) if f.endswith('.tif')]

#SAVE LABELED MASK SO THAT FEATURES CAN BE EXTRACTED AFTER TRACKING
folder_cellmasks_labeled = analysis_output_folder+'/'+Path(labeled_cellmasks_path).stem
#folder_nucmasks_labeled = analysis_output_folder+'/'+Path(labeled_nucmasks_path).stem
if not os.path.exists(folder_cellmasks_labeled):
  os.mkdir(folder_cellmasks_labeled);
# if not os.path.exists(folder_nucmasks_labeled):
#   os.mkdir(folder_nucmasks_labeled);
try:
  with open(cell_reading_params_path,'rb') as f:
    reading_cells_parameters = pickle.load(f);
except Exception as e:
  raise Exception("Error: missing reading cells parameters, did you make sure to run segmentation quality control first?") from e

basename = reading_cells_parameters['basename'];
frames = reading_cells_parameters['frames'];
movies = reading_cells_parameters['movies'];

#not necessary to specify columns, but this way 'movie' and 'frame' will be at the beginning of the cells data frame
cellmetrics=pd.DataFrame(columns=['movie','frame'],dtype=int);
metriclist = [];
m = master_bar(movies)
for s in m:
  # print(s,frames[s]);
  for fr in progress_bar(frames[s],parent=m):
      filename = folder_incell+'/'+(basename + "_s" + str(s)+'_t'+"{:01d}".format(fr)+'_cp_masks.tif');
      #filenamenuc = folder_innuc+'/'+(basename + "_s" + str(s)+'_t'+str(fr)+'.TIF');
      
      mem = imread(filename);
      #mem = ndimage.binary_fill_holes(mem);
      #mem[mem != 0] = 1;
      # nuc = imread(filenamenuc);
      # nuc[nuc != 0] = 1;
      
      #get cell features in a mask and the labeled mask
      cellmetric, labeledcellmask = getcells(mem,reading_cells_parameters,True);


      #SAVE LABELED MASK SO THAT FEATURES CAN BE EXTRACTED AFTER TRACKING
      imsave(folder_cellmasks_labeled+'/'+(basename + "_s" + str(s)+'_t'+str(fr)+'.TIF') , labeledcellmask,check_contrast=False);
      #imsave(folder_nucmasks_labeled+'/'+(basename + "_s" + str(s)+'_t'+str(fr)+'.TIF') , labelednucmask,check_contrast=False);
      if not cellmetric.empty:
        # print("hello");
        #Add columns corresponding to movie and frame
        cellmetric['movie']=s
        cellmetric['frame']=fr
        #adds metricsmaskcells below cells
        metriclist.append(cellmetric)

cellmetrics = pd.concat(metriclist)

cellmetrics.to_csv(analysis_output_folder+'/cell_features.csv')


#######################################################################################################################


# ### TRACK CELLS ###

# with open(cell_reading_params_path,'rb') as f:
#   reading_cells_parameters = pickle.load(f);

# # Cell Tracking Parameters

# #which center type (centroid or medoid) to use for tracking
# centroidtype = 'approximate-medoid' ###May need to modify
# #maximum jump size allowed in pixels
# max_tracked_speed =  75 ###May need to modify 
# #maximum fractional area change 
# dfracsize=0.99 ###May need to modify



# ## Disappeared Cell Tracking Parameters
# #### Filters on how a track whose that has disappeared can snap to a newly appeared cell

# #maximum frames the tracker will remember a cell's position when not in frame before it gets deleted
# max_track_disappeared_time =  2 ###May need to modify

# #minimum length of continuous track required for persistence through disappeared
# min_track_persistence_time = 2 ###May need to modify

# options = ["Speed (linear)", "Diffusivity (MSD)"]
# class SpeedType(Enum): 
#   linear = "linear";
#   MSD = "MSD";
#   def __str__(self): 
#     return self.value;

# #Speed type - whether to use linear speed or mean squared displacement to filter for the distance a cell can move while disappeared
# untracked_speed_type_str = "Speed (linear)" 
# untracked_speed_type = [SpeedType.linear,SpeedType.MSD][options.index(untracked_speed_type_str)];

# #If using linear speed: maximum average distance a cell that has disappeared can move per frame while still being snapped back to nearest previous position; that is, cell is accepted if distance(C1,C2) < time*max_untracked_speed
# max_untracked_speed = 50 ###May need to modify

# #If using MSD: maximum MSD of an cell that has disappeared moving over the course of t frames to be considered the same as a previously tracked cell; that is, cells will be considered the same if distance(C1,C2)^2 < time*max_untracked_diffusivity. First frame's distance will be sqrt(D), then sqrt(2D), then sqrt(3D), etc
# max_untracked_diffusivity = 5 ###May need to modify

# ###IGNORE THE CELL COLLECTION STUFF
# ### Cell Collection: Whether to have only a specific section of movie where new tracks can be added
# #Probably outdated, initial track delay in track qc is probably better, which does the same thing but throughout the movie with after-the-fact information
# do_cell_collection = False 
# #Cell Collection Time: For how many frames at the begininng of the movie do we still accept new cells
# cell_collection_time = 36 
# ###

# trackinging_parameters={
#     'centroidtype':centroidtype, 
#     'max_tracked_speed':max_tracked_speed, 
#     'dfracsize':dfracsize,
#     'max_disappeared_time':max_track_disappeared_time,
#     'untracked_speed_type':str(untracked_speed_type),
#     'do_cell_collection':do_cell_collection,
#     'cell_collection_time':cell_collection_time,
#     'min_track_persistence_time':min_track_persistence_time};
# if untracked_speed_type == SpeedType.linear:
#   trackinging_parameters['max_untracked_speed'] = max_untracked_speed;
# elif untracked_speed_type == SpeedType.MSD:
#   trackinging_parameters['max_untracked_diffusivity'] = max_untracked_diffusivity;


# with open(track_params_path, 'wb') as handle:
#     pickle.dump(trackinging_parameters, handle)

# #This specify the cell metrics tracking is going to use
# fields = ['label','movie','frame',centroidtype+'x',centroidtype+'y','area']


# #read cell shape features as data frame
# sample_cells:pd.DataFrame = pd.read_csv(analysis_output_folder+'/cell_features.csv', usecols=fields)

# sample:Dict[int,Dict[int,pd.DataFrame]] = {}; #movie index, tracks
# # to_use = [m for m in movies if m not in fails] if 'fails' in locals() else movies;

# class Cell:
#   def __init__(self,frame:pd.DataFrame):
#     self.frame = frame;
#     self.min_dist = None;
  
#   @staticmethod
#   def cell_distance(t0:Cell,t1:Cell)->float:
#       t0_f = t0.frame.iloc[0];
#       t1_f = t1.frame.iloc[0];
#       out = math.sqrt((t0_f[centroidtype+'x']-t1_f[centroidtype+'x'])**2 + (t0_f[centroidtype+'y']-t1_f[centroidtype+'y'])**2);
#       if (t1.min_dist is None or out < t1.min_dist):
#         t1.min_dist = out;
#       return out;

#   def __str__(self):
#     return f"Cell <{builtins.id(self)}>:\nframe: \n" + str(self.frame) + ",\nminimum distance:" + str(self.min_dist)

#   @staticmethod
#   def cells_filter(t0:Cell,t1:Cell,disappeared_time:int,dist:float)->bool:
#     t0_f = None;
#     t1_f = None;
#     good_area = None;
#     try:
#       t0_f = t0.frame.iloc[0];
#       t1_f = t1.frame.iloc[0];
#       # raise Exception();  
#       good_area:bool = float(t1_f['area']) > (float(t0_f['area'])*(1-dfracsize)) and float(t1_f['area']) < (float(t0_f['area'])*(1+dfracsize));
#     except Exception as e:
#       print(t0,t1);
#       print(t0_f,t1_f);
#       print(t0_f['area'],t1_f['area']);
#       print(type(t0_f['area']),type(t1_f['area']));
#       print(good_area);
#       raise e;

#     if disappeared_time == 0:
#         result = dist < max_tracked_speed;
#     else:
#         if untracked_speed_type == SpeedType.linear:
#           result = dist < max_untracked_speed*disappeared_time;
#         elif untracked_speed_type == SpeedType.MSD:
#           result = dist**2 < max_untracked_diffusivity*disappeared_time;
#         else:
#           raise Exception();

#     if not(good_area) and result:
#         result = False;

#     return result


# frames = reading_cells_parameters['frames'];
# movies = reading_cells_parameters['movies'];
# m = master_bar(movies);
# for s in m:
#     print(f"tracking movie #{s}");
#     tracklists:Dict[int,list[pd.DataFrame]] = {} #objectid, frames
#     tracker = CentroidTracker[Cell](
#         Cell.cell_distance,
#         frame_filter=Cell.cells_filter,
#         maxDisappearedFrames=max_track_disappeared_time,
#         minPersistenceFrames=min_track_persistence_time);
#     for fr in progress_bar(frames[s],parent=m):

#         #get cells in current frame
#         maskcells:pd.DataFrame = sample_cells[(sample_cells['movie']==s) & (sample_cells['frame']==fr)]

#         cells = [Cell(maskcells.iloc[[i]]) for i in range(len(maskcells))];
#         objects = tracker.update(cells,allow_new = (fr < cell_collection_time or not(do_cell_collection)));        
#         pass;
#         for id,cell in objects.items():
#             frame = cell.frame.copy();
#             if id in tracklists:
#               if tracker.disappeared[id] > 0:
#                 frame['label'] = -1; #disconnect repeated frame from mask label
#                 frame['frame'] = fr;
#               tracklists[id].append(frame);
#             else:
#               tracklists[id] = [frame];
    
#     #concat all tracks together
#     tracks:Dict[int,pd.DataFrame] = {tid:pd.concat(tracklists[tid],ignore_index=True) for tid in tracklists}
#     sample[s] = tracks

# with open(raw_tracks_path, 'wb') as handle:
#     pickle.dump(sample, handle)


# print("Tracking complete");

# #######################################################################################################################

# ### TRACKING QUALITY CONTROL AND FILTERING ###

# # sample_cells_metrics = pd.read_csv(analysis_output_folder+'/cell_features.csv')
# # reading_cells_parameters = None;
# # with open(cell_reading_params_path,'rb') as handle:
# #   reading_cells_parameters = pickle.load(handle,encoding='latin1');


# # keep: {movie:[track1,track2,etc]} - note that any unspecified movies will be left with all tracks

# # example:
# # keep={4:[1],6:[1],8:[7],11:[1,2],14:[9],17:[4],18:[1],19:[1],23:[3],24:[1,2,6],28:[1]}

# # trim: {(movie,track):(firstframe_keep,lastframe_keep)}; if track is -1, will trim the whole movie to that trim

# # example:
# # trim={(4,1):(1,6),(17,4):(1,10),(25,1):(1,31),(26,1):(1,6)}

# # removemov: [movie]

# # example:
# # removemov=[1,5,10,12,22,27,30]

# # exclude: [movie,track]

# # exclude=[(3,7),(13,2)]

# # Here "movie" is the number assigned by the microscope, in filename [basename]_s[movie]_t[frame].tif

# #####THIS NEEDS TO BE MODIFIED!!!! 
# ####AT THE BEGINNING OF EACH NEW DATASET BEING ANALYZED ALL THE OLD INPUTS FOR THE VARIABLES "keep", "trim", "removemov", AND "exclude" NEED TO BE EMPTY AND AS YOU PERFORM QUALITY CONTROL FOR THIS DATASET MODIFY AS NECESSARY

# #INPUT QC OPERATIONS 
# #the minimum track length to include (frames)
# minTrackLength=30 #MAY NEED TO MODIFY

# #the minimum displacement over the length of the track (pixels)
# minTrackDisplacement=None 

# #how long to wait after a track's appearance before including it (buffer period for cell division/collisions, segmentation issues, etc)
# initialTrackDelay = 0 


# #dict of {movie:[track1,track2,etc]} specific tracks to keep from particular movies; *only* the tracks specified in the movie will be kept
# #when empty should look like: keep={}
# keep={}

# #dict of {(movie,track):(startframe,endframe)} sets the frame bounds of specific tracks in the sample
# #when empty should look like: trim={}
# trim={}  

# #list of specific movies to exclude (overrides keep)
# #when empty should look like: removemov=[]
# removemov=[] 

# #list of (movie,track): specific tracks to exclude
# #when empty should look like: exclude=[]
# exclude=[] 

# in_tracks = {}
# with open(raw_tracks_path, 'rb') as handle:
#   in_tracks = pickle.load(handle, encoding='latin1')

# #apply QC operations
# sampTrStatus, sample = apply_qc(in_tracks,keep,trim,removemov,exclude,minTrackLength=minTrackLength,minTrackDisplacement=minTrackDisplacement,initialTrackDelay=initialTrackDelay);
# ##SampTrStatus: dict of {movie#:statuses}, where statuses is a dict of {trackid:status}; in this case, status = 0 means bad track, status = 1 means good track

# trackingChanged = True;

# qc_output= {'tracks_status':sampTrStatus, 'qc_tracks':sample, 
#             'minTrackLength':minTrackLength,
#             'minTrackDisplacement':minTrackDisplacement,
#             'initialTrackDelay': initialTrackDelay,
#             'keep':keep, 'trim':trim, 'removemov':removemov, 
#             'exclude':exclude}

# with open(qc_params_path, 'wb') as handle:
#     pickle.dump(qc_output, handle)

# tracksreg={}
# for imov in sample:
#   tracksreg[imov] = {}
#   for itr in sample[imov]:
#     #if the track satus == 1 (track passed QC)
#     if sampTrStatus[imov][itr]==1:
#       tracksreg[imov][itr] = (sample[imov][itr])

# with open(qc_tracks_path, 'wb') as f:
#   pickle.dump(tracksreg,f)

# #######################################################################################################################

# ### VISUALIZE TRACKS AND PERFORM TRACK QC ###

# ## Input Tracking Display Parameters
# #Nucleus intensity for combined image (between 0 and 1)
# nucfrac=0.7 

# #Frame downsample rate (how often a frame is shown); increase this number for a faster processing time but less fine time-control
# frspace=1 

# #Mask intensity of cells segmented but not tracked (between 0 and 1);
# untracked_intensity = 0.2 

# #Mask intensity of cells segmented and tracked but excluded from tracking by quality control (will be in red)
# excluded_intensity = 0.3 

# #how much to resize the image by (the more downscaled, the faster the playing but the less the resolution)
# downfrac = 0.5 

# #how large to plot the cell centers (pixels)
# centersize = 2 

# #how large to draw the track id# (scaling from base size)
# textsize =  0.5

# ### Track Display

# #how thick to draw the cell tracks
# trackwidth = 1

# trackingChanged = True;


# images = {} if 'images' not in globals() or trackingChanged else images; #dict of movie: {True:[with_names],False:[without_names]}
# masks = {} if 'masks' not in globals() or trackingChanged else masks; #dict of movie {True:[with_centers],False:[without_centers]}
# tracks_images = {} if 'tracks' not in globals() or trackingChanged else tracks_images; #dict of movie:list[tracks]

# process_ready = False


# try:
#     from google.colab.patches import cv2_imshow
# except:
#     cv2_imshow = cv2.imshow
# ##Prepare tracking display function (Required for computing display frames)
# track_params = {};
# with open(track_params_path,'rb') as handle:
#   track_params = pickle.load(handle,encoding='latin1');
# centertype = track_params['centroidtype'];

# textfont = cv2.FONT_HERSHEY_SIMPLEX


# imagebasename = reading_cells_parameters['basename'];
# movies = reading_cells_parameters['movies'];

# def get_frame(images,masks,tracks_images,frame,showCenters=False,showNames=False,showTime=False,tracks_display="Neither"):
#     image = images[showNames][frame].copy();
#     mask = masks[showCenters][frame].copy();
#     if tracks_display != "Neither":
#       tracks = tracks_images[frame];
#       alpha = np.where((tracks != 0).any(axis=2),1,0); #pixels set to 1 where there is at least one nonzero element of the rgb (not black)
#       if tracks_display in ["Image side","Both"]:
#         overlay_image_alpha(image,tracks,0,0,alpha);

#       if tracks_display in ["Mask side","Both"]:
#         overlay_image_alpha(mask,tracks,0,0,alpha);


#     combined = np.hstack((mask,image))
#     combined = rescale_intensity(combined,out_range=np.uint8).astype(np.uint8);
#     if showTime:
#       combined = cv2.putText(combined,str(frame),(0,combined.shape[0]),textfont,textsize,(255,255,255),1);
#     return combined

# def f(frame,showCenters=False,showNames=False,tracks_display="Neither"):
#   c = get_frame(images,masks,tracks_images,frame,showCenters=showCenters,showNames=showNames);
#   l.set_data(c);
#   fig.canvas.draw() #use with %matplotlib notebook
#   display(fig) #use with %matplotlib inline


# def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
#     """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

#     `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
#     """
#     # Image ranges
#     y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
#     x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

#     # Overlay ranges
#     y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
#     x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

#     # Exit if nothing to do
#     if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
#         return

#     # Blend overlay within the determined ranges
#     img_crop = img[y1:y2, x1:x2,:]
#     img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o,:]
#     alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
#     alpha_inv = 1.0 - alpha

#     img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

# def process_movie(movie:int,frames:list,parentbar=None):
#       if movie not in movies:
#         raise Exception(f"movie selection {movie} not in experimental list of movies {movies}");
#       if not process_ready:
#         raise Exception("Attempted to process movies with out of date parameters; run the \"Prepare Tracking Function\" Cell")
#       movie_tracks = sample[movie];
#       print(f"processing movie #{movie}");
    
      
#       #random color per id, will be consistent for that id within the movie
#       centerColors = DefaultDict(lambda: (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)));
#       grey = (50,50,50);
#       [centerColors.update([(id,grey)]) for id,status in sampTrStatus[movie].items() if not status];
      
#       acc_tracks_image = None;
#       prev_track_pos = None;

#       timage = {True:{},False:{}}
#       tmask = {True:{},False:{}}
#       ttrack_images = {}

#       for i in progress_bar(frames,parent=(parentbar or None)):
#           filename_mask = imagebasename + "_s" + str(movie)+'_t'+"{:01d}".format(i)+'_cp_masks.tif'

#           maskmem = imread(foldermasks+'/'+filename_mask);
          
#           #read image
#           filename_img = 's' + str(movie) + '/' + imagebasename + "_s" + str(movie)+'_t'+"{:01d}".format(i)+".tif";
#           imageorig = imread(folderimages+'/'+filename_img)

#           #rescale image intensity
#           imageorig = rescale_intensity(imageorig);
#           image=resize(imageorig, (maskmem.shape[0] * downfrac , maskmem.shape[1] * downfrac),  anti_aliasing=True); #this order matters so that the data types work out
#           image = rescale_intensity(image,out_range=np.uint8).astype(np.uint8);
#           image = np.stack((image,image,image),axis=2); #make color image
          
          
          
#           trackedLabels = [];
#           rejectedLabels = [];
#           centers = {};
#           for tid,data in movie_tracks.items():
#             fDat = data[data['frame']==i];
#             if not fDat.empty:
#               trackedLabels.append(fDat)
#               if not sampTrStatus[movie][tid]:
#                 rejectedLabels.append(fDat);
#               centers[tid] = fDat[[centertype+'x',centertype+'y']].reset_index();
          
          
#           #get the label of every tracked mask in this frame
#           trackedLabels = pd.concat(trackedLabels)['label'] if len(trackedLabels) > 0 else [];
#           rejectedLabels = pd.concat(rejectedLabels)['label'] if len(rejectedLabels) > 0 else []


#           #create bitmasks of untracked and rejected cells
#           cellmask = (maskmem != 0);
#           untracked = np.isin(maskmem,trackedLabels,invert=True) & cellmask
#           rejected = np.isin(maskmem,rejectedLabels);
          
#           #draw tracks
#           if acc_tracks_image is None:
#             acc_tracks_image = np.zeros(np.array(image.shape),dtype=np.uint8);
#             prev_track_pos = centers;
#           else:
#             for id in centers:
#               if id in prev_track_pos:
#                 prev = prev_track_pos[id];
#                 prev = (int(prev[centertype+'x'][0]*downfrac),int(prev[centertype+'y'][0]*downfrac));

#                 pos = centers[id];
#                 pos = (int(pos[centertype+'x'][0]*downfrac),int(pos[centertype+'y'][0]*downfrac));

#                 acc_tracks_image = cv2.line(acc_tracks_image,prev,pos,centerColors[id],trackwidth);
#             prev_track_pos = centers;

#           ttrack_images[i] = acc_tracks_image.copy();        

#           #unlabel for visualization
#           maskmem[maskmem>=1] = 1

#           #combine membrane and nucleus masks
#           maskcomb = maskmem
#           maskcomb[maskcomb<0]=0; #floating point stuff, this is so stupid

#           #apply untracked and excluded intensities
#           maskcomb[untracked] = maskcomb[untracked]*untracked_intensity;
#           maskcomb[rejected] = maskcomb[rejected]*excluded_intensity

#           #rescale to full intensity, make int8 again
#           maskcomb = rescale_intensity(maskcomb,out_range=np.uint8).astype(np.uint8);

#           #color rejected cells red
#           qcomb = (maskcomb//4).astype(maskcomb.dtype);
#           halfmask = np.where(rejected,qcomb,maskcomb);
          
#           #turn into rgb image
#           maskcomb = np.stack((maskcomb,halfmask,halfmask),axis=2);

#           #downscale
#           maskcomb=resize(maskcomb,(int(maskcomb.shape[0]*downfrac), int(maskcomb.shape[1]*downfrac)),preserve_range=True).astype(np.uint8);

#           #save unannotated frames
#           tmask[False][i] = maskcomb.copy();
#           timage[False][i] = image.copy();

#           #annotate with centers and names
#           for id,pos in centers.items():
#             pos = (int(pos[centertype+'x'][0]*downfrac),int(pos[centertype+'y'][0]*downfrac));
#             maskcomb = cv2.circle(maskcomb,pos,2,centerColors[id],-1);
#             image = cv2.putText(image,str(id),pos,textfont,textsize,centerColors[id],1);

#           #save annotated frames
#           tmask[True][i] = maskcomb;
#           timage[True][i] = image;
#       return timage,tmask,ttrack_images

# process_ready = True


# ### Save all movies
# ## Save Video Output (Mediapy)
# ### (Requires computing display frames for selected movie)
# #Framerate: speed of movie playing in frames per second (not limited by processing time, can go as fast as you want);

# if not os.path.exists(analysis_output_folder + "/tracks_movies"):
#   os.mkdir(analysis_output_folder + "/tracks_movies")

# framerate = 5
# ### Annotation Parameters:
# centers = True 
# names = True 
# frameNumber = True 
# #whether and where to draw the tracks of each cell on the image frames
# display_tracks_side = "Mask side" # ["Neither", "Mask side", "Image side", "Both"]

# for movie_selection in movies:
# #for movie_selection in range(45, 47):
# #for movie_selection in [1]:
    
#     #if you want to process the other movies while loading the selection
#     process_all = False 

#     #Force reprocessing of selected movie (select if running this cell with new data/parameters says "previously prepared)
#     force_reprocess = False 

#     to_process = master_bar(movies) if process_all else [movie_selection];


#     for movie in to_process:
#       frames = reading_cells_parameters['frames'][movie][::frspace];

#       movie_tracked = True;
#       if movie not in images or len(images[movie][True]) != len(frames):
#         movie_tracked = False;
#       if movie not in masks or len(masks[movie][True]) != len(frames):
#         movie_tracked = False;
#       if movie not in tracks_images or len(tracks_images) != len(frames):
#         movie_tracked = False;

#       if not movie_tracked or (movie == movie_selection and force_reprocess):
#         images[movie],masks[movie],tracks_images[movie] = process_movie(movie,frames,to_process if process_all else None);
#     trackingChanged = False;

#     track_save_path = analysis_output_folder + "/tracks_movies/track_mov{}.mp4".format(movie_selection)


#     frlist = [[f] for f in reading_cells_parameters['frames'][movie][::frspace]];
#     try:
#       media.write_video(track_save_path,starmap(partial(get_frame,images[movie_selection],masks[movie_selection],tracks_images[movie_selection],showCenters=centers,showNames=names,showTime=frameNumber,tracks_display=display_tracks_side),progress_bar(frlist)),fps=framerate);
#     except NameError as e:
#       raise Exception("Unable to load movie - did you set up the frame function?") from e;
#     except KeyError as k:
#       raise Exception("Unable to load entire movie - run the parameters and computation cells to ensure full movie complete") from k;