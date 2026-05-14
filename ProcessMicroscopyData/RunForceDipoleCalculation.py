import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.measure import regionprops, regionprops_table, find_contours
import os
import sys
from scipy import ndimage
from numpy import linalg as LA
from scipy.linalg import eigh

#### Note: the expected format of the data for this code is main_folder -> experiment_date -> treatment (ARPC2KO/WT) -> track -> masks/tractions_csv
main_dir = '/proj/telston_lab/projects/data/DataForTFM/40x_TFM_Data_12042025'

# this is specific to the microscope used for data

# pixel_size = 0.645 * (10**(-6)) #micron to m (10x objective)
pixel_size = 0.1625 * (10**(-6)) #micron to m (40x objective)
area_pixel = pixel_size**2 #micron^2


def not_empty(arr):
    return any(any(row) for row in arr)

# function to get center of cell using binary mask
# uses the approximate medoid method to calculate cell center
def get_center(cell_mask):
    y, x = np.nonzero(cell_mask)
    
    ym_temp, xm_temp = np.median(y), np.median(x)
    imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
    ym, xm = y[imin], x[imin]
    
    return (ym, xm)

# get just traction forces and positions for region and frame/timepoint of interest
def get_txty_pos_dmask(t_csv, t_mask):
    x_pos = t_csv[0].to_numpy()
    y_pos = t_csv[1].to_numpy()
    tx = t_csv[2].to_numpy()
    ty = t_csv[3].to_numpy()
    
    
    #get dilated mask (in future change to use mask in dilated_border directory and fill in hole)
    mask2 = t_mask.astype(np.bool_)
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(mask2, struct)
    edges = mask2 ^ erode
    dilate_mask = ndimage.binary_dilation(mask2, structure=struct, iterations=7) #adjust interations to adjust width of dilation
    
    #get tractions and their respective positions that are only within the dilated mask region
    x_mask_pos = (np.where(dilate_mask)[1])
    y_mask_pos = (np.where(dilate_mask)[0])
    
    ind_list = []
    for i in range(len(x_pos)):
        if x_pos[i] in x_mask_pos and y_pos[i] in y_mask_pos:
            potential_y_ind = np.where((y_mask_pos == y_pos[i]))[0]
            potential_x_ind = np.where((x_mask_pos == x_pos[i]))[0]
            for j in potential_y_ind:
                if j in potential_x_ind:
                    ind_list.append(i)
                    
    tx_mask = tx[ind_list]
    ty_mask = ty[ind_list]
    
    tx_pos_mask = x_pos[ind_list]
    ty_pos_mask = y_pos[ind_list]
    
    #get matrix of tractions in x and y that is same shape as image
    tx_matrix = np.zeros(np.shape(t_mask))
    ty_matrix = np.zeros(np.shape(t_mask))

    for ind in range(len(tx)):
        i = x_pos[ind]
        j = y_pos[ind]
        tx_matrix[j, i] = tx[ind]
        ty_matrix[j, i] = ty[ind]
        
        
    #get matrix of traction only in region of dilated mask
    tx_mask_matrix = dilate_mask * tx_matrix
    ty_mask_matrix = dilate_mask * ty_matrix
    
    return tx_mask, ty_mask, tx_pos_mask, ty_pos_mask, tx_mask_matrix, ty_mask_matrix, edges



exp_list = os.listdir(main_dir)
for exp in exp_list:
    if not exp.startswith(".") and not exp.startswith('2024_09_06') and not exp.startswith('2024_09_27') and not exp.startswith('2025_01_30'): ###CHANGE THIS:
        treatment_list = os.listdir(main_dir+'/'+exp)
        for treatment in treatment_list:
            if not treatment.startswith("."):
                track_list = os.listdir(main_dir+'/'+exp+'/'+treatment)
                for track in track_list:
                    if not track.startswith("."):
                        basepath = main_dir+'/'+exp+'/'+treatment+'/'+track
                        t_csv_fold = basepath + '/tractions_csv' #for traction csv files
                        track_maskfold = basepath + '/masks' #path to mask directory for track of interest in fov ref frame
                        
                        #traction files by time
                        time_files = os.listdir(t_csv_fold)
                        time_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                        #corresponding mask for each time
                        mask_files = os.listdir(track_maskfold)
                        mask_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                        
                        fig_savepath_base = basepath + '/figs'

                        if not os.path.exists(fig_savepath_base):
                            os.mkdir(fig_savepath_base)
                        
                        fig_savepath = fig_savepath_base + '/force_dipole'

                        if not os.path.exists(fig_savepath):
                            os.mkdir(fig_savepath)
                            
                        major_axis_angle = []
                        minor_axis_angle = []
                        major_eigenvalues = []
                        minor_eigenvalues = []

                        strain_energy = []
                        total_force_mag = []
                        avg_trac_mag = []
                        max_trac_mag = []

                        for count in range(len(mask_files)):
                            # load in traction csv for this time point
                            t_path = t_csv_fold + '/' + time_files[count]
                            t_csv = pd.read_csv(t_path,header=None,skiprows=1)

                            # # load in displacement csv for this time point
                            # displ_path = displ_csv_fold + '/' + time_files[count]
                            # displ_csv = pd.read_csv(displ_path,header=None,skiprows=1)

                            # get mask for this time point
                            t_mask = imread(track_maskfold + '/' + mask_files[count])
                            if not_empty(t_mask):
                                # get tractions and their positions that are associated with the dilated mask as well as matrix versions of tractions, and lastly the edge of the cell to plot later
                                tx_mask, ty_mask, tx_pos_mask, ty_pos_mask, tx_mask_matrix, ty_mask_matrix, edges = get_txty_pos_dmask(t_csv, t_mask)

                                # # get displacements and their positions that are associated with the dilated mask as well as matrix versions of tractions, and lastly the edge of the cell to plot later
                                # dx_mask, dy_mask, dx_pos_mask, dy_pos_mask, dx_mask_matrix, dy_mask_matrix, edges = get_dxdy_pos_dmask(displ_csv, t_mask)

                                # # strain energy 
                                # se = 0.5 * np.sum((tx_mask_matrix*dx_mask_matrix) + (ty_mask_matrix*dy_mask_matrix))
                                # strain_energy.append(se)

                                # f_tot = np.sum(np.sqrt((tx_mask**2 + ty_mask**2))*area_pixel)
                                f_tot = np.sum(np.sqrt((tx_mask**2 + ty_mask**2)))
                                total_force_mag.append(f_tot)

                                area_cell = len(tx_mask) * area_pixel
                                # t_avg = f_tot/area_cell
                                t_avg = f_tot/len(tx_mask)
                                avg_trac_mag.append(t_avg)

                                # t_max = np.max(np.sqrt(tx_mask**2 + ty_mask**2))
                                # max_trac_mag.append(t_max)


                                # get position of center of cell
                                centroid = get_center(t_mask) 
                                x_center = (centroid[1])
                                y_center = (centroid[0])

                                # find positions of tractions relative to position of cell center
                                pos_x_cent = np.zeros(np.shape(tx_mask_matrix)) #position of x pixels in image relative to cell center
                                pos_y_cent = np.zeros(np.shape(tx_mask_matrix)) #position of y pixels in image relative to cell center
                                for yval in range(0,np.shape(tx_mask_matrix)[0]):
                                    for xval in range(0,np.shape(tx_mask_matrix)[1]):
                                        pos_x_cent[yval][xval]=xval - x_center #in terms of pixels
                                        pos_y_cent[yval][xval]=yval - y_center #in terms of pixels

                                tx_mask_matrix = np.nan_to_num(tx_mask_matrix)
                                ty_mask_matrix = np.nan_to_num(ty_mask_matrix)
                                
                                # calculate force dipole matrix
                                Mxx = np.sum((pos_x_cent * pixel_size * tx_mask_matrix).flatten()) * area_pixel
                                Mxy = np.sum((pos_x_cent * pixel_size * ty_mask_matrix).flatten()) * area_pixel
                                Myx = np.sum((pos_y_cent * pixel_size * tx_mask_matrix).flatten()) * area_pixel
                                Myy = np.sum((pos_y_cent * pixel_size * ty_mask_matrix).flatten()) * area_pixel

                                M = np.array([[Mxx, Mxy], [Myx, Myy]])

                                # find eignenvalues and eigenvectors for force dipole matrix
                                # eigenvalues, eigenvectors = LA.eig(M)
                                eigenvalues, eigenvectors = eigh(M)

                                # get index of major and minor eigenvalues
                                major_pos = np.argmax(np.abs(eigenvalues))
                                minor_pos = np.argmin(np.abs(eigenvalues))

                                # get major and minor eigenvalues
                                major_eigenval = eigenvalues[major_pos]
                                minor_eigenval = eigenvalues[minor_pos]

                                major_eigenvalues.append(major_eigenval)
                                minor_eigenvalues.append(minor_eigenval)

                                # get major and minor eigenvectors
                                major_eigenvec = eigenvectors[:,major_pos]
                                minor_eigenvec = eigenvectors[:,minor_pos]

                                # scale major and minor eigenvalues so they sum to 1
                                major_eigenval_ratio = np.abs(major_eigenval)/(np.abs(major_eigenval) + np.abs(minor_eigenval))
                                minor_eigenval_ratio = np.abs(minor_eigenval)/(np.abs(major_eigenval) + np.abs(minor_eigenval))


                                # # plot tractions used to calculate force dipole, their magnitudes, as well as the major and minor force dipole axes on top of the outline of the cell
                                # plt.figure(dpi=200)
                                # fig = plt.quiver(tx_pos_mask, ty_pos_mask, tx_mask, ty_mask, np.sqrt(tx_mask**2 + ty_mask**2), width=.002, clim=[0,60])
                                # cb = plt.colorbar(fig)
                                # plt.imshow(edges, cmap='binary', origin='lower')

                                major_scale = major_eigenval_ratio * 200
                                minor_scale = minor_eigenval_ratio * 200

                                y_minor = minor_eigenvec[1]
                                x_minor = minor_eigenvec[0]
                                y_major = major_eigenvec[1]
                                x_major = major_eigenvec[0]

                                # ang_minor = np.arctan2(y_minor,x_minor)
                                # ang_major = np.arctan2(y_major,x_major)

                                ang_minor = np.arctan2(y_minor,x_minor)
                                ang_major = np.arctan2(y_major,x_major)

                                major_axis_angle.append(ang_major)
                                minor_axis_angle.append(ang_minor)

                                y_minor_rs = 0.5*minor_scale*np.sin(ang_minor)
                                x_minor_rs = 0.5*minor_scale*np.cos(ang_minor)

                                y_major_rs = 0.5*major_scale*np.sin(ang_major)
                                x_major_rs = 0.5*major_scale*np.cos(ang_major)

#                                 plt.plot([x_center-x_major_rs, x_center+x_major_rs], [y_center-y_major_rs, y_center+y_major_rs])
#                                 plt.plot([x_center-x_minor_rs, x_center+x_minor_rs], [y_center-y_minor_rs, y_center+y_minor_rs])
#                                 plt.xlim(0,np.shape(tx_mask_matrix)[1])
#                                 plt.ylim(0,np.shape(tx_mask_matrix)[0])

#                                 plt.savefig(fig_savepath +'/forcedipole_t{}.png'.format(count))
#                                 plt.clf()
#                                 plt.close()
                                
                        dipole_savepath_base = basepath + '/dipole_analysis'

                        if not os.path.exists(dipole_savepath_base):
                            os.mkdir(dipole_savepath_base)
                            
                        dipole_savepath_majorangle = dipole_savepath_base + '/major_dipole_angle'

                        if not os.path.exists(dipole_savepath_majorangle):
                            os.mkdir(dipole_savepath_majorangle)
                            
                        dipole_savepath_minorangle = dipole_savepath_base + '/minor_dipole_angle'

                        if not os.path.exists(dipole_savepath_minorangle):
                            os.mkdir(dipole_savepath_minorangle)
                            
                        dipole_savepath_majoreigenval = dipole_savepath_base + '/major_dipole_eigenval'

                        if not os.path.exists(dipole_savepath_majoreigenval):
                            os.mkdir(dipole_savepath_majoreigenval)
                            
                        dipole_savepath_minoreigenval = dipole_savepath_base + '/minor_dipole_eigenval'

                        if not os.path.exists(dipole_savepath_minoreigenval):
                            os.mkdir(dipole_savepath_minoreigenval)
                            
                        dipole_savepath_totf = dipole_savepath_base + '/total_force_mag'

                        if not os.path.exists(dipole_savepath_totf):
                            os.mkdir(dipole_savepath_totf)
                            
                        dipole_savepath_avgtrac = dipole_savepath_base + '/avg_trac_mag'

                        if not os.path.exists(dipole_savepath_avgtrac):
                            os.mkdir(dipole_savepath_avgtrac)
                            
                        dipole_savepath_maxtrac = dipole_savepath_base + '/max_trac_mag'

                        if not os.path.exists(dipole_savepath_maxtrac):
                            os.mkdir(dipole_savepath_maxtrac)
                            
                            
                        # np.savetxt(dipole_savepath_strainenergy+'/{}_strainenergy.csv'.format(track_name), strain_energy, delimiter =", ", fmt ='% s')

                        np.savetxt(dipole_savepath_majorangle+'/{}_majoraxisangle.csv'.format(track), major_axis_angle, delimiter =", ", fmt ='% s')
                        np.savetxt(dipole_savepath_minorangle+'/{}_minoraxisangle.csv'.format(track), minor_axis_angle, delimiter =", ", fmt ='% s')
                        np.savetxt(dipole_savepath_majoreigenval+'/{}_majoraxiseigenval.csv'.format(track), major_eigenvalues, delimiter =", ", fmt ='% s')
                        np.savetxt(dipole_savepath_minoreigenval+'/{}_minoraxiseigenval.csv'.format(track), minor_eigenvalues, delimiter =", ", fmt ='% s')

                        np.savetxt(dipole_savepath_totf+'/{}_totalforcemag.csv'.format(track), total_force_mag, delimiter =", ", fmt ='% s')
                        np.savetxt(dipole_savepath_avgtrac+'/{}_avgtracmag.csv'.format(track), avg_trac_mag, delimiter =", ", fmt ='% s')
                        np.savetxt(dipole_savepath_maxtrac+'/{}_maxtracmag.csv'.format(track), max_trac_mag, delimiter =", ", fmt ='% s')