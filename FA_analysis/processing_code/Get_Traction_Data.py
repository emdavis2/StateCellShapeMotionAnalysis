import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.linalg import eigh

from skimage.io import imread, imsave, imshow
from scipy.interpolate import griddata
from scipy import ndimage
import numpy as np
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
import pickle

# function to get center of cell using binary mask
# uses the approximate medoid method to calculate cell center
def get_center(cell_mask):
    y, x = np.nonzero(cell_mask)
    
    ym_temp, xm_temp = np.median(y), np.median(x)
    imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
    ym, xm = y[imin], x[imin]
    
    return (ym, xm)

pixel_size = .108 #um
pixel_area = pixel_size**2

dir1 = '/Volumes/EMD_Drive/TractionForce_20260623_JR20Pax'
dir2 = '/Volumes/EMD_Drive/Crop_TractionForce_20260623_JR20Pax'

mask_dir = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/masks'
crop_mask_dir = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/crop/masks'

save_path = '/Volumes/EMD_Drive'

traction_df = {}

# cell_folders = [f for f in os.listdir(dir) if not f.endswith('.mat')]

cell_folders = [f"{dir1+'/'}{file}" for file in [f for f in os.listdir(dir1) if not f.endswith('.mat')]] + [f"{dir2+'/'}{file}" for file in [f for f in os.listdir(dir2) if not f.endswith('.mat')]]

for fold in cell_folders:

    t_path = fold + '/tractions_csv/time_1.csv'
    t_csv = pd.read_csv(t_path,header=None,skiprows=1)

    x_pos = t_csv[0].to_numpy()
    y_pos = t_csv[1].to_numpy()
    tx = t_csv[2].to_numpy()
    ty = t_csv[3].to_numpy()

    img = imread(fold + '/{}.tif'.format(fold.split('/')[-1]))

    if "Crop" in fold:
       mask_path = crop_mask_dir + '/{}.tif'.format(fold.split('/')[-1])
    else:
       mask_path = mask_dir + '/{}.tif'.format(fold.split('/')[-1])

    mask = imread(mask_path)

    mask2 = mask.astype(np.bool_)
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(mask2, struct)
    edges = mask2 ^ erode
    mask = ndimage.binary_dilation(mask2, structure=struct, iterations=7) #adjust interations to 


    img_shape = np.shape(img)

    tx_matrix = np.zeros(img_shape, dtype=tx.dtype)
    ty_matrix = np.zeros(img_shape, dtype=ty.dtype)

    tx_matrix[y_pos, x_pos] = tx
    ty_matrix[y_pos, x_pos] = ty

    area = np.count_nonzero(mask) * pixel_area
    t_mag = np.sqrt(tx_matrix**2 + ty_matrix**2)
    t_mag_cell = t_mag[mask]


    avg_traction = t_mag_cell.sum() / area



    #dipole section

    tx_mask_matrix = tx_matrix * mask
    ty_mask_matrix = ty_matrix * mask

    # get position of center of cell
    centroid = get_center(mask) 
    x_center = (centroid[1])
    y_center = (centroid[0])
    
    ny, nx = tx_mask_matrix.shape

    x = np.arange(nx)
    y = np.arange(ny)

    X, Y = np.meshgrid(x, y)

    pos_x_cent = X - x_center
    pos_y_cent = Y - y_center

    tx_mask_matrix = np.nan_to_num(tx_mask_matrix)
    ty_mask_matrix = np.nan_to_num(ty_mask_matrix)
   
    # calculate force dipole matrix
    Mxx = np.sum((pos_x_cent * pixel_size * tx_mask_matrix).flatten()) * pixel_area
    Mxy = np.sum((pos_x_cent * pixel_size * ty_mask_matrix).flatten()) * pixel_area
    Myx = np.sum((pos_y_cent * pixel_size * tx_mask_matrix).flatten()) * pixel_area
    Myy = np.sum((pos_y_cent * pixel_size * ty_mask_matrix).flatten()) * pixel_area

    M = np.array([[Mxx, Mxy], [Myx, Myy]])

    # find eignenvalues and eigenvectors for force dipole matrix
    # eigenvalues, eigenvectors = LA.eig(M)
    eigenvalues, eigenvectors = eigh(M)

    # get index of major and minor eigenvalues
    major_pos = np.argmax(np.abs(eigenvalues))
    minor_pos = np.argmin(np.abs(eigenvalues))

    # get major and minor eigenvalues
    major_eigenval = np.abs(eigenvalues[major_pos])
    minor_eigenval = np.abs(eigenvalues[minor_pos])

    dip_ratio = minor_eigenval/major_eigenval

    # get major and minor eigenvectors
    major_eigenvec = eigenvectors[:,major_pos]
    minor_eigenvec = eigenvectors[:,minor_pos]


    df = pd.DataFrame({'avg_trac_mag': [avg_traction], 'traction_mag': [t_mag_cell.sum()], 'dipole_ratio': [dip_ratio]})

    traction_df[fold.split('/')[-1]] = df


with open(save_path+'/traction_df.pkl', 'wb') as handle:
  pickle.dump(traction_df, handle, protocol=2)