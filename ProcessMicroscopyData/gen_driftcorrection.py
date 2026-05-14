import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.measure import regionprops, regionprops_table, find_contours
import os
from scipy import ndimage
from numpy import linalg as LA
from scipy.linalg import eigh
import glob
import sys

from skimage.util import img_as_uint
from skimage.util import img_as_float32
from skimage import exposure

from image_registration import chi2_shift
from image_registration.fft_tools import shift
import image_registration


base_path = '/proj/telston_lab/projects/data/rf_2025_01_30_Jr20Overnight10x/ARPC2KO'

save_folder_beads = base_path + '/dc_beads'
save_folder_gfp = base_path + '/dc_gfp'

if not os.path.exists(save_folder_beads):
  os.mkdir(save_folder_beads)

if not os.path.exists(save_folder_gfp):
  os.mkdir(save_folder_gfp)


movie_num = int(sys.argv[1])

# path to where bead images are located 
bead_dir = base_path + '/beads'

bead_ref_dir = base_path + '/bead_ref'

gfp_dir = base_path + '/cp_gfp/s{}'.format(movie_num)

gfp_movie_save_folder = save_folder_gfp + '/s{}'.format(movie_num)
if not os.path.exists(gfp_movie_save_folder):
  os.mkdir(gfp_movie_save_folder)

ref_img = imread(bead_ref_dir + '/*_s'+str(movie_num)+'.tif')
# p0, p99 = np.percentile(ref_img, (0, 99.9))
# ref_img_rescale = exposure.rescale_intensity(ref_img, in_range=(p0, p99))

file_list = glob.glob(bead_dir + '/*_s'+str(movie_num)+'_*.tif')

for file in file_list:
    basefilename = os.path.basename(file)
    t_img = imread(file)
    gfp_img = imread(gfp_dir + '/'.format(movie_num) + basefilename)

    # p0, p99 = np.percentile(t_img, (0, 99.9))
    # t_img_rescale = exposure.rescale_intensity(t_img, in_range=(p0, p99))

    image = ref_img
    offset_image = t_img
    # image = img_as_uint(ref_img)
    # offset_image = img_as_uint(t_img)

    #Get Fused Image
    xoff, yoff, exoff, eyoff = chi2_shift(image, offset_image, upsample_factor=20)
    corrected_beadimage = shift.shiftnd(offset_image, (-yoff, -xoff),return_real=True)

    # imsave(save_folder_beads+'/'+basefilename, img_as_uint(corrected_beadimage/np.max(corrected_beadimage)), check_contrast=False)
    imsave(save_folder_beads+'/'+basefilename, corrected_beadimage.astype(np.float32), check_contrast=False)

    corrected_gfpimage = img_as_float32(shift.shiftnd(gfp_img, (-yoff, -xoff),return_real=True))

    imsave(gfp_movie_save_folder+'/'+basefilename, corrected_gfpimage, check_contrast=False)