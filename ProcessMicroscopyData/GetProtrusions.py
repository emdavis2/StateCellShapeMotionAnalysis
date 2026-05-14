import pickle
import os
import numpy as np
import pandas as pd
from skimage.io import imread, imsave, imshow
from skimage.morphology import opening, disk, binary_dilation, convex_hull_image
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

def get_cell_body(mask, radius_fraction=0.1, min_fraction=0.02, visualize=False):

    mask = mask.astype(bool)

    if mask.sum() == 0:
        if visualize:
            plt.imshow(mask, cmap="gray")
            plt.title("Empty Mask")
            plt.axis("off")
            plt.show()
        return np.zeros_like(mask, dtype=bool)

    region = regionprops(label(mask))[0]
    area = region.area

    r_eq = np.sqrt(area / np.pi)

    frac = radius_fraction
    body = None

    while frac >= min_fraction:

        radius = max(2, int(frac * r_eq))

        opened = opening(mask, disk(radius))

        # if opening removed everything → reduce radius
        if opened.sum() == 0:
            frac *= 0.7
            continue

        body_candidate = convex_hull_image(opened)
        body_candidate &= mask

        if body_candidate.sum() > 0:
            body = body_candidate
            break

        frac *= 0.7   # reduce by 30%

    if body is None:
        body = mask.copy()

    if visualize:

        fig, ax = plt.subplots(1,3, figsize=(12,4))

        ax[0].imshow(mask, cmap="gray")
        ax[0].set_title("Original")

        ax[1].imshow(opened, cmap="gray")
        ax[1].set_title(f"Opening (radius={radius})")

        ax[2].imshow(body, cmap="gray")
        ax[2].set_title("Cell Body")

        for a in ax:
            a.axis("off")

        plt.show()

    return body


import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

def measure_protrusions_perp_projection(mask, body, min_area=10):
    """
    Measures protrusions:
        - length = tip -> closest body pixel
        - width = full edge-to-edge perpendicular to tip->base vector using all protrusion pixels
    """
    protrusions = mask & ~body
    lbl = label(protrusions)
    results = []

    body_coords = np.column_stack(np.nonzero(body))
    body_dist = distance_transform_edt(~body)

    for region in regionprops(lbl):
        if region.area < min_area:
            continue

        protrusion = lbl == region.label
        coords = np.column_stack(np.nonzero(protrusion))

        # tip = furthest from body
        d = body_dist[protrusion]
        tip = coords[np.argmax(d)]

        # base = closest body pixel
        d_to_body = np.linalg.norm(body_coords - tip, axis=1)
        base = body_coords[np.argmin(d_to_body)]

        # Vector from base → tip
        v = tip - base
        v_len = np.linalg.norm(v)
        v_unit = v / v_len
        
        # Midpoint along the main axis
        midpoint = base + 0.5 * v
        
        # Perpendicular unit vector
        perp = np.array([-v[1], v[0]]) / v_len
        
        # Relative coordinates to base
        rel_coords = coords - base
        
        # Project onto main axis
        along_axis = rel_coords @ v_unit

        # Length
        length = v_len
        
        # midpoint along axis
        half_len = length / 2
        
        # try increasing tolerance until pixels found
        tolerance = 1
        mid_pixels = []
        
        while len(mid_pixels) == 0 and tolerance < 10:
        
            mid_pixels = rel_coords[np.abs(along_axis - half_len) < tolerance]
        
            tolerance += 1
        
        mid_pixels = np.array(mid_pixels)
        
        # If still empty skip protrusion
        if mid_pixels.size == 0:
            continue
        
        # Project midpoint pixels onto perpendicular axis
        perp_proj = mid_pixels @ perp
        
        # Width = edge-to-edge
        width = perp_proj.max() - perp_proj.min()
        
        # Coordinates for visualization
        w1 = midpoint + perp_proj.min() * perp
        w2 = midpoint + perp_proj.max() * perp
        

        results.append({
            "protrusion_id": region.label,
            "length": length,
            "width": width,
            "tip": tip,
            "base": base,
            "midpoint": midpoint,
            "width_pts": [w1, w2]
        })

    return protrusions, results


main_dir = '/proj/telston_lab/projects/data/DataForTFM/TFM_Data_02242025'

# #make directory to save skeletons
# fig_savepath = main_dir + '/skeleton_figs'
# if not os.path.exists(fig_savepath):
#     os.mkdir(fig_savepath)

all_results = []

exp_list = os.listdir(main_dir)
for exp in exp_list:
    if not exp.startswith(".") and '2025_01_30_Jr20Overnight' not in exp and 'Duo_Seg' not in exp and 'skeleton_figs' not in exp:
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

                        if treatment == 'WT':
                            rad_frac = 0.5
                        elif treatment == 'ARPC2KO':
                            rad_frac = 0.5

                        # savepath = fig_savepath + '/' + track_name
                        # if not os.path.exists(savepath):
                        #     os.mkdir(savepath)

                        t_list = os.listdir(mask_dir)
                        for t_file in t_list:
                            if not t_file.startswith(".") and t_file.endswith('.tif'):
                                mask = imread(mask_dir+'/'+t_file)
                                # print(track_name+'_'+t_file)
                                body = get_cell_body(mask,radius_fraction=rad_frac,visualize=False)

                                protrusions, results = measure_protrusions_perp_projection(mask, body)
                                
                                # savepath_name = savepath + '/' + track_name+ '_' + t_file +'.png'
                                # plot_protrusions(mask, skeleton, results, savepath_name)
                                
                                for r in results:
                                    r['name'] = track_name+'_'+t_file

                                all_results.extend(results)
                                


protrusion_df = pd.DataFrame(all_results)

savepath = '/proj/telston_lab/projects/data/DataForTFM/'

protrusion_df.to_pickle(savepath+'protrusion.pkl')
protrusion_df.to_csv(savepath+'protrusion.csv', index=False)