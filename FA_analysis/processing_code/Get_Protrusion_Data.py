import pickle
import numpy as np
import os
from skimage.io import imread, imsave, imshow
from skimage.morphology import opening, disk, binary_dilation, convex_hull_image
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.measure import regionprops, regionprops_table, find_contours
import os
import sys
from scipy import ndimage
from numpy import linalg as LA
from scipy.linalg import eigh


from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.morphology import opening, disk, convex_hull_image

def get_cell_body(mask, radius_fraction=0.1, min_fraction=0.02, ds=4, visualize=False):

    mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # --- downsample ---
    small = resize(
        mask.astype(float),
        (mask.shape[0] // ds, mask.shape[1] // ds),
        order=0,
        anti_aliasing=False,
        preserve_range=True
    ).astype(bool)

    # --- regionprops ---
    region = regionprops(label(small))[0]
    area = region.area
    r_eq = np.sqrt(area / np.pi)

    frac = radius_fraction
    body_small = None

    while frac >= min_fraction:

        radius = max(2, int(frac * r_eq))
        opened = opening(small, disk(radius))

        if not opened.any():
            frac *= 0.7
            continue

        
        body_candidate = convex_hull_image(opened) & small

        if body_candidate.any():
            body_small = body_candidate
            break

        frac *= 0.7

    if body_small is None:
        body_small = small.copy()

    # --- upsample ---
    body = resize(
        body_small.astype(float),
        mask.shape,
        order=0,
        anti_aliasing=False,
        preserve_range=True
    ).astype(bool)

    # --- enforce subset ---
    body &= mask

    return body


def compute_fa_stats(fa_labels, fa_area_dict):
    fa_labels = fa_labels[fa_labels > 0]  # remove background
    
    if len(fa_labels) == 0:
        return 0, 0, np.nan
    
    unique_labels = np.unique(fa_labels)
    areas = [fa_area_dict[l] for l in unique_labels]
    
    return (
        len(unique_labels),          # count
        np.sum(areas),               # total area
        np.mean(areas)               # avg size
    )

pixel_S = .108 #um pixel size

def measure_protrusions_forces(mask, body, force, fa_mask, min_area=10, pixel_size=pixel_S ):
    """
    Measures protrusions:
        - length = tip -> closest body pixel
        - width = full edge-to-edge perpendicular to tip->base vector using all protrusion pixels
    """

    pixel_area = pixel_size**2
    
    protrusions = mask & ~body
    protrusions_force = force - body
    lbl = label(protrusions)
    results = []

    body_coords = np.column_stack(np.nonzero(body))
    body_dist = distance_transform_edt(~body)


    fa_lbl = label(fa_mask)
    fa_props = regionprops(fa_lbl)

    # Build lookup: label → area
    fa_area_dict = {r.label: r.area for r in regionprops(fa_lbl)}


    ########Cell body#############
    body_area_pixels = np.sum(body)
    body_area = body_area_pixels * pixel_area

    # Force in cell body
    body_force_vals = force[body]

    body_total_force = np.sum(body_force_vals)

    body_force_avg = (
        body_total_force / body_area_pixels
        if body_area_pixels > 0
        else np.nan
    )

    # FA labels within body
    fa_labels_in_body = fa_lbl[body]

    body_fa_count, body_fa_area, body_fa_avg = compute_fa_stats(
        fa_labels_in_body,
        fa_area_dict
    )

    body_results = pd.DataFrame({
        "body_area": [body_area],
        "body_total_force": [body_total_force],
        "body_force_avg": [body_force_avg],
        "body_fa_count": [body_fa_count],
        "body_fa_area": [body_fa_area * pixel_area],
        "body_fa_avg": [body_fa_avg * pixel_area],
    })

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
    
        # Get force values for protrusion pixels
        force_vals = force[protrusion]
    
        # Total force
        total_force = np.sum(force_vals)
        total_force_avg = total_force / region.area
    
        # Normalize axis position (0 → base, 1 → tip)
        axis_norm = along_axis / length
    
        # Region masks
        base_region = (axis_norm >= 0.0) & (axis_norm <= 0.2)
        mid_region  = (axis_norm >= 0.4) & (axis_norm <= 0.6)
        tip_region  = (axis_norm >= 0.8) & (axis_norm <= 1.0)

        # Map protrusion pixels → FA labels
        fa_labels_in_protrusion = fa_lbl[protrusion]

        # --- WHOLE protrusion FA stats ---
        fa_count_total, fa_area_total, fa_area_avg = compute_fa_stats(
            fa_labels_in_protrusion,
            fa_area_dict
        )

        # Split FA labels by region
        base_fa_labels = fa_labels_in_protrusion[base_region]
        mid_fa_labels  = fa_labels_in_protrusion[mid_region]
        tip_fa_labels  = fa_labels_in_protrusion[tip_region]
    
        # Extract forces by region
        base_force = np.sum(force_vals[base_region])
        mid_force  = np.sum(force_vals[mid_region])
        tip_force  = np.sum(force_vals[tip_region])
    
        # average force
        base_force_avg = base_force / np.sum(base_region) if np.sum(base_region) > 0 else np.nan
        mid_force_avg  = mid_force  / np.sum(mid_region)  if np.sum(mid_region) > 0 else np.nan
        tip_force_avg  = tip_force  / np.sum(tip_region)  if np.sum(tip_region) > 0 else np.nan

        #FA properities
        base_fa_count, base_fa_area, base_fa_avg = compute_fa_stats(base_fa_labels, fa_area_dict)
        mid_fa_count,  mid_fa_area,  mid_fa_avg  = compute_fa_stats(mid_fa_labels, fa_area_dict)
        tip_fa_count,  tip_fa_area,  tip_fa_avg  = compute_fa_stats(tip_fa_labels, fa_area_dict)
    
        results.append({
            "protrusion_id": region.label,
            "protr_area": region.area * pixel_area,
            "length": length * pixel_S,
            "width": width * pixel_S,
            "tip": tip,
            "base": base,
            "midpoint": midpoint,
            "width_pts": [w1, w2],
    
            # force metrics
            "total_force": total_force,
            "base_force": base_force,
            "mid_force": mid_force,
            "tip_force": tip_force,
            "total_force_avg": total_force_avg,
            "base_force_avg": base_force_avg,
            "mid_force_avg": mid_force_avg,
            "tip_force_avg": tip_force_avg,

            # focal adhesion metrics
            "base_fa_count": base_fa_count,
            "mid_fa_count": mid_fa_count,
            "tip_fa_count": tip_fa_count,

            "base_fa_area": base_fa_area * pixel_area,
            "mid_fa_area": mid_fa_area * pixel_area,
            "tip_fa_area": tip_fa_area * pixel_area,

            "base_fa_avg": base_fa_avg * pixel_area,
            "mid_fa_avg": mid_fa_avg * pixel_area,
            "tip_fa_avg": tip_fa_avg * pixel_area,

            # FA metrics (whole protrusion)
            'fa_count_total': fa_count_total,
            'fa_area_total': fa_area_total * pixel_area,
            'fa_area_avg': fa_area_avg * pixel_area,
        })

    return protrusions, results, body_results




dir1 = '/Volumes/EMD_Drive/TractionForce_20260623_JR20Pax'
dir2 = '/Volumes/EMD_Drive/Crop_TractionForce_20260623_JR20Pax'

mask_dir = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/masks'
crop_mask_dir = '/Volumes/EMD_Drive/rf_2026_06_23_JR20Pax/crop/masks'

fa_dir = '/Volumes/EMD_Drive/faas_results_minA20'
fa_crop_dir = '/Volumes/EMD_Drive/faas_results_minA20crop'

save_path = '/Volumes/EMD_Drive'

protrusion_df = {}
body_df = {}


cell_folders = [f"{dir1+'/'}{file}" for file in [f for f in os.listdir(dir1) if not f.endswith('.mat')]] + [f"{dir2+'/'}{file}" for file in [f for f in os.listdir(dir2) if not f.endswith('.mat')]]

for fold in cell_folders:

    t_path = fold + '/tractions_csv/time_1.csv'
    t_csv = pd.read_csv(t_path,header=None,skiprows=1)

    x_pos = t_csv[0].to_numpy()
    y_pos = t_csv[1].to_numpy()
    tx = t_csv[2].to_numpy()
    ty = t_csv[3].to_numpy()


    if "Crop" in fold:
       mask_path = crop_mask_dir + '/{}.tif'.format(fold.split('/')[-1])
       fa_exp = os.listdir(fa_crop_dir + '/{}'.format(fold.split('/')[-1]))[0]
       fa_mask_path = fa_crop_dir + '/{}/{}/individual_pictures/00001/adhesions_binary.png'.format(fold.split('/')[-1], fa_exp)
    else:
       mask_path = mask_dir + '/{}.tif'.format(fold.split('/')[-1])
       fa_exp = os.listdir(fa_dir + '/{}'.format(fold.split('/')[-1]))[0]
       fa_mask_path = fa_dir + '/{}/{}/individual_pictures/00001/adhesions_binary.png'.format(fold.split('/')[-1], fa_exp)

    mask = imread(mask_path)

    if not os.path.isfile(fa_mask_path):
        fa_mask = np.zeros_like(mask, dtype=bool)
    else:
        fa_mask = imread(fa_mask_path)

    if fa_mask.shape != mask.shape:
        fa_mask = resize(
            fa_mask.astype(float),
            mask.shape,
            order=0,
            anti_aliasing=False,
            preserve_range=True
        ).astype(bool)

    mask2 = mask.astype(np.bool_)
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(mask2, struct)
    edges = mask2 ^ erode
    mask = ndimage.binary_dilation(mask2, structure=struct, iterations=7) #adjust interations to 


    img_shape = np.shape(mask)

    tx_matrix = np.zeros(img_shape, dtype=tx.dtype)
    ty_matrix = np.zeros(img_shape, dtype=ty.dtype)

    tx_matrix[y_pos, x_pos] = tx
    ty_matrix[y_pos, x_pos] = ty

    tx_mask_matrix = tx_matrix * mask
    ty_mask_matrix = ty_matrix * mask

    force = np.sqrt(ty_mask_matrix**2 + tx_mask_matrix**2)

    body = get_cell_body(mask,radius_fraction=0.5,visualize=False)
    protrusions, results, body_results = measure_protrusions_forces(mask, body, force, fa_mask, min_area=100)

    df = results
    df_body = body_results

    protrusion_df[fold.split('/')[-1]] = df

    body_df[fold.split('/')[-1]] = df_body



with open(save_path+'/protrusion_df.pkl', 'wb') as handle:
  pickle.dump(protrusion_df, handle, protocol=2)

with open(save_path+'/body_df.pkl', 'wb') as handle:
  pickle.dump(body_df, handle, protocol=2)