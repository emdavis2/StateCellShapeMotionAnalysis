import numpy as np
import pandas as pd
import os

# function to get center of cell using binary mask
# uses the approximate medoid method to calculate cell center
def get_center(cell_mask):
    y, x = np.nonzero(cell_mask)
    
    ym_temp, xm_temp = np.median(y), np.median(x)
    imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
    ym, xm = y[imin], x[imin]
    
    return (ym, xm)

##############################################################################################################################################

#function to get the angle between two axes
def axis_angle_difference(theta1, theta2):
    """
    Compute the smallest angle between two axes (in radians),
    ignoring axis direction (i.e., 0° == 180°).

    Parameters
    ----------
    theta1, theta2 : float
        Angles of the axes in radians.

    Returns
    -------
    float
        Smallest difference between the two axes (in radians).
    """
    # Wrap angles to [0, π) since axes are directionless
    t1 = theta1 % np.pi
    t2 = theta2 % np.pi
    
    # Compute absolute difference
    diff = abs(t1 - t2)
    
    # Because axes are periodic over π, take the smaller arc
    return min(diff, np.pi - diff)

##############################################################################################################################################

#function to find the angle between two vectors
def ang_between_2vec(ax, ay, bx, by):
    # normalize x and y components of vectors
    a_norm = np.sqrt(ax**2 + ay**2)
    b_norm = np.sqrt(bx**2 + by**2)
    
    if (a_norm*b_norm) == 0.0:
        theta = np.pi/2
    else:
        theta = np.arccos(np.round(((ax*bx) + (ay*by))/(a_norm*b_norm), decimals=5))
    
    return min(theta, np.pi - theta)


def combine_data(tracksgeo_dict, ellipse_dict, dipole_dict, quad_dict, protrusion_df):

    perimratio_df = pd.read_pickle("./data/perim_ratio_crof.pkl")

    protact_df = pd.read_csv('./data/PrActR_long_format.csv')

    cellgeo_df = pd.read_csv('./data/FiloTrack_long_format.csv')

    pixel_size = 0.645 #microns
    dt = 5 #min

    cellgeo_df['mean_wid'] = cellgeo_df['mean_wid'] * pixel_size
    cellgeo_df['mean_len'] = cellgeo_df['mean_len'] * pixel_size
    cellgeo_df['max_len'] = cellgeo_df['max_len'] * pixel_size
    cellgeo_df['mean_protr_lt'] = cellgeo_df['mean_protr_lt'] * dt

    WT_09062024_Dish1 = np.arange(1,16)
    WT_09272024_Dish5 = np.arange(1,21)
    WT_09182024_Dish9 = np.arange(16,31)
    ARPC2KO_09062024_Dish3 = np.arange(31,46)
    ARPC2KO_09272024_Dish7 = np.arange(41,61)
    
    # Example: list of DataFrames, one per cell
    
    arpc2ko_cells = []
    wt_cells = []
    
    track_names_arpc2ko = []
    track_names_wt = []
    
    for name in tracksgeo_dict.keys():
        step = 3
        if 'ARPC2KO' in name and '20250130' not in name and 'Bleb' not in name:
            track_names_arpc2ko.append(name)
            new_df = {}
            turning_angles = []
            df = tracksgeo_dict[name]
            dx = np.diff(df['approximate-medoidx'].dropna()[::step])*pixel_size
            dy = np.diff(df['approximate-medoidy'].dropna()[::step])*pixel_size
            x_vec = dx
            y_vec = dy
            for ind in range(len(x_vec)-1):
                ang_dir = 1 #np.sign(x_vec[ind]*y_vec[ind+1] - y_vec[ind]*x_vec[ind+1])
                turning_angles.append(ang_dir*ang_between_2vec(x_vec[ind], y_vec[ind], x_vec[ind+1], y_vec[ind+1]))
                
            maj_dip = dipole_dict[name]['major_axis_eigenval'][::step]
            maj_dip_ang = dipole_dict[name]['major_axis_ang'][::step]
            diff = []
            for ind in range(len(maj_dip_ang)-1):
                ax = np.cos(maj_dip_ang)[ind]
                ay = np.sin(maj_dip_ang)[ind]
                bx = np.cos(maj_dip_ang)[ind+1]
                by = np.sin(maj_dip_ang)[ind+1]
                ang_dir = np.sign(ax*by - ay*bx)
                ang_diff = ang_between_2vec(ax, ay, bx, by)
                diff.append(ang_diff*ang_dir)
    
            maj_ell_ang = ellipse_dict[name]['major_axis_ang'][::step]
            ell_diff = []
            for ind in range(len(maj_ell_ang)-1):
                ax = np.cos(maj_ell_ang)[ind]
                ay = np.sin(maj_ell_ang)[ind]
                bx = np.cos(maj_ell_ang)[ind+1]
                by = np.sin(maj_ell_ang)[ind+1]
                ang_dir = np.sign(ax*by - ay*bx)
                ang_diff = ang_between_2vec(ax, ay, bx, by)
                ell_diff.append(ang_diff*ang_dir)
    
            diff_dip_ell = []
            for ind in range(len(maj_ell_ang)):
                ang1 = maj_ell_ang[ind]
                ang2 = maj_dip_ang[ind]
                ang_diff = axis_angle_difference(ang1, ang2)
                diff_dip_ell.append(ang_diff)
    
            limit = min(len(ell_diff), len(turning_angles))
            new_df['type'] = ['ARPC2KO']*limit
            new_df['ang_bt_ell_dip'] = diff_dip_ell[:limit]
            new_df['change_dip_ang'] = diff[:limit]
            new_df['change_ell_ang'] = ell_diff[:limit]
            new_df['step_length'] = np.sqrt(dx**2 + dy**2).flatten()[:limit]
            new_df['change_steplength'] = np.concatenate(([0],np.diff(new_df['step_length'])))[:limit]
            new_df['total_force_mag'] = dipole_dict[name]['total_force_mag'][::step][:limit]
            new_df['avg_trac_mag'] = dipole_dict[name]['avg_trac_mag'][::step][:limit]
            new_df['dip_ratio'] = np.abs((dipole_dict[name]['minor_axis_eigenval']/dipole_dict[name]['major_axis_eigenval'])[::step][:limit])
            new_df['maj_dip'] = dipole_dict[name]['major_axis_eigenval'][::step][:limit]
            new_df['MxxTx'] = np.abs(quad_dict[name]['MxxTx'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MxxTy'] = np.abs(quad_dict[name]['MxxTy'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MxyTx'] = np.abs(quad_dict[name]['MxyTx'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MyyTx'] = np.abs(quad_dict[name]['MyyTx'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MyyTy'] = np.abs(quad_dict[name]['MyyTy'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['eccentricity'] = tracksgeo_dict[name]['eccentricity'].dropna().array[::step][:limit]
            new_df['change_ecc'] = np.diff(tracksgeo_dict[name]['eccentricity'].dropna().array[::step])[:limit]
            new_df['solidity'] = tracksgeo_dict[name]['solidity'].dropna().array[::step][:limit]
            new_df['absskew'] = tracksgeo_dict[name]['abs-skew'].dropna().array[::step][:limit]
            new_df['approximate-medoidx'] = tracksgeo_dict[name]['approximate-medoidx'].dropna().array[::step][:limit]
            new_df['approximate-medoidy'] = tracksgeo_dict[name]['approximate-medoidy'].dropna().array[::step][:limit]
            new_df['polarity_angle'] = tracksgeo_dict[name]['polarity_angle'].dropna().array[::step][:limit]
            new_df['area'] = tracksgeo_dict[name]['area'].iloc[:,0].dropna().array[::step][:limit] * (pixel_size**2)
            new_df['perimeter'] = tracksgeo_dict[name]['perimeter'].dropna().array[::step][:limit] * (pixel_size)
            new_df['perim_area_ratio'] = tracksgeo_dict[name]['perimeter'].dropna().array[::step][:limit] * (pixel_size)/tracksgeo_dict[name]['area'].iloc[:,0].dropna().array[::step][:limit] * (pixel_size**2)
            new_df['frame'] = tracksgeo_dict[name]['frame'].dropna().array[::step][:limit]
            new_df['maj_dip_ang'] = maj_dip_ang[:limit]
            new_df['turning_angle'] = turning_angles[:limit]
            new_df['dx_smooth'] = tracksgeo_dict[name]['dx_smooth'].dropna().array[:limit]
            new_df['dy_smooth'] = tracksgeo_dict[name]['dy_smooth'].dropna().array[:limit]
            new_df['velsmooth_ang'] = np.arctan2(tracksgeo_dict[name]['dy_smooth'].dropna().array,tracksgeo_dict[name]['dx_smooth'].dropna().array)[:limit]
            new_df['experiment'] = tracksgeo_dict[name]['experiment'].dropna().array[:limit]
            new_df['date'] = [int(tracksgeo_dict[name]['experiment'][0].split('_')[0])]*limit
            new_df['movie'] = tracksgeo_dict[name]['movie'].dropna().array[:limit]
            new_df['track_id'] = tracksgeo_dict[name]['track_id'].dropna().array[:limit]
            new_df = pd.DataFrame(new_df)
            date = new_df['date'].iloc[0]
            movie_num = new_df['movie'].iloc[0]
            if date == 20240906:
                if movie_num in ARPC2KO_09062024_Dish3:
                    dishnum = [1]
                else:
                    dishnum = [2]
            elif date == 20240927:
                if movie_num in ARPC2KO_09272024_Dish7:
                    dishnum = [3]
                else:
                    dishnum = [4]
            new_df['dishnum'] = dishnum*limit
            new_df['name'] = (new_df['date']).astype('int').astype('str') + '_' + (new_df['type']).astype('str') + '_movie' + (new_df['movie']).astype('int').astype('str') + '_track' + (new_df['track_id']).astype('int').astype('str') + '_t' + (new_df['frame']).astype('int').astype('str') + '.tif'
            med_protr_length = protrusion_df.groupby(["name"])["length"].median() * pixel_size
            # med_protr_width75 = protrusion_df.groupby(["name"])["width_75"].median() * pixel_size
            med_protr_width = protrusion_df.groupby(["name"])["width"].median() * pixel_size
            protr_num = protrusion_df.groupby(["name"])["protrusion_id"].count()
            new_df = pd.merge(new_df, med_protr_length, on="name")
            # new_df = pd.merge(new_df, med_protr_width75, on="name")
            new_df = pd.merge(new_df, med_protr_width, on="name")
            new_df = pd.merge(new_df, protr_num, on="name")
            new_df = pd.merge(new_df, perimratio_df, on="name")
            new_df = pd.merge(new_df, protact_df, on="name")
            new_df = pd.merge(new_df, cellgeo_df, on="name")
            arpc2ko_cells.append(new_df)
        elif 'WT' in name and '20250130' not in name and 'Bleb' not in name:
            track_names_wt.append(name)
            new_df = {}
            turning_angles = []
            df = tracksgeo_dict[name]
            dx = np.diff(df['approximate-medoidx'].dropna()[::step])*pixel_size
            dy = np.diff(df['approximate-medoidy'].dropna()[::step])*pixel_size
            x_vec = dx
            y_vec = dy
            for ind in range(len(x_vec)-1):
                ang_dir = 1 #np.sign(x_vec[ind]*y_vec[ind+1] - y_vec[ind]*x_vec[ind+1])
                turning_angles.append(ang_dir*ang_between_2vec(x_vec[ind], y_vec[ind], x_vec[ind+1], y_vec[ind+1]))
            
            maj_dip = dipole_dict[name]['major_axis_eigenval'][::step]
            maj_dip_ang = dipole_dict[name]['major_axis_ang'][::step]
            diff = []
            for ind in range(len(maj_dip_ang)-1):
                ax = np.cos(maj_dip_ang)[ind]
                ay = np.sin(maj_dip_ang)[ind]
                bx = np.cos(maj_dip_ang)[ind+1]
                by = np.sin(maj_dip_ang)[ind+1]
                ang_dir = np.sign(ax*by - ay*bx)
                ang_diff = ang_between_2vec(ax, ay, bx, by)
                diff.append(ang_diff*ang_dir)
            
            maj_ell_ang = ellipse_dict[name]['major_axis_ang'][::step]
            ell_diff = []
            for ind in range(len(maj_ell_ang)-1):
                ax = np.cos(maj_ell_ang)[ind]
                ay = np.sin(maj_ell_ang)[ind]
                bx = np.cos(maj_ell_ang)[ind+1]
                by = np.sin(maj_ell_ang)[ind+1]
                ang_dir = np.sign(ax*by - ay*bx)
                ang_diff = ang_between_2vec(ax, ay, bx, by)
                ell_diff.append(ang_diff*ang_dir)
    
            diff_dip_ell = []
            for ind in range(len(maj_ell_ang)):
                ang1 = maj_ell_ang[ind]
                ang2 = maj_dip_ang[ind]
                ang_diff = axis_angle_difference(ang1, ang2)
                diff_dip_ell.append(ang_diff)
    
            limit = min(len(ell_diff), len(turning_angles))
            new_df['type'] = ['WT']*limit
            new_df['ang_bt_ell_dip'] = diff_dip_ell[:limit]
            new_df['change_dip_ang'] = diff[:limit]
            new_df['change_ell_ang'] = ell_diff[:limit]
            new_df['step_length'] = np.sqrt(dx**2 + dy**2).flatten()[:limit]
            new_df['change_steplength'] = np.concatenate(([0],np.diff(new_df['step_length'])))[:limit]
            new_df['total_force_mag'] = dipole_dict[name]['total_force_mag'][::step][:limit]
            new_df['avg_trac_mag'] = dipole_dict[name]['avg_trac_mag'][::step][:limit]
            new_df['dip_ratio'] = np.abs((dipole_dict[name]['minor_axis_eigenval']/dipole_dict[name]['major_axis_eigenval'])[::step][:limit])
            new_df['maj_dip'] = dipole_dict[name]['major_axis_eigenval'][::step][:limit]
            new_df['MxxTx'] = np.abs(quad_dict[name]['MxxTx'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MxxTy'] = np.abs(quad_dict[name]['MxxTy'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MxyTx'] = np.abs(quad_dict[name]['MxyTx'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MyyTx'] = np.abs(quad_dict[name]['MyyTx'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['MyyTy'] = np.abs(quad_dict[name]['MyyTy'][::step][:limit] * (10**9) * ((10**6)**2))
            new_df['eccentricity'] = tracksgeo_dict[name]['eccentricity'].dropna().array[::step][:limit]
            new_df['change_ecc'] = np.diff(tracksgeo_dict[name]['eccentricity'].dropna().array[::step])[:limit]
            new_df['solidity'] = tracksgeo_dict[name]['solidity'].dropna().array[::step][:limit]
            new_df['absskew'] = tracksgeo_dict[name]['abs-skew'].dropna().array[::step][:limit]
            new_df['approximate-medoidx'] = tracksgeo_dict[name]['approximate-medoidx'].dropna().array[::step][:limit]
            new_df['approximate-medoidy'] = tracksgeo_dict[name]['approximate-medoidy'].dropna().array[::step][:limit]
            new_df['polarity_angle'] = tracksgeo_dict[name]['polarity_angle'].dropna().array[::step][:limit]
            new_df['area'] = tracksgeo_dict[name]['area'].iloc[:,0].dropna().array[::step][:limit] * (pixel_size**2)
            new_df['perimeter'] = tracksgeo_dict[name]['perimeter'].dropna().array[::step][:limit] * (pixel_size)
            new_df['perim_area_ratio'] = tracksgeo_dict[name]['perimeter'].dropna().array[::step][:limit] * (pixel_size)/tracksgeo_dict[name]['area'].iloc[:,0].dropna().array[::step][:limit] * (pixel_size**2)
            new_df['frame'] = tracksgeo_dict[name]['frame'].dropna().array[::step][:limit]
            new_df['maj_dip_ang'] = maj_dip_ang[:limit]
            new_df['turning_angle'] = turning_angles[:limit]
            new_df['dx_smooth'] = tracksgeo_dict[name]['dx_smooth'].dropna().array[:limit]
            new_df['dy_smooth'] = tracksgeo_dict[name]['dy_smooth'].dropna().array[:limit]
            new_df['velsmooth_ang'] = np.arctan2(tracksgeo_dict[name]['dy_smooth'].dropna().array,tracksgeo_dict[name]['dx_smooth'].dropna().array)[:limit]
            new_df['experiment'] = tracksgeo_dict[name]['experiment'].dropna().array[:limit]
            new_df['date'] = [int(tracksgeo_dict[name]['experiment'][0].split('_')[0])]*limit
            new_df['movie'] = tracksgeo_dict[name]['movie'].dropna().array[:limit]
            new_df['track_id'] = tracksgeo_dict[name]['track_id'].dropna().array[:limit]
            new_df = pd.DataFrame(new_df)
            date = new_df['date'].iloc[0]
            movie_num = new_df['movie'].iloc[0]
            if date == 20240906:
                if movie_num in WT_09062024_Dish1:
                    dishnum = [1]
                else:
                    dishnum = [2]
            elif date == 20240927:
                if movie_num in WT_09272024_Dish5:
                    dishnum = [3]
                else:
                    dishnum = [4]
            elif date == 20240918:
                if movie_num in WT_09182024_Dish9:
                    dishnum = [5]
                else:
                    dishnum = [6]
            new_df['dishnum'] = dishnum*limit
            new_df['name'] = (new_df['date']).astype('int').astype('str') + '_' + (new_df['type']).astype('str') + '_movie' + (new_df['movie']).astype('int').astype('str') + '_track' + (new_df['track_id']).astype('int').astype('str') + '_t' + (new_df['frame']).astype('int').astype('str') + '.tif'
            med_protr_length = protrusion_df.groupby(["name"])["length"].median() * pixel_size
            # med_protr_width75 = protrusion_df.groupby(["name"])["width_75"].median() * pixel_size
            med_protr_width = protrusion_df.groupby(["name"])["width"].median() * pixel_size
            protr_num = protrusion_df.groupby(["name"])["protrusion_id"].count()
            new_df = pd.merge(new_df, med_protr_length, on="name")
            # new_df = pd.merge(new_df, med_protr_width75, on="name")
            new_df = pd.merge(new_df, med_protr_width, on="name")
            new_df = pd.merge(new_df, protr_num, on="name")
            new_df = pd.merge(new_df, perimratio_df, on="name")
            new_df = pd.merge(new_df, protact_df, on="name")
            new_df = pd.merge(new_df, cellgeo_df, on="name")
            wt_cells.append(new_df)
    
    
    # Get lengths of each DataFrame
    lengths_arpc2ko = [len(arr) for arr in arpc2ko_cells]
    lengths_wt = [len(arr) for arr in wt_cells]
    
    # Combine all into one DataFrame
    pooled_arpc2ko_df = pd.concat(arpc2ko_cells, ignore_index=True)
    pooled_wt_df = pd.concat(wt_cells, ignore_index=True)

    return arpc2ko_cells, wt_cells, lengths_arpc2ko, lengths_wt, pooled_arpc2ko_df, pooled_wt_df