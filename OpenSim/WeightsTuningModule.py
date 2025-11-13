import os
import shutil
import numpy as np
import pandas as pd
import optuna
from SensorFusion_automatic import main as sensor_fusion
from ViconIK import main as vicon_ik
import matplotlib.pyplot as plt
import statistics as stats
from scipy.stats import pearsonr
import argparse
from optuna.samplers import TPESampler

def standardize_df(df, exclude_cols=None, return_stats=False):
    """
    Standardize DataFrame columns: (x - mean) / std per column.
    - exclude_cols: iterable of column names to skip (e.g. ['Time', 'time']).
    - return_stats: if True, also return dict of {col: (mean, std)}.
    Non-numeric columns are left unchanged.
    """
    exclude_cols = set(exclude_cols or [])
    df_out = df.copy()
    stats = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        # try to convert to numeric series (will coerce non-numeric -> NaN)
        try:
            series = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            # leave column as-is if conversion fails
            continue
        mean = series.mean()
        std = series.std(ddof=0)  # population std; change ddof=1 if you prefer sample std
        stats[col] = (mean, std)
        if np.isnan(std) or std == 0:
            # avoid division by zero: just subtract mean
            df_out[col] = series - mean
        else:
            df_out[col] = (series - mean) / std
    if return_stats:
        return df_out, stats
    return df_out
def normalize_imu_df_minmax(imu_df, exclude_cols=None, feature_range=(0.0, 1.0), return_stats=False):
    """
    Min-max normalize IMU dataframe columns that contain vector/list values (e.g. quaternions).
    - imu_df: DataFrame with columns like 'time' and IMU columns containing iterable of numeric components.
    - exclude_cols: iterable of column names to skip (default ['time']).
    - feature_range: tuple (min, max) of desired range after scaling.
    - return_stats: if True, also return dict {col: {'min': [...], 'max': [...]}}.
    Non-convertible or malformed columns are left unchanged.
    """
    exclude = set(exclude_cols or ['time'])
    df_out = imu_df.copy()
    stats_dict = {}
    fr_min, fr_max = feature_range

    for col in imu_df.columns:
        if col in exclude:
            continue
        try:
            arr = np.vstack([np.asarray(x, dtype=float) for x in imu_df[col].values])
            # expect shape (N, D)
            if arr.ndim != 2:
                continue
        except Exception:
            continue

        col_min = np.nanmin(arr, axis=0)
        col_max = np.nanmax(arr, axis=0)
        col_range = col_max - col_min
        # avoid division by zero: replace zeros with 1.0 so (x-min)/1 -> 0
        safe_range = np.where(col_range == 0, 1.0, col_range)

        scaled = (arr - col_min) / safe_range
        # scale to feature_range
        scaled = scaled * (fr_max - fr_min) + fr_min

        df_out[col] = [row.tolist() for row in scaled]
        stats_dict[col] = {'min': col_min.tolist(), 'max': col_max.tolist()}

    if return_stats:
        return df_out, stats_dict
    return df_out

def time_correction(subject_ID, trial_ID):
    """ Time synchronisation of the vicon data based on the "raw" imu data since they both are collected at 100Hz"""

    # Read the mot file 
    vicon_trc_path = f'recordings/subject{subject_ID}/vicon_{trial_ID}.trc'
    header_vicon, columns_vicon, vicon_df = read_trc_file(vicon_trc_path)
    # Read the imu data
    imu_sto_path = f'recordings/subject{subject_ID}/imu_{trial_ID}/{trial_ID}_orientations_updatedTime.sto'
    header_imu, columns_imu, imu_df = read_sto_file(imu_sto_path)
    # First change the time vector to start at "imus first timestamp" and delete the Frame# column
    vicon_df['Time'] = vicon_df['Time'] - vicon_df['Time'].iloc[0] + imu_df['time'].iloc[0]
    vicon_df_norm = vicon_df.drop(columns=['Frame#'])

    #print(imu_df.head())
    #print(vicon_df_norm.head())
    # Normalization, but not the time column # ToDo: look into standardisation instead
    #vicon_df = standardize_df(vicon_df, exclude_cols=['Time'])
    vicon_df_norm = (vicon_df_norm - vicon_df_norm.min()) / (vicon_df_norm.max() - vicon_df_norm.min())
    vicon_df_norm['Time'] = imu_df['time'].iloc[0] + 0.01 * np.arange(len(vicon_df_norm))  # assuming 100Hz

    imu_df = normalize_imu_df_minmax(imu_df, exclude_cols=['time'])

    # For debugging, print the first few rows
    #print(vicon_df_norm.head())
    #print(imu_df.head())
    return vicon_df, vicon_df_norm, imu_df, header_vicon, columns_vicon


def find_lag(vicon_df, imu_df):
    """ Find the time lag between the vicon data and the imu data based on cross-correlation of selected segments
        Robust to NaNs in either IMU quaternions or Vicon marker components.
    """
    # Now select some pairs of columns to compare
    humerus_r_imu_quat = imu_df['humerus_r_imu'].tolist()
    humerus_l_imu_quat = imu_df['humerus_l_imu'].tolist()
    radius_r_imu_quat = imu_df['radius_r_imu'].tolist()
    radius_l_imu_quat = imu_df['radius_l_imu'].tolist()
    hand_r_imu_quat = imu_df['hand_r_imu'].tolist()
    hand_l_imu_quat = imu_df['hand_l_imu'].tolist()
    humerus_r_vicon = vicon_df[['X14', 'Y14', 'Z14']].values.tolist()
    humerus_l_vicon = vicon_df[['X7', 'Y7', 'Z7']].values.tolist()
    radius_r_vicon = vicon_df[['X16', 'Y16', 'Z16']].values.tolist()
    radius_l_vicon = vicon_df[['X9', 'Y9', 'Z9']].values.tolist()
    hand_r_vicon = vicon_df[['X19', 'Y19', 'Z19']].values.tolist()
    hand_l_vicon = vicon_df[['X12', 'Y12', 'Z12']].values.tolist()


    # Then find the time lag be using cross-correlation between the vicon data and the imu data
    nb_lags = len(vicon_df)
    lags = np.arange(-nb_lags//10 + 1, nb_lags//10)
    best_lag_hum_l = 0
    best_corr_lag_hum_l = 0
    best_lag_hum_r = 0
    best_corr_lag_hum_r = 0
    best_lag_rad_l = 0
    best_corr_lag_rad_l = 0
    best_lag_rad_r = 0
    best_corr_lag_rad_r = 0
    best_lag_hand_l = 0
    best_corr_lag_hand_l = 0
    best_lag_hand_r = 0
    best_corr_lag_hand_r = 0
    lag_coor = [[], [], [], [], [], []]  # to store lag correlations for each of the 6 segments

    print("Calculating cross-correlations for lag detection...")
    
    for lag in lags:
        corr = 0
        for i in range(len(humerus_l_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(humerus_l_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = humerus_l_imu_quat[i] 
            q = (-np.array(q)).tolist() + [1, 1, 1, 1]
            v = humerus_l_vicon[j]
            if np.isnan(q).any() or np.isnan(v).any():
                corr += 0
            else:
                # Simple dot product as correlation measure
                corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        lag_coor[0].append(corr)
        if corr > best_corr_lag_hum_l:
            best_corr_lag_hum_l = corr
            best_lag_hum_l = lag
        corr = 0
        for i in range(len(humerus_r_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(humerus_r_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = humerus_r_imu_quat[i]
            v = humerus_r_vicon[j]
            if np.isnan(q).any() or np.isnan(v).any():
                corr += 0
            else:  
                # Simple dot product as correlation measure
                corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        lag_coor[1].append(corr)
        if corr > best_corr_lag_hum_r:
            best_corr_lag_hum_r = corr
            best_lag_hum_r = lag
        corr = 0
        for i in range(len(radius_l_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(radius_l_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = radius_l_imu_quat[i]
            q = (-np.array(q)).tolist() + [1, 1, 1, 1]
            v = radius_l_vicon[j]
            if np.isnan(q).any() or np.isnan(v).any():
                corr += 0
            else:
                # Simple dot product as correlation measure
                corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        lag_coor[2].append(corr)
        if corr > best_corr_lag_rad_l:
            best_corr_lag_rad_l = corr
            best_lag_rad_l = lag
        corr = 0
        for i in range(len(radius_r_imu_quat)): 
            j = i + lag
            if j < 0 or j >= len(radius_r_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = radius_r_imu_quat[i]
            v = radius_r_vicon[j]
            if np.isnan(q).any() or np.isnan(v).any():
                corr += 0
            else:
               # Simple dot product as correlation measure
                corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        lag_coor[3].append(corr)
        if corr > best_corr_lag_rad_r:
            best_corr_lag_rad_r = corr
            best_lag_rad_r = lag
        corr = 0
        for i in range(len(hand_l_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(hand_l_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = hand_l_imu_quat[i] 
            q = (-np.array(q)).tolist() + [1, 1, 1, 1]
            v = hand_l_vicon[j]
            if np.isnan(q).any() or np.isnan(v).any():
                corr += 0
            else:
                # Simple dot product as correlation measure
                corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        lag_coor[4].append(corr)
        if corr > best_corr_lag_hand_l:
            best_corr_lag_hand_l = corr
            best_lag_hand_l = lag
        corr = 0
        for i in range(len(hand_r_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(hand_r_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = hand_r_imu_quat[i]
            v = hand_r_vicon[j]
            if np.isnan(q).any() or np.isnan(v).any():
                corr += 0
            else:
                # Simple dot product as correlation measure
                corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        lag_coor[5].append(corr)
        if corr > best_corr_lag_hand_r:
            best_corr_lag_hand_r = corr
            best_lag_hand_r = lag
    # plot lag_corr for all 6 over the lags
    plt.figure()
    plt.plot(lags, lag_coor[0], label='Left Humerus')
    plt.plot(lags, lag_coor[1], label='Right Humerus')
    plt.plot(lags, lag_coor[2], label='Left Radius')
    plt.plot(lags, lag_coor[3], label='Right Radius')
    plt.plot(lags, lag_coor[4], label='Left Hand')
    plt.plot(lags, lag_coor[5], label='Right Hand')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Cross-correlation')
    plt.title('Cross-correlation vs Lag')
    plt.legend()
    plt.show()

    # total correlation 
    array1 = np.array(lag_coor[0])
    array2 = np.array(lag_coor[1])
    array3 = np.array(lag_coor[2])
    array4 = np.array(lag_coor[3])
    array5 = np.array(lag_coor[4])
    array6 = np.array(lag_coor[5])
    total_corr = array1 + array2 + array3 + array4 + array5 + array6
    plt.figure()
    plt.plot(lags, total_corr, label='Total Correlation', color='black')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Total Cross-correlation')
    plt.title('Total Cross-correlation vs Lag')
    plt.legend()
    plt.show()

    print(f'Best lag hum left: {best_lag_hum_l} samples and best lag hum right: {best_lag_hum_r} samples')
    print(f'Best lag rad left: {best_lag_rad_l} samples and best lag rad right: {best_lag_rad_r} samples')
    print(f'Best lag hand left: {best_lag_hand_l} samples and best lag hand right: {best_lag_hand_r} samples')
    best_lag = np.argmax(total_corr) - (nb_lags//10 - 1)
    print(f'Overall best lag: {best_lag} samples')
    
    return best_lag
    
    # Now correct the vicon time vector
def apply_lag_correction(vicon_df, best_lag, imu_df, subject_ID, trial_ID, header_vicon, columns_vicon):
    '''

    '''
    time_shift = - best_lag / 100.0  # assuming 100Hz
    vicon_df['Time'] = vicon_df['Time'] + time_shift
    # Getting the imu data to compare with
    humerus_r_imu_quat = imu_df['humerus_r_imu'].tolist()
    humerus_l_imu_quat = imu_df['humerus_l_imu'].tolist()
    radius_r_imu_quat = imu_df['radius_r_imu'].tolist()
    radius_l_imu_quat = imu_df['radius_l_imu'].tolist()
    hand_r_imu_quat = imu_df['hand_r_imu'].tolist()
    hand_l_imu_quat = imu_df['hand_l_imu'].tolist()

    # helper: nan-safe normalization to [0,1]; preserves NaNs and handles constant arrays
    def safe_normalize(arr, invert=False):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return a
        # if all NaN, return array of NaNs (matplotlib will skip)
        if np.isnan(a).all():
            return a
        amin = np.nanmin(a)
        amax = np.nanmax(a)
        if np.isfinite(amin) and np.isfinite(amax) and (amax - amin) > 0:
            scaled = (a - amin) / (amax - amin)
        else:
            # constant array (or degenerate) -> map finite values to 0, keep NaNs
            scaled = a - amin
            scaled[~np.isfinite(scaled)] = np.nan
            scaled[~np.isnan(scaled)] = 0.0
        if invert:
            scaled = scaled * (-1) + 1
        return scaled

    # Plot to verify, normalized for better visualisation
    imu_plot = [ (q[1] if (hasattr(q, '__len__') and len(q) > 1) else np.nan) for q in humerus_l_imu_quat ]
    imu_plot = safe_normalize(imu_plot)
    vicon_plot = safe_normalize(vicon_df['X7'].values, invert=True)

    # Plot with subplots for left humerus, right humerus, left radius, right radius, left hand, right hand, torso
    plt.figure(figsize=(12, 10))    
    plt.subplot(3, 2, 1)
    plt.plot(imu_df['time'], imu_plot, label='IMU Humerus L X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Humerus L X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Left Humerus X Component')
    plt.legend()

    plt.subplot(3, 2, 2)
    imu_plot = [ (q[1] if (hasattr(q, '__len__') and len(q) > 1) else np.nan) for q in humerus_r_imu_quat ]
    imu_plot = safe_normalize(imu_plot)
    vicon_plot = safe_normalize(vicon_df['X14'].values)
    plt.plot(imu_df['time'], imu_plot, label='IMU Humerus R X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Humerus R X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Right Humerus X Component')
    plt.legend()

    plt.subplot(3, 2, 3)
    imu_plot = [ (q[1] if (hasattr(q, '__len__') and len(q) > 1) else np.nan) for q in radius_l_imu_quat ]
    imu_plot = safe_normalize(imu_plot)
    vicon_plot = safe_normalize(vicon_df['X9'].values, invert=True)
    plt.plot(imu_df['time'], imu_plot, label='IMU Radius L X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Radius L X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Left Radius X Component')
    plt.legend()

    plt.subplot(3, 2, 4)
    imu_plot = [ (q[1] if (hasattr(q, '__len__') and len(q) > 1) else np.nan) for q in radius_r_imu_quat ]
    imu_plot = safe_normalize(imu_plot)
    vicon_plot = safe_normalize(vicon_df['X16'].values)
    plt.plot(imu_df['time'], imu_plot, label='IMU Radius R X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Radius R X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Right Radius X Component')
    plt.legend()

    plt.subplot(3, 2, 5)
    imu_plot = [ (q[1] if (hasattr(q, '__len__') and len(q) > 1) else np.nan) for q in hand_l_imu_quat ]
    imu_plot = safe_normalize(imu_plot)
    vicon_plot = safe_normalize(vicon_df['X12'].values, invert=True)
    plt.plot(imu_df['time'], imu_plot, label='IMU Hand L X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Hand L X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')  
    plt.title('Left Hand X Component')
    plt.legend()

    plt.subplot(3, 2, 6)    
    imu_plot = [ (q[1] if (hasattr(q, '__len__') and len(q) > 1) else np.nan) for q in hand_r_imu_quat ]
    imu_plot = safe_normalize(imu_plot)
    vicon_plot = safe_normalize(vicon_df['X19'].values)
    plt.plot(imu_df['time'], imu_plot, label='IMU Hand R X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Hand R X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Right Hand X Component')
    plt.legend()
    plt.tight_layout()
    plt.show()

    confirmed = input("Are you satisfied with the time correction? (y/n): ")
    #confirmed = 'y'  # for now, assume user is satisfied

    if confirmed.lower() == 'y':
        # Finally, save the corrected vicon file
        # write to a new trc file with the same header
        corrected_vicon_trc_path = f'recordings/subject{subject_ID}/vicon_{trial_ID}.trc'
        # change header_vicon value: line 3, position 7
        tokens = header_vicon[2].rstrip().split('\t')
        tokens[6] = '1'  # OrigDataStartFrame
        header_vicon[2] = '\t'.join(tokens) + '\n'
        with open(corrected_vicon_trc_path, 'w') as f:
            for line in header_vicon:
                f.write(line)
            # write data: simple, set Frame# to 1..N and format numeric values to 6 decimals
            for seq, (_, row) in enumerate(vicon_df.iterrows(), start=1):
                vals = list(row.values)
                # replace the first column (Frame#) with sequential integer
                vals[0] = seq
                out_cells = []
                for idx, v in enumerate(vals):
                    try:
                        if idx == 0:
                            # Frame# should be an integer without decimals
                            out_cells.append(str(int(v)))
                        else:
                            out_cells.append(f"{float(v):.6f}")
                    except Exception:
                        out_cells.append(str(v))
                f.write('\t'.join(out_cells) + '\n')
    else:
        # Undo previous lag correction
        vicon_df['Time'] = vicon_df['Time'] - time_shift
        input_lag = input("Do you want to propose another lag value? (value or no)")
        if input_lag != 'no':
            input_lag = int(input_lag)
            apply_lag_correction(vicon_df, input_lag, imu_df, subject_ID, trial_ID, header_vicon, columns_vicon)
        else: 
            print("Time correction not applied.")

def read_sto_file(filepath):
    """Reads a .sto file and returns the header, column labels, and data as a pandas DataFrame."""
    header = []
    data = []
    columns = []
    row = []
    with open(filepath, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                header.append(line)
                if line.strip().startswith('endheader'):
                    in_header = False
            elif not columns:
                columns = line.strip().split('\t')
            else:
                if line.strip() == '':
                    continue
                # prefer splitting by tab, fallback to whitespace
                if '\t' in line:
                    tokens = line.strip().split('\t')
                else:
                    tokens = line.strip().split()
                # first token is time
                t = float(tokens[0])
                row = [t]
                # for each remaining token which is comma separated quaternion
                for i, col in enumerate(columns[1:], start=1):
                    if i >= len(tokens):
                        # missing data
                        q = [np.nan, np.nan, np.nan, np.nan]
                    else:
                        cell = tokens[i].strip()
                        # tokens may already be like '0.2188,0.8209,-0.0937,0.5189'
                        parts = [p for p in cell.split(',') if p != '']
                        if len(parts) != 4:
                            # try splitting by whitespace if comma not present
                            parts = cell.split()
                        try:
                            q = [float(x) if x else np.nan for x in parts]
                        except:
                            # fallback to NaNs
                            q = [np.nan, np.nan, np.nan, np.nan]
                    row.append(q)
                data.append(row)
                row = []
    df = pd.DataFrame(data, columns=columns)
    return header, columns, df

def read_trc_file(filepath):
    """Reads a .trc file and returns the header, column labels, and data as a pandas DataFrame."""
    header = []
    data = []
    columns = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        header = lines[:5]                 # first five lines as header
        col_line_1 = lines[3]                      # line 4 (index 3)
        col_line_2 = lines[4]                      # line 5 (index 4)
        data_lines = lines[5:]                   # data starts from line 6

        # Column labels
        col_parts_1 = col_line_1.strip().split('\t')
        col_parts_2 = col_line_2.strip().split('\t')

        columns.append(col_parts_1[0])  # Frame#
        columns.append(col_parts_1[1])  # Time
        for i in range(len(col_parts_2)):
            marker_name = col_parts_2[i]
            columns.append(marker_name)

        # Data 
        for line in data_lines:
            # slit by tab
            split_line = line.strip().split('\t')
            data.append([float(x) if x else np.nan for x in split_line])

        df = pd.DataFrame(data, columns=columns)


    return header, columns, df


def read_mot_file(filepath):
    """Reads a .mot file and returns the header, column labels, and data as a pandas DataFrame."""
    header = []
    data = []
    columns = []
    with open(filepath, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                header.append(line)
                if line.strip().startswith('endheader'):
                    in_header = False
            elif not columns:
                columns = line.strip().split('\t')
            else:
                if line.strip() == ' ':
                    continue
                row = [float(x) if x else np.nan for x in line.strip().split()]
                data.append(row)
    df = pd.DataFrame(data, columns=columns)
    return header, columns, df

def compare_joint_angles(df1, df2):
    return df1 - df2

def downsample(groundtruth_df, time_vector): #TODO: check that this is working
    """Downsamples the ground truth DataFrame to match the time vector."""
    rows = []
    # for each time, find the closest time in groundtruth_df and copy the row (keep as DataFrame row)
    for t in time_vector:
        closest_time_idx = (groundtruth_df['time'] - t).abs().idxmin()
        # use double brackets to get a one-row DataFrame instead of a Series
        rows.append(groundtruth_df.iloc[[closest_time_idx]])

    if not rows:
        return pd.DataFrame(columns=groundtruth_df.columns)

    downsampled_df = pd.concat(rows, ignore_index=True)
    # ensure same columns and order
    downsampled_df = downsampled_df[groundtruth_df.columns]
    return downsampled_df

def objective_all(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
    # Define weight parameters to optimize
    webcam_weights = [trial.suggest_int(f'webcam_weight_{i}', 0, 10) for i in range(12)]  # Example: 12 webcam weights
    orientation_weights = [trial.suggest_int(f'orientation_weight_{i}', 0, 10) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [trial.suggest_int(f'stereocamera_weight_{i}', 0, 10) for i in range(15)]  # Example: 15 stereocamera weights

    # Run IK with updated weights
    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    
    # Read latest output
    latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

    # Downsample ground truth to match the time vector of the latest IK results
    time_vector = latest_ik_results_df['time'].values
    ground_truth_df = downsample(ground_truth_df, time_vector)

    # Compare with ground truth
    error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

    # Calculate the RMSE
    rmse = np.sqrt(1/len(latest_ik_results_df) * np.sum(error_df.drop(columns=['time'])**2))

    # Sensor penalty
    if webcam_weights != [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] and stereocamera_weights != [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
        penalty = 200 # high penalty if both webcam and stereocamera are used
    else:
        penalty = 0

    # Sum up the RMSE values
    total = np.sum(rmse) + penalty

    return total 

def objective_webcam(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
    # Define weight parameters to optimize
    webcam_weights = [trial.suggest_int(f'webcam_weight_{i}', 0, 10) for i in range(12)]  # Example: 12 webcam weights
    orientation_weights = [trial.suggest_int(f'orientation_weight_{i}', 0, 10) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Stereocamera not taken into account

    # Run IK with updated weights
    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    
    # Read latest output
    latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

    # Downsample ground truth to match the time vector of the latest IK results
    time_vector = latest_ik_results_df['time'].values
    ground_truth_df = downsample(ground_truth_df, time_vector)

    # Compare with ground truth
    error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

    # Calculate the RMSE
    rmse = np.sqrt(1/len(latest_ik_results_df) * np.sum(error_df.drop(columns=['time'])**2))

    # Sum up the RMSE values
    total = np.sum(rmse)

    return total 

def objective_stereocamera(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
    # Define weight parameters to optimize
    webcam_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Webcam not taken into account
    orientation_weights = [trial.suggest_int(f'orientation_weight_{i}', 0, 10) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [trial.suggest_int(f'stereocamera_weight_{i}', 0, 10) for i in range(15)]  # Example: 15 stereocamera weights

    # Run IK with updated weights
    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    
    # Read latest output
    latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

    # Downsample ground truth to match the time vector of the latest IK results
    time_vector = latest_ik_results_df['time'].values
    ground_truth_df = downsample(ground_truth_df, time_vector)

    # Compare with ground truth
    error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

    # Calculate the RMSE
    rmse = np.sqrt(1/len(latest_ik_results_df) * np.sum(error_df.drop(columns=['time'])**2))

    # Sum up the RMSE values
    total = np.sum(rmse)

    return total 

def objective_imu_only(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
    # Define weight parameters to optimize
    webcam_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Webcam not taken into account
    orientation_weights = [trial.suggest_int(f'orientation_weight_{i}', 0, 10) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  #  Stereocameranot taken into account

    # Run IK with updated weights
    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    
    # Read latest output
    latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

    # Downsample ground truth to match the time vector of the latest IK results
    time_vector = latest_ik_results_df['time'].values
    ground_truth_df = downsample(ground_truth_df, time_vector)

    # Compare with ground truth
    error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

    # Calculate the RMSE
    rmse = np.sqrt(1/len(latest_ik_results_df) * np.sum(error_df.drop(columns=['time'])**2))

    # Sum up the RMSE values
    total = np.sum(rmse)

    return total 


def main(subject_ID=None, trial_ID=None, subject_mass=None, subject_height=None, subject_age=None, subject_sex=None, lag=0):
    # Default weights
    # WEBCAM WEIGHTS: right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, right pinky, left pinky, right index, left index, right hip, left hip
    # ORIENTATION WEIGHTS: torso, pelvis, upper right, lower right, upper left, lower left, hand right, hand left
    # STEREOCAMERA WEIGHTS: neck, right clavicle, left clavicle, right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, spine 3, spine 2, spine 1, pelvis, right hip, left hip

    constraint_var = 1000

    # Sensor fusion result with default weights
    webcam_weights = [1.0] * 12
    orientation_weights = [1.0] * 8
    stereocamera_weights = [1.0] * 15

    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    # Read the output
    latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

    # Time synchronise the vicon data
    vicon_df, vicon_ik_norm, imu_df, header_vicon, columns_vicon = time_correction(subject_ID, trial_ID)
    
    apply_lag_correction(vicon_df, lag, imu_df, subject_ID, trial_ID, header_vicon, columns_vicon)
    

    # Ground truth IK run - Vicon
    ground_truth_ik_file = vicon_ik(subject_ID, trial_ID, constraint_var, subject_mass, subject_height, subject_age, subject_sex) 
    ground_truth_ik_header, ground_truth_ik_columns, ground_truth_df = read_mot_file(ground_truth_ik_file)


    # Downsample ground truth to match the time vector of the latest IK results
    time_vector = latest_ik_results_df['time'].values.tolist()
    ground_truth_df = downsample(ground_truth_df, time_vector)

    # Compare with ground truth
    error_df = compare_joint_angles(latest_ik_results_df, ground_truth_df)
    # delete the column where error is 0
    error_df = error_df.loc[:, (error_df != 0).any(axis=0)]
    error_df['time'] = time_vector
   
    # Make a folder for the analytics 
    results_folder = f'analytics_imu_only/subject_{subject_ID}_trial_{trial_ID}'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    

    # Evaluation metrics 
    # Range of Motion and difference in RoM between vicon and sensor fusion
    rom_vicon_default = ground_truth_df.max() - ground_truth_df.min()
    rom_fusion_default = latest_ik_results_df.max() - latest_ik_results_df.min()
    rom_error_default = rom_vicon_default - rom_fusion_default
    plt.figure(figsize=(10, 6))
    plt.bar(rom_error_default.index, rom_error_default.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('Range of Motion Error (degrees)')
    plt.title('Range of Motion Error between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/rom_error_default_weights.png')

    # RMSE (Root Mean Square Error) for each joint angle
    rmse_default = np.sqrt(1/len(latest_ik_results_df) * np.sum(error_df.drop(columns=['time'])**2))
    plt.figure(figsize=(10, 6))
    plt.bar(rmse_default.index, rmse_default.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('RMSE (degrees)')
    plt.title('Root Mean Square Error between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/rmse_default_weights.png')

    # MAD (Mean Absolute Deviation) for each joint angle
    mad_default = 1/len(latest_ik_results_df) * np.sum(error_df.drop(columns=['time']).abs())
    plt.figure(figsize=(10, 6))
    plt.bar(mad_default.index, mad_default.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('MAD (degrees)')
    plt.title('Mean Absolute Deviation between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/mad_default_weights.png')

    '''

    # Optimization setup
    study_name = "IMU_ONLY_weights_tuning_study_" + subject_ID + "_" + trial_ID
    study = optuna.create_study(direction='minimize',storage="sqlite:///db.sqlite3", study_name=study_name, load_if_exists=True, sampler=TPESampler(multivariate=True))
    # By default, Optuna uses Tree-structured Parzen Estimator algorithm implemented in TPESampler
    study.optimize(lambda trial: objective_imu_only(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex), n_trials=500)

    # Save best trial data in a file 
    file_name = "optuna_results_subject_" + subject_ID + "_trial_" + trial_ID + ".txt"
    file_name = os.path.join(results_folder, file_name)
    with open(file_name, 'w') as f:
        f.write("Best trial parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nBest trial value: " + str(study.best_value) + "\n")
        f.write("\nNumber of trials: " + str(len(study.trials)) + "\n")
    print(f"Results saved to {file_name}")

    # Run sensor fusion with best weights
    best_params = study.best_params
    webcam_weights = [best_params[f'webcam_weight_{i}'] for i in range(12)]
    orientation_weights = [best_params[f'orientation_weight_{i}'] for i in range(8)]
    stereocamera_weights = [best_params[f'stereocamera_weight_{i}'] for i in range(15)]


    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    # Read the output
    output_header, output_columns, output_df = read_mot_file(output)
    output_df.to_csv(results_folder + '/sensor_fusion_best_params_output.csv', index=False)
    # Copy output to results folder
    try:
        shutil.copy(output, results_folder + '/sensor_fusion_best_params_output.mot')
    except Exception as e:
        print(f"Error copying sensor fusion output: {e}")
    # Copy ground truth ik file to results folder
    try:
        shutil.copy(ground_truth_ik_file, results_folder + '/ground_truth_ik_output.mot')
    except Exception as e:
        print(f"Error copying ground truth IK file: {e}")

    # Compare with ground truth
    error_df = compare_joint_angles(output_df, ground_truth_df)
    # delete the column where error is 0
    error_df = error_df.loc[:, (error_df != 0).any(axis=0)]
    error_df['time'] = time_vector

    # Evaluation metrics 
    # Range of Motion and difference in RoM between vicon and sensor fusion
    rom_vicon = ground_truth_df.max() - ground_truth_df.min()
    rom_fusion = output_df.max() - output_df.min()
    rom_error = rom_vicon - rom_fusion
    plt.figure(figsize=(10, 6))
    plt.bar(rom_error.index, rom_error.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('Range of Motion Error (degrees)')
    plt.title('Range of Motion Error between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/rom_error_best_weights.png')

    # RMSE (Root Mean Square Error) for each joint angle
    rmse = np.sqrt(1/len(output_df) * np.sum(error_df.drop(columns=['time'])**2))
    plt.figure(figsize=(10, 6))
    plt.bar(rmse.index, rmse.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('RMSE (degrees)')
    plt.title('Root Mean Square Error between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/rmse_best_weights.png')

    # MAD (Mean Absolute Deviation) for each joint angle
    mad = 1/len(output_df) * np.sum(error_df.drop(columns=['time']).abs())
    plt.figure(figsize=(10, 6))
    plt.bar(mad.index, mad.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('MAD (degrees)')
    plt.title('Mean Absolute Deviation between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/mad_best_weights.png')

    # Plot also the difference in RoMd, RMSE and MAD between default weights and best weights
    rom_error_diff = rom_error_default - rom_error
    plt.figure(figsize=(10, 6))
    plt.bar(rom_error_diff.index, rom_error_diff.values)
    error_df = compare_joint_angles(latest_ik_results_df, ground_truth_df)
    # delete the column where error is 0
    error_df = error_df.loc[:, (error_df != 0).any(axis=0)]
    error_df['time'] = time_vector
   
    # Make a folder for the analytics 
    results_folder = f'analytics_webcam/subject_{subject_ID}_trial_{trial_ID}'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    

    # Evaluation metrics 
    # Range of Motion and difference in RoM between vicon and sensor fusion
    rom_vicon_default = ground_truth_df.max() - ground_truth_df.min()
    rom_fusion_default = latest_ik_results_df.max() - latest_ik_results_df.min()
    rom_error_default = rom_vicon_default - rom_fusion_default
    plt.figure(figsize=(10, 6))
    plt.bar(rom_error_default.index, rom_error_default.values)
    plt.xlabel('Joint Angles')
    plt.xlabel('Joint Angles')
    plt.ylabel('Difference in Range of Motion Error (degrees)')
    plt.title('Difference in Range of Motion Error between Default and Best Weights')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/rom_error_difference.png')

    rmse_diff = rmse_default - rmse
    plt.figure(figsize=(10, 6))
    plt.bar(rmse_diff.index, rmse_diff.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('Difference in RMSE (degrees)')
    plt.title('Difference in RMSE between Default and Best Weights')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/rmse_difference.png')

    mad_diff = mad_default - mad
    plt.figure(figsize=(10, 6))
    plt.bar(mad_diff.index, mad_diff.values)
    plt.xlabel('Joint Angles')  
    plt.ylabel('Difference in MAD (degrees)')
    plt.title('Difference in MAD between Default and Best Weights')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder + '/mad_difference.png')
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subject data.")
    parser.add_argument("--subject_ID", type=str, required=True, help="Subject ID")
    parser.add_argument("--trial_ID", type=str, required=True, help="Trial ID (movement name)")
    parser.add_argument("--subject_mass", type=str, required=True, help="Subject mass (kg)")
    parser.add_argument("--subject_height", type=str, required=True, help="Subject height (mm)")
    parser.add_argument("--subject_age", type=str, required=True, help="Subject age (years)")
    parser.add_argument("--subject_sex", type=str, required=True, help="Subject sex (M/F)")
    parser.add_argument("--lag", type=int, required=True, help="Lag between IMU and vicon for this trial" )
    args = parser.parse_args()

    main(args.subject_ID, args.trial_ID, args.subject_mass, args.subject_height, args.subject_age, args.subject_sex, args.lag)