import os
import shutil
import numpy as np
import pandas as pd
import opensim as osim
import optuna
from SensorFusion_automatic import sensor_fusion_initialization 
from SensorFusion_automatic import run_sensor_fusion
from ViconIK import main as vicon_ik
import matplotlib.pyplot as plt
import statistics as stats
from scipy.stats import pearsonr
import argparse
from optuna.samplers import TPESampler

target_joints_per_trial = {
    "shoulder_flex_ext": ["arm_flex_r", "arm_flex_l"],
    "shoulder_abd_add": ["arm_add_r", "arm_add_l"],
    "shoulder_rot": ["arm_rot_r", "arm_rot_l"],
    "elbow_flex_ext": ["elbow_flex_r", "elbow_flex_l"],
    "elbow_pro_sup": ["pro_sup_r", "pro_sup_l"],
    "wrist_flex_ext": ["wrist_flex_r", "wrist_flex_l"],
    "lumbar_flex_ext": ["lumbar_extension"],
    "lumbar_lat": ["lumbar_bending"],
    "drinking": ["arm_flex_r", "arm_flex_l", "arm_add_r", "arm_add_l", "arm_rot_r", "arm_rot_l", "elbow_flex_r", "elbow_flex_l", "pro_sup_r", "pro_sup_l", "wrist_flex_r", "wrist_flex_l", "lumbar_extension", "lumbar_bending"],
    "finger_nose":["arm_flex_r", "arm_flex_l", "arm_add_r", "arm_add_l", "arm_rot_r", "arm_rot_l", "elbow_flex_r", "elbow_flex_l", "pro_sup_r", "pro_sup_l", "wrist_flex_r", "wrist_flex_l", "lumbar_extension", "lumbar_bending"],
    "clapping": ["arm_flex_r", "arm_flex_l", "arm_add_r", "arm_add_l", "arm_rot_r", "arm_rot_l", "elbow_flex_r", "elbow_flex_l", "pro_sup_r", "pro_sup_l", "wrist_flex_r", "wrist_flex_l", "lumbar_extension", "lumbar_bending"]
    }
joints_of_interest = [
    "arm_flex_r", "arm_flex_l", "arm_add_r", 
    "arm_add_l", "arm_rot_r", "arm_rot_l", 
    "elbow_flex_r", "elbow_flex_l", "pro_sup_r", 
    "pro_sup_l", "wrist_flex_r", "wrist_flex_l", 
    "lumbar_extension", "lumbar_bending", "lumbar_rotation"
    ]
max_range_of_motion_per_joint = {
    "arm_flex_r": 270.0,
    "arm_flex_l": 270.0,
    "arm_add_r": 210.0,
    "arm_add_l": 210.0,
    "arm_rot_r": 180.0,
    "arm_rot_l": 180.0,
    "elbow_flex_r": 180.0,
    "elbow_flex_l": 180.0,
    "pro_sup_r": 180.0,
    "pro_sup_l": 180.0,
    "wrist_flex_r": 140.0,
    "wrist_flex_l": 1.0,
    "lumbar_extension": 180.0,
    "lumbar_bending": 180.0,
    "lumbar_rotation": 180.0
}

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

    # Read the trc file 
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

def apply_lag_correction(vicon_df, best_lag, imu_df, subject_ID, trial_ID, header_vicon, columns_vicon):
    '''
    Apply the lag correction to the vicon dataframe and plot the results for verification.

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

    '''
    # Plot to verify, normalized for better visualisation
    imu_plot = [ (q[1] if (hasattr(q, '__len__') and len(q) > 1) else np.nan) for q in humerus_l_imu_quat ]
    imu_plot = safe_normalize(imu_plot)
    vicon_plot = safe_normalize(vicon_df['X7'].values, invert=True)

    # Plot with subplots for left humerus, right humerus, left radius, right radius, left hand, right hand, torso
    plt.clf()
    plt.close('all')
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
    #plt.show()
    '''

    #confirmed = input("Are you satisfied with the time correction? (y/n): ")
    confirmed = 'y'  # for now, assume user is satisfied

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

def normalize_error_rom(error_df):
    """Normalize the error DataFrame with regards to max range of motion of each joint."""
    normalized_error_df = error_df.copy()
    for joint in error_df.columns:
        if joint == 'time':
            continue
        max_rom = max_range_of_motion_per_joint.get(joint, None)
        if max_rom is not None and max_rom != 0:
            normalized_error_df[joint] = error_df[joint] / max_rom
        else:
            normalized_error_df[joint] = error_df[joint]  # leave unchanged if no max ROM defined
    return normalized_error_df

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


def objective_imu_only(sensor_fusion, trial, ground_truth_df, constraint_var, subject_ID, trial_ID):
    # Define weight parameters to optimize
    webcam_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Webcam not taken into account
    orientation_weights = [trial.suggest_int(f'orientation_weight_{i}', 0, 1) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  #  Stereocameranot taken into account

    # Run IK with updated weights
    try:
        output_storage = run_sensor_fusion(sensor_fusion, webcam_weights, orientation_weights, stereocamera_weights, constraint_var)
        # Save the results to mot file
        output = f'analytics/subject_{subject_ID}/{trial_ID}/optimization/optuna_trial_{trial.number}ik_results_subject_{subject_ID}_trial_{trial_ID}.mot'
        output_storage.printResult(output_storage, f"optuna_trial_{trial.number}ik_results_subject_{subject_ID}_trial_{trial_ID}", f'analytics/subject_{subject_ID}/{trial_ID}/optimization/', -1, ".mot")

        # Read latest output
        latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

        # Downsample ground truth to match the time vector of the latest IK results
        time_vector = latest_ik_results_df['time'].values
        ground_truth_df = downsample(ground_truth_df, time_vector)
        # DATAFRAME downsample groundtruth df to match time vector of latest_ik_results_df 

        # Compare with ground truth
        error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

        # Normalize error df with regards to max range of motion of each joint
        error_df = normalize_error_rom(error_df)

        # Calculate the RMSE
        rmse = np.sqrt((error_df.drop(columns=['time'])**2).sum(axis=0) / len(latest_ik_results_df))

        # Sum up the RMSE values only for the joints of interest
        total = 0 
        for joint in joints_of_interest:
            total = total + rmse[joint]

        # Delete the output file to save space
        os.remove(output)
    
    except Exception as e:
        print(f"Error during optimization trial {trial.number}: {e}")
        total = float('inf')  # Assign a large penalty value in case of error
        # Set optuna trial as failed
        raise optuna.TrialPruned()

    return total 

def objective_imu_webcam(sensor_fusion, trial, ground_truth_df, constraint_var, subject_ID, trial_ID):
    # Define weight parameters to optimize
    webcam_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # webcam weights
    orientation_weights = [trial.suggest_int(f'orientation_weight_{i}', 0, 1) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  #  Stereocameranot taken into account

    # Run IK with updated weights
    try:
        output_storage = run_sensor_fusion(sensor_fusion, webcam_weights, orientation_weights, stereocamera_weights, constraint_var)
        # Save the results to mot file
        output = f'analytics/subject_{subject_ID}/{trial_ID}/optimization/optuna_trial_{trial.number}ik_results_subject_{subject_ID}_trial_{trial_ID}.mot'
        output_storage.printResult(output_storage, f"optuna_trial_{trial.number}ik_results_subject_{subject_ID}_trial_{trial_ID}", f'analytics/subject_{subject_ID}/{trial_ID}/optimization/', -1, ".mot")

        # Read latest output
        latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

        # Downsample ground truth to match the time vector of the latest IK results
        time_vector = latest_ik_results_df['time'].values
        ground_truth_df = downsample(ground_truth_df, time_vector)

         # Compare with ground truth
        error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

        # Normalize error df with regards to max range of motion of each joint
        error_df = normalize_error_rom(error_df)

        # Calculate the RMSE
        rmse = np.sqrt((error_df.drop(columns=['time'])**2).sum(axis=0) / len(latest_ik_results_df))

        # Sum up the RMSE values only for the joints of interest
        total = 0 
        for joint in joints_of_interest:
            total = total + rmse[joint]

        # Delete the output file to save space
        os.remove(output)
    except Exception as e:
        print(f"Error during optimization trial {trial.number}: {e}")
        total = float('inf')  # Assign a large penalty value in case of error
        # Set optuna trial as failed
        raise optuna.TrialPruned()

    return total 

def objective_imu_stereocamera(sensor_fusion, trial, ground_truth_df, constraint_var, subject_ID, trial_ID):
    # Define weight parameters to optimize
    webcam_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Example: 12 webcam weights
    orientation_weights = [trial.suggest_int(f'orientation_weight_{i}', 0, 1) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Example: 15 stereocamera weights

    # Run IK with updated weights
    try:
        output_storage = run_sensor_fusion(sensor_fusion, webcam_weights, orientation_weights, stereocamera_weights, constraint_var)
        # Save the results to mot file
        output = f'analytics/subject_{subject_ID}/{trial_ID}/optimization/optuna_trial_{trial.number}ik_results_subject_{subject_ID}_trial_{trial_ID}.mot'
        output_storage.printResult(output_storage, f"optuna_trial_{trial.number}ik_results_subject_{subject_ID}_trial_{trial_ID}", f'analytics/subject_{subject_ID}/{trial_ID}/optimization/', -1, ".mot")

        # Read latest output
        latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(output)

        # Downsample ground truth to match the time vector of the latest IK results
        time_vector = latest_ik_results_df['time'].values
        ground_truth_df = downsample(ground_truth_df, time_vector)

        # Compare with ground truth
        error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

        # Normalize error df with regards to max range of motion of each joint
        error_df = normalize_error_rom(error_df)

        # Calculate the RMSE
        rmse = np.sqrt((error_df.drop(columns=['time'])**2).sum(axis=0) / len(latest_ik_results_df))

        # Sum up the RMSE values only for the joints of interest
        total = 0 
        for joint in joints_of_interest:
            total = total + rmse[joint]

        # Delete the output file to save space
        os.remove(output)
    except Exception as e:
        print(f"Error during optimization trial {trial.number}: {e}")
        total = float('inf')  # Assign a large penalty value in case of error
        # Set optuna trial as failed
        raise optuna.TrialPruned()

    return total 


def main(subject_ID=None, trial_ID=None, subject_mass=None, subject_height=None, subject_age=None, subject_sex=None, lag=0, optimization=True):
    ## SENSOR FUSION: Default weights

    # WEBCAM WEIGHTS: right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, right pinky, left pinky, right index, left index, right hip, left hip
    # ORIENTATION WEIGHTS: torso, pelvis, upper right, lower right, upper left, lower left, hand right, hand left
    # STEREOCAMERA WEIGHTS: neck, right clavicle, left clavicle, right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, spine 3, spine 2, spine 1, pelvis, right hip, left hip

    constraint_var = 9000

    # Sensor fusion result with default weights
    webcam_weights = [1.0] * 12
    orientation_weights = [1.0] * 8
    stereocamera_weights = [1.0] * 15

    # Make a folder for the analytics 
    results_folder_default = f'analytics/subject_{subject_ID}/{trial_ID}/default_weights/'
    if not os.path.exists(results_folder_default):
        os.makedirs(results_folder_default)

    # Initialize and run sensor fusion with default weights
    fusion_default = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    default_results = run_sensor_fusion(fusion_default, webcam_weights, orientation_weights, stereocamera_weights, constraint_var)

    # Save the results to mot file
    motFileName = results_folder_default + "ik_results_subject_"+str(fusion_default.subject_ID)+"_trial_"+str(fusion_default.trial_ID)+".mot"
    default_results.printResult(default_results, "ik_results_subject_"+str(fusion_default.subject_ID)+"_trial_"+str(fusion_default.trial_ID), results_folder_default, -1, ".mot")
    print(f"Results saved to: {motFileName}")
    
    # Read the output
    latest_ik_results_header, latest_ik_results_columns, latest_ik_results_df = read_mot_file(motFileName)


    ## VICON
    # Time synchronise the vicon data
    vicon_df, vicon_ik_norm, imu_df, header_vicon, columns_vicon = time_correction(subject_ID, trial_ID)
    apply_lag_correction(vicon_df, lag, imu_df, subject_ID, trial_ID, header_vicon, columns_vicon)
    
    # Make a folder for the analytics 
    results_folder_gt = f'analytics/subject_{subject_ID}/{trial_ID}/ground_truth/'
    if not os.path.exists(results_folder_gt):
        os.makedirs(results_folder_gt)

    # Ground truth IK run - Vicon
    ground_truth_ik_file = vicon_ik(subject_ID, trial_ID, constraint_var, subject_mass, subject_height, subject_age, subject_sex)
    # Move ground truth ik file to results folder
    shutil.move(ground_truth_ik_file, os.path.join(results_folder_gt, os.path.basename(ground_truth_ik_file)))
    ground_truth_ik_file = os.path.join(results_folder_gt, os.path.basename(ground_truth_ik_file))
    # Read ground truth ik file
    ground_truth_ik_header, ground_truth_ik_columns, ground_truth_df = read_mot_file(ground_truth_ik_file)


    # Downsample ground truth to match the time vector of the latest IK results
    time_vector = latest_ik_results_df['time'].values.tolist()
    ground_truth_df_downsampled = downsample(ground_truth_df, time_vector)  ## TODO!!


    ## ANALYTICS - Default weights
    # Compare with ground truth
    error_df = compare_joint_angles(latest_ik_results_df, ground_truth_df_downsampled)
    # delete the column where error is 0
    error_df = error_df.loc[:, (error_df != 0).any(axis=0)]
    error_df['time'] = time_vector

    # Evaluation metrics 
    # Range of Motion and difference in RoM between vicon and sensor fusion
    rom_vicon_default = ground_truth_df_downsampled.max() - ground_truth_df_downsampled.min()
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
    plt.savefig(results_folder_default + '/rom_error_default_weights.png')

    # RMSE (Root Mean Square Error) for each joint angle
    rmse_default = np.sqrt((error_df.drop(columns=['time'])**2).sum(axis=0) / len(latest_ik_results_df))
    plt.figure(figsize=(10, 6))
    plt.bar(rmse_default.index, rmse_default.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('RMSE (degrees)')
    plt.title('Root Mean Square Error between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder_default + '/rmse_default_weights.png')

    # MAD (Mean Absolute Deviation) for each joint angle
    mad_default = (error_df.drop(columns=['time']).abs()).sum(axis=0) / len(latest_ik_results_df)
    plt.figure(figsize=(10, 6))
    plt.bar(mad_default.index, mad_default.values)
    plt.xlabel('Joint Angles')
    plt.ylabel('MAD (degrees)')
    plt.title('Mean Absolute Deviation between Vicon and Sensor Fusion Results')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(results_folder_default + '/mad_default_weights.png')
    

    if optimization: 
        # Number of trials for optimization
        nb_trials = 300 #TODO: CHANGE DEPENDING ON THE COMPUTE POWER

        # Make a folder for the analytics 
        results_folder_opt = f'analytics/subject_{subject_ID}/{trial_ID}/optimization/grid_search/'
        if not os.path.exists(results_folder_opt):
            os.makedirs(results_folder_opt)
        
        # Initialize Sensor Fusion for optimization
        orientation_weights = [1.0] * 8
        webcam_weights = [0.0] * 12  # Webcam not taken into account
        stereocamera_weights = [0.0] * 15  #  Stereocamera not taken into account
        fusion_opt = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

        # IMU ONLY WEIGHTS OPTIMIZATION
        # Maske a folder to save optimization trials
        results_folder_opt_imu = results_folder_opt + "imu_only/"
        if not os.path.exists(results_folder_opt_imu):
            os.makedirs(results_folder_opt_imu)
        # Optimization setup
        study_name = "imu_only_weights_tuning_study_" + subject_ID + "_" + trial_ID + "_" + str(nb_trials)
        #study = optuna.create_study(direction='minimize',storage="sqlite:///db.sqlite3", study_name=study_name, load_if_exists=True, sampler=TPESampler(multivariate=True))
        #study = optuna.create_study(direction='minimize', sampler=TPESampler(multivariate=True))
        search_space = {}
        for i in range(8):
            search_space[f"orientation_weight_{i}"] = [0, 1]
        print("SEARCH_SPACE:", search_space)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space))
        # By default, Optuna uses Tree-structured Parzen Estimator algorithm implemented in TPESampler
        #study.optimize(lambda trial: objective_imu_only(fusion_opt, trial, ground_truth_df, constraint_var, subject_ID, trial_ID), n_trials=nb_trials, n_jobs=10)
        study.optimize(lambda trial: objective_imu_only(fusion_opt, trial, ground_truth_df, constraint_var, subject_ID, trial_ID), n_jobs=1)



        # Save best trial data in a file 
        file_name = "optuna_results_subject_" + subject_ID + "_trial_" + trial_ID + ".txt"
        file_name = os.path.join(results_folder_opt_imu, file_name)
        with open(file_name, 'w') as f:
            f.write("Best trial parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write("\nBest trial value: " + str(study.best_value) + "\n")
            f.write("\nNumber of trials: " + str(len(study.trials)) + "\n")
        print(f"Results saved to {file_name}")

        # Extract parameters for plotting
        trials_data = []

        for trial in study.trials:
            trial_data = {"number": trial.number, "value": trial.value}
            # Assuming your parameters are named 'orientation_weight_0', 'orientation_weight_1', ..., 'orientation_weight_7'
            for key in sorted(trial.params.keys()):
                trial_data[key] = trial.params[key]
            trials_data.append(trial_data)

        # Create a DataFrame for easier plotting
        df_trials = pd.DataFrame(trials_data)

        # Save the trial data to CSV
        csv_file_name = os.path.join(results_folder_opt_imu, 'trial_data.csv')
        df_trials.to_csv(csv_file_name, index=False)
        print(f"Trial data saved to: {csv_file_name}")

        # Set the trial number as index (optional)
        df_trials.set_index('number', inplace=True)

        # Plotting with 8 subplots for each orientation weight
        n_params = 8  # Assuming there are 8 orientation weights
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 6), sharex=True)

        # Plot each parameter in a separate subplot
        for i in range(n_params):
            param_key = f'orientation_weight_{i}'
            if param_key in df_trials.columns:
                axes[i].plot(df_trials.index, df_trials[param_key], label=param_key)
                axes[i].set_title(param_key)
                axes[i].set_ylabel('Value')
                axes[i].grid()
                axes[i].legend()

        axes[-1].set_xlabel('Trial Number')  # Label the x-axis for the last subplot
        plt.tight_layout()

        # Save the subplot figure
        plot_file_name = os.path.join(results_folder_opt_imu, 'parameter_evolution_subplots.png')
        plt.savefig(plot_file_name)

        # Run sensor fusion with best weights
        best_params = study.best_params
        orientation_weights = [best_params[f'orientation_weight_{i}'] for i in range(8)]
        webcam_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Webcam not taken into account
        stereocamera_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  #  Stereocamera not taken into account

        # Initialize Sensor Fusion for optimization
        fusion_opt = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
        output = run_sensor_fusion(fusion_opt, webcam_weights, orientation_weights, stereocamera_weights, constraint_var)
        # Save the results to mot file
        output_file_name = results_folder_opt_imu + "ik_results_subject_"+str(fusion_opt.subject_ID)+"_trial_"+str(fusion_opt.trial_ID)+"_best_params.mot"
        output.printResult(output, "ik_results_subject_"+str(fusion_opt.subject_ID)+"_trial_"+str(fusion_opt.trial_ID)+"_best_params", results_folder_opt_imu, -1, ".mot")
        print(f"Results saved to: {output_file_name}")

        output_header, output_columns, output_df = read_mot_file(output_file_name)

        # Downsample ground truth to match the time vector of the latest IK results
        time_vector = output_df['time'].values.tolist()
        ground_truth_df_downsampled = downsample(ground_truth_df, time_vector)

        # Compare with ground truth
        error_df = compare_joint_angles(output_df, ground_truth_df_downsampled)
        # delete the column where error is 0
        error_df = error_df.loc[:, (error_df != 0).any(axis=0)]
        error_df['time'] = time_vector

        # Evaluation metrics 
        # Range of Motion and difference in RoM between vicon and sensor fusion
        rom_vicon = ground_truth_df_downsampled.max() - ground_truth_df_downsampled.min()
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
        plt.savefig(results_folder_opt_imu + '/rom_error_best_weights.png')

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
        plt.savefig(results_folder_opt_imu + '/rmse_best_weights.png')

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
        plt.savefig(results_folder_opt_imu + '/mad_best_weights.png')

        # IMU and WEBCAM WEIGHTS OPTIMIZATION
        # Maske a folder to save optimization trials
        results_folder_opt_webcam = results_folder_opt + "imu_webcam/"
        if not os.path.exists(results_folder_opt_webcam):
            os.makedirs(results_folder_opt_webcam)

        # Initialize Sensor Fusion for optimization
        orientation_weights = [1.0] * 8
        webcam_weights = [1.0] * 12  # Webcam not taken into account
        stereocamera_weights = [0.0] * 15  #  Stereocamera not taken into account
        fusion_opt = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

        # Optimization setup
        study_name = "imu_webcam_weights_tuning_study_" + subject_ID + "_" + trial_ID + "_" + str(nb_trials)
        search_space = {}
        for i in range(8):
            search_space[f"orientation_weight_{i}"] = [0, 1]
        print("SEARCH_SPACE:", search_space)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space))
        #study = optuna.create_study(direction='minimize',storage="sqlite:///db.sqlite3", study_name=study_name, load_if_exists=True, sampler=TPESampler(multivariate=True))
        #study = optuna.create_study(direction='minimize', sampler=TPESampler(multivariate=True))
        # By default, Optuna uses Tree-structured Parzen Estimator algorithm implemented in TPESampler
        study.optimize(lambda trial: objective_imu_webcam(fusion_opt, trial, ground_truth_df, constraint_var, subject_ID, trial_ID), n_jobs=1)

        # Save best trial data in a file 
        file_name = "optuna_results_subject_" + subject_ID + "_trial_" + trial_ID + ".txt"
        file_name = os.path.join(results_folder_opt_webcam, file_name)
        with open(file_name, 'w') as f:
            f.write("Best trial parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write("\nBest trial value: " + str(study.best_value) + "\n")
            f.write("\nNumber of trials: " + str(len(study.trials)) + "\n")
        print(f"Results saved to {file_name}")

        # Extract parameters for plotting
        trials_data = []

        for trial in study.trials:
            trial_data = {"number": trial.number, "value": trial.value}
            # Assuming your parameters are named 'orientation_weight_0', 'orientation_weight_1', ..., 'orientation_weight_7'
            for key in sorted(trial.params.keys()):
                trial_data[key] = trial.params[key]
            trials_data.append(trial_data)

        # Create a DataFrame for easier plotting
        df_trials = pd.DataFrame(trials_data)

        
        # Save the trial data to CSV
        csv_file_name = os.path.join(results_folder_opt_webcam, 'trial_data.csv')
        df_trials.to_csv(csv_file_name, index=False)
        print(f"Trial data saved to: {csv_file_name}")

        # Set the trial number as index (optional)
        df_trials.set_index('number', inplace=True)

        # Plotting with 8 subplots for each orientation weight
        n_params = 8  # Assuming there are 8 orientation weights
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 6), sharex=True)

        # Plot each parameter in a separate subplot
        for i in range(n_params):
            param_key = f'orientation_weight_{i}'
            if param_key in df_trials.columns:
                axes[i].plot(df_trials.index, df_trials[param_key], label=param_key)
                axes[i].set_title(param_key)
                axes[i].set_ylabel('Value')
                axes[i].grid()
                axes[i].legend()

        axes[-1].set_xlabel('Trial Number')  # Label the x-axis for the last subplot
        plt.tight_layout()

        # Save the subplot figure
        plot_file_name = os.path.join(results_folder_opt_webcam, 'parameter_evolution_subplots.png')
        plt.savefig(plot_file_name)

        # Run sensor fusion with best weights
        best_params = study.best_params
        orientation_weights = [best_params[f'orientation_weight_{i}'] for i in range(8)]
        webcam_weights = [1.0] * 12  # webcam weights
        stereocamera_weights = [0.0] * 15  #  Stereocamera not taken into account
        fusion_opt = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

        output = run_sensor_fusion(fusion_opt, webcam_weights, orientation_weights, stereocamera_weights, constraint_var)
        # Save the results to mot file
        output_file_name = results_folder_opt_webcam + "ik_results_subject_"+str(fusion_opt.subject_ID)+"_trial_"+str(fusion_opt.trial_ID)+"_best_params.mot"
        output.printResult(output, "ik_results_subject_"+str(fusion_opt.subject_ID)+"_trial_"+str(fusion_opt.trial_ID)+"_best_params", results_folder_opt_webcam, -1, ".mot")
        print(f"Results saved to: {output_file_name}")

        output_header, output_columns, output_df = read_mot_file(output_file_name)

        # Downsample ground truth to match the time vector of the latest IK results
        time_vector = output_df['time'].values.tolist()
        ground_truth_df_downsampled = downsample(ground_truth_df, time_vector)

        # Compare with ground truth
        error_df = compare_joint_angles(output_df, ground_truth_df_downsampled)
        # delete the column where error is 0
        error_df = error_df.loc[:, (error_df != 0).any(axis=0)]
        error_df['time'] = time_vector

        # Evaluation metrics 
        # Range of Motion and difference in RoM between vicon and sensor fusion
        rom_vicon = ground_truth_df_downsampled.max() - ground_truth_df_downsampled.min()
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
        plt.savefig(results_folder_opt_webcam + '/rom_error_best_weights.png')

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
        plt.savefig(results_folder_opt_webcam + '/rmse_best_weights.png')

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
        plt.savefig(results_folder_opt_webcam + '/mad_best_weights.png')


        # IMU and STEREOCAMERA WEIGHTS OPTIMIZATION
        # Make a folder to save optimization trials
        results_folder_opt_stereo = results_folder_opt + "imu_stereocamera/"
        if not os.path.exists(results_folder_opt_stereo):
            os.makedirs(results_folder_opt_stereo)
        # Initialize Sensor Fusion for optimization
        orientation_weights = [1.0] * 8
        webcam_weights = [0.0] * 12  # Webcam not taken into account
        stereocamera_weights = [1.0] * 15  #  Stereocamera not taken into account
        fusion_opt = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

        # Optimization setup
        study_name = "imu_stereocamera_weights_tuning_study_" + subject_ID + "_" + trial_ID + "_" + str(nb_trials)
        #study = optuna.create_study(direction='minimize',storage="sqlite:///db.sqlite3", study_name=study_name, load_if_exists=True, sampler=TPESampler(multivariate=True))
        #study = optuna.create_study(direction='minimize', sampler=TPESampler(multivariate=True))
        search_space = {}
        for i in range(8):
            search_space[f"orientation_weight_{i}"] = [0, 1]
        print("SEARCH_SPACE:", search_space)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space))
        # By default, Optuna uses Tree-structured Parzen Estimator algorithm implemented in TPESampler
        study.optimize(lambda trial: objective_imu_stereocamera(fusion_opt, trial, ground_truth_df, constraint_var, subject_ID, trial_ID), n_jobs=1)


        # Save best trial data in a file 
        file_name = "optuna_results_subject_" + subject_ID + "_trial_" + trial_ID + ".txt"
        file_name = os.path.join(results_folder_opt_stereo, file_name)
        with open(file_name, 'w') as f:
            f.write("Best trial parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write("\nBest trial value: " + str(study.best_value) + "\n")
            f.write("\nNumber of trials: " + str(len(study.trials)) + "\n")
        print(f"Results saved to {file_name}")

        # Extract parameters for plotting
        trials_data = []

        for trial in study.trials:
            trial_data = {"number": trial.number, "value": trial.value}
            # Assuming your parameters are named 'orientation_weight_0', 'orientation_weight_1', ..., 'orientation_weight_7'
            for key in sorted(trial.params.keys()):
                trial_data[key] = trial.params[key]
            trials_data.append(trial_data)

        # Create a DataFrame for easier plotting
        df_trials = pd.DataFrame(trials_data)

        # Save the trial data to CSV
        csv_file_name = os.path.join(results_folder_opt_stereo, 'trial_data.csv')
        df_trials.to_csv(csv_file_name, index=False)
        print(f"Trial data saved to: {csv_file_name}")

        # Set the trial number as index (optional)
        df_trials.set_index('number', inplace=True)

        # Plotting with 8 subplots for each orientation weight
        n_params = 8  # Assuming there are 8 orientation weights
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 6), sharex=True)

        # Plot each parameter in a separate subplot
        for i in range(n_params):
            param_key = f'orientation_weight_{i}'
            if param_key in df_trials.columns:
                axes[i].plot(df_trials.index, df_trials[param_key], label=param_key)
                axes[i].set_title(param_key)
                axes[i].set_ylabel('Value')
                axes[i].grid()
                axes[i].legend()

        axes[-1].set_xlabel('Trial Number')  # Label the x-axis for the last subplot
        plt.tight_layout()

        # Save the subplot figure
        plot_file_name = os.path.join(results_folder_opt_stereo, 'parameter_evolution_subplots.png')
        plt.savefig(plot_file_name)
        # Run sensor fusion with best weights
        best_params = study.best_params
        orientation_weights = [best_params[f'orientation_weight_{i}'] for i in range(8)]
        webcam_weights = [0.0] * 12  # Webcam not taken into account
        stereocamera_weights = [1.0] * 15  #  Stereocamera weights taken into account

        fusion_opt = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
        output = run_sensor_fusion(fusion_opt, webcam_weights, orientation_weights, stereocamera_weights, constraint_var)
        # Save the results to mot file
        output_file_name = results_folder_opt_stereo + "ik_results_subject_"+str(fusion_opt.subject_ID)+"_trial_"+str(fusion_opt.trial_ID)+"_best_params.mot"
        output.printResult(output, "ik_results_subject_"+str(fusion_opt.subject_ID)+"_trial_"+str(fusion_opt.trial_ID)+"_best_params", results_folder_opt_stereo, -1, ".mot")
        print(f"Results saved to: {output_file_name}")

        output_header, output_columns, output_df = read_mot_file(output_file_name)

        # Downsample ground truth to match the time vector of the latest IK results
        time_vector = output_df['time'].values.tolist()
        ground_truth_df_downsampled = downsample(ground_truth_df, time_vector)

        # Compare with ground truth
        error_df = compare_joint_angles(output_df, ground_truth_df_downsampled)
        # delete the column where error is 0
        error_df = error_df.loc[:, (error_df != 0).any(axis=0)]
        error_df['time'] = time_vector

        # Evaluation metrics 
        # Range of Motion and difference in RoM between vicon and sensor fusion
        rom_vicon = ground_truth_df_downsampled.max() - ground_truth_df_downsampled.min()
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
        plt.savefig(results_folder_opt_stereo + '/rom_error_best_weights.png')

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
        plt.savefig(results_folder_opt_stereo + '/rmse_best_weights.png')

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
        plt.savefig(results_folder_opt_stereo + '/mad_best_weights.png')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subject data.")
    parser.add_argument("--subject_ID", type=str, required=True, help="Subject ID")
    parser.add_argument("--trial_ID", type=str, required=True, help="Trial ID (movement name)")
    parser.add_argument("--subject_mass", type=str, required=True, help="Subject mass (kg)")
    parser.add_argument("--subject_height", type=str, required=True, help="Subject height (mm)")
    parser.add_argument("--subject_age", type=str, required=True, help="Subject age (years)")
    parser.add_argument("--subject_sex", type=str, required=True, help="Subject sex (M/F)")
    parser.add_argument("--lag", type=int, required=True, help="Lag between IMU and vicon for this trial" )
    parser.add_argument("--optimization", type=bool, default=True, help="Enable optimization")
    args = parser.parse_args()

    main(args.subject_ID, args.trial_ID, args.subject_mass, args.subject_height, args.subject_age, args.subject_sex, args.lag, args.optimization)