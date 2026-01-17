import json
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
    "pro_sup_l", "wrist_flex_r", "wrist_flex_l"
    ]

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

    vicon_df_norm = (vicon_df_norm - vicon_df_norm.min()) / (vicon_df_norm.max() - vicon_df_norm.min())
    vicon_df_norm['Time'] = imu_df['time'].iloc[0] + 0.01 * np.arange(len(vicon_df_norm))  # assuming 100Hz

    imu_df = normalize_imu_df_minmax(imu_df, exclude_cols=['time'])

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


def rmse_imu_only(sensor_fusion, trial, ground_truth_df, constraint_var, subject_ID, trial_ID):
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


def main(subject_ID=None, trial_ID=None, subject_mass=None, subject_height=None, subject_age=None, subject_sex=None, lag=0, rmse=True):
    ## SENSOR FUSION: Default weights

    # WEBCAM WEIGHTS: right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, right pinky, left pinky, right index, left index, right hip, left hip
    # ORIENTATION WEIGHTS: torso, pelvis, upper right, lower right, lower left, upper left, hand right, hand left
    # STEREOCAMERA WEIGHTS: neck, right clavicle, left clavicle, right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, spine 3, spine 2, spine 1, pelvis, right hip, left hip

    constraint_var = 9000

    # Sensor fusion result with default weights
    webcam_weights = [1.0] * 12
    orientation_weights = [1.0] * 8
    stereocamera_weights = [1.0] * 15

    # Make a folder for the analytics 
    results_folder_rmse = f'analytics/subject_{subject_ID}/{trial_ID}/rmse/'
    if not os.path.exists(results_folder_rmse):
        os.makedirs(results_folder_rmse)

    sensor_count_data = {}
    sensor_count_data[trial_ID] = {}
    sensor_count_data[trial_ID]["imu_only"] = {}
    sensor_count_data[trial_ID]["imu_webcam"] = {}
    sensor_count_data[trial_ID]["imu_stereocamera"] = {}

    ## VICON
    
    # Make a folder for the analytics 
    results_file_gt = f'analytics/subject_{subject_ID}/{trial_ID}/ground_truth/inverse_kinematics_results_subject_{subject_ID}_trial_{trial_ID}.mot'
  
    # Read ground truth ik file
    ground_truth_ik_header, ground_truth_ik_columns, ground_truth_df = read_mot_file(results_file_gt)

    
    if rmse: 

        # IMU ONLY 
        
        # Initialize Sensor Fusion for rmse calculation
        orientation_weights = [1.0] * 8
        webcam_weights = [0.0] * 12  # Webcam not taken into account
        stereocamera_weights = [0.0] * 15  #  Stereocamera not taken into account
        fusion_opt = sensor_fusion_initialization(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
       
        # Maske a folder to save optimization trials
        results_folder_opt_imu = results_folder_rmse + "imu_only/"
        if not os.path.exists(results_folder_opt_imu):
            os.makedirs(results_folder_opt_imu)
        
        # Read trial data results
        trial_data_filepath = f'analytics/subject_{subject_ID}/{trial_ID}/optimization/grid_search/imu_only/trial_data.csv'
        df_trial_results = pd.read_csv(trial_data_filepath)
        
        sensor_count_data[trial_ID]["imu_only"] = {i: [] for i in range(1, 9)}

        # clean trial dataframe (drop rows with missing value)
        orientation_cols = [col for col in df_trial_results.columns if col.startswith('orientation_weight_')]
        df_trial_results = df_trial_results.copy()
        # ensure value numeric and drop NaNs
        df_trial_results['value'] = pd.to_numeric(df_trial_results['value'], errors='coerce')
        df_trial_results = df_trial_results.dropna(subset=['value'])

        best_rows = []
        for sensor_count in range(1, 9):
            # select rows where the number of enabled orientation sensors equals sensor_count
            mask = df_trial_results[orientation_cols].sum(axis=1).astype(int) == sensor_count
            subset = df_trial_results[mask]
            if subset.empty:
                # no trial with this sensor count
                continue
            # pick the row with minimum value
            best_idx = subset['value'].idxmin()
            best_row = subset.loc[best_idx]
            best_rows.append((sensor_count, best_row))

        # for each best row re-run sensor fusion (IMU only) and compute RMSE vs ground truth
        results_list = []
        for sensor_count, best_row in best_rows:
            # extract orientation weights as integers (0/1)
            best_orientation_weights = [int(best_row.get(f'orientation_weight_{i}', 0)) for i in range(len(orientation_cols))]

            # prepare fusion with the chosen weights (IMU only -> webcam and stereocamera zero)
            webcam_weights_run = [0.0] * 12
            stereocamera_weights_run = [0.0] * 15
            fusion_run = sensor_fusion_initialization(webcam_weights_run, best_orientation_weights, stereocamera_weights_run,
                                                      constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

            try:
                output_storage = run_sensor_fusion(fusion_run, webcam_weights_run, best_orientation_weights, stereocamera_weights_run, constraint_var)
                out_name = f"ik_results_subject_{subject_ID}_trial_{trial_ID}_imu_only_s{sensor_count}.mot"
                output_storage.printResult(output_storage, out_name.replace('.mot',''), results_folder_opt_imu, -1, ".mot")
                output_file = os.path.join(results_folder_opt_imu, out_name)

                # read output and compute RMSE
                out_header, out_columns, out_df = read_mot_file(output_file)
                time_vector = out_df['time'].values
                gt_down = downsample(ground_truth_df, time_vector)
                # ensure alignment of columns
                # compute error (ground_truth - output) to match prior approach
                error_df = compare_joint_angles(gt_down, out_df)
                # rmse per column (exclude time)
                rmse_series = np.sqrt((error_df.drop(columns=['time'])**2).sum(axis=0) / len(out_df))
                total_rmse = 0.0
                for j in joints_of_interest:
                    if j in rmse_series.index:
                        total_rmse += float(rmse_series[j])

                # save a CSV summary row
                row_result = {
                    'sensor_count': sensor_count,
                    'total_rmse': total_rmse
                }
                for i, w in enumerate(best_orientation_weights):
                    row_result[f'orientation_weight_{i}'] = w
                results_list.append(row_result)

                # optional: save detailed rmse per joint
                rmse_file = os.path.join(results_folder_opt_imu, f"rmse_per_joint_s{sensor_count}.csv")
                rmse_series.to_frame(name='rmse').to_csv(rmse_file)

                # remove output to save space (keep the saved RMSE files)
                try:
                    os.remove(output_file)
                except OSError:
                    pass

            except Exception as e:
                print(f"Failed to run fusion for sensor_count {sensor_count}: {e}")
                continue

        # write summary CSV
        if results_list:
            df_results_summary = pd.DataFrame(results_list)
            summary_file = os.path.join(results_folder_opt_imu, "best_weights_and_rmse_by_sensor_count.csv")
            df_results_summary.to_csv(summary_file, index=False)
            print(f"Saved IMU-only best weights and RMSE summary to {summary_file}")

        # store in sensor_count_data structure for downstream use/inspection
        sensor_count_data[trial_ID]["imu_only"] = {int(r['sensor_count']): r for r in results_list}
 

        # IMU and WEBCAM
        
        # Make a folder to save optimization trials for imu+webcam
        results_folder_opt_imu_webcam = results_folder_rmse + "imu_webcam/"
        if not os.path.exists(results_folder_opt_imu_webcam):
            os.makedirs(results_folder_opt_imu_webcam)

        # Attempt to read trial data for imu_webcam grid search
        trial_data_filepath_webcam = f'analytics/subject_{subject_ID}/{trial_ID}/optimization/grid_search/imu_webcam/trial_data.csv'
        if os.path.exists(trial_data_filepath_webcam):
            df_trial_results_webcam = pd.read_csv(trial_data_filepath_webcam)

            # prepare storage
            sensor_count_data[trial_ID]["imu_webcam"] = {i: [] for i in range(1, 9)}

            # clean trial dataframe (drop rows with missing value)
            orientation_cols_webcam = [col for col in df_trial_results_webcam.columns if col.startswith('orientation_weight_')]
            df_trial_results_webcam = df_trial_results_webcam.copy()
            df_trial_results_webcam['value'] = pd.to_numeric(df_trial_results_webcam['value'], errors='coerce')
            df_trial_results_webcam = df_trial_results_webcam.dropna(subset=['value'])

            best_rows_webcam = []
            for sensor_count in range(1, 9):
                mask = df_trial_results_webcam[orientation_cols_webcam].sum(axis=1).astype(int) == sensor_count
                subset = df_trial_results_webcam[mask]
                if subset.empty:
                    continue
                best_idx = subset['value'].idxmin()
                best_row = subset.loc[best_idx]
                best_rows_webcam.append((sensor_count, best_row))

            results_list_webcam = []
            for sensor_count, best_row in best_rows_webcam:
                # extract orientation weights (0/1)
                best_orientation_weights = [int(best_row.get(f'orientation_weight_{i}', 0)) for i in range(len(orientation_cols_webcam))]

                # IMU+WEBCAM: enable webcam weights, stereocamera off
                webcam_weights_run = [1.0] * 12
                stereocamera_weights_run = [0.0] * 15
                fusion_run = sensor_fusion_initialization(webcam_weights_run, best_orientation_weights, stereocamera_weights_run,
                                                          constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

                try:
                    output_storage = run_sensor_fusion(fusion_run, webcam_weights_run, best_orientation_weights, stereocamera_weights_run, constraint_var)
                    out_name = f"ik_results_subject_{subject_ID}_trial_{trial_ID}_imu_webcam_s{sensor_count}.mot"
                    output_storage.printResult(output_storage, out_name.replace('.mot',''), results_folder_opt_imu_webcam, -1, ".mot")
                    output_file = os.path.join(results_folder_opt_imu_webcam, out_name)

                    # read output and compute RMSE
                    out_header, out_columns, out_df = read_mot_file(output_file)
                    time_vector = out_df['time'].values
                    gt_down = downsample(ground_truth_df, time_vector)
                    error_df = compare_joint_angles(gt_down, out_df)
                    rmse_series = np.sqrt((error_df.drop(columns=['time'])**2).sum(axis=0) / len(out_df))
                    total_rmse = 0.0
                    for j in joints_of_interest:
                        if j in rmse_series.index:
                            total_rmse += float(rmse_series[j])

                    # save a CSV summary row
                    row_result = {
                        'sensor_count': sensor_count,
                        'total_rmse': total_rmse
                    }
                    for i, w in enumerate(best_orientation_weights):
                        row_result[f'orientation_weight_{i}'] = w
                    results_list_webcam.append(row_result)

                    # save detailed rmse per joint
                    rmse_file = os.path.join(results_folder_opt_imu_webcam, f"rmse_per_joint_s{sensor_count}.csv")
                    rmse_series.to_frame(name='rmse').to_csv(rmse_file)

                    # remove output to save space
                    try:
                        os.remove(output_file)
                    except OSError:
                        pass

                except Exception as e:
                    print(f"Failed to run IMU+WEBCAM fusion for sensor_count {sensor_count}: {e}")
                    continue

            # write summary CSV
            if results_list_webcam:
                df_results_summary_webcam = pd.DataFrame(results_list_webcam)
                summary_file_webcam = os.path.join(results_folder_opt_imu_webcam, "best_weights_and_rmse_by_sensor_count.csv")
                df_results_summary_webcam.to_csv(summary_file_webcam, index=False)
                print(f"Saved IMU+WEBCAM best weights and RMSE summary to {summary_file_webcam}")

            # store in sensor_count_data
            sensor_count_data[trial_ID]["imu_webcam"] = {int(r['sensor_count']): r for r in results_list_webcam}

        else:
            print(f"No grid search trial_data found for IMU+WEBCAM at {trial_data_filepath_webcam}")




        # IMU and STEREOCAMERA
        
        # Make a folder to save optimization trials for imu+stereocamera
        results_folder_opt_imu_stereo = results_folder_rmse + "imu_stereocamera/"
        if not os.path.exists(results_folder_opt_imu_stereo):
            os.makedirs(results_folder_opt_imu_stereo)

        # Attempt to read trial data for imu_stereocamera grid search
        trial_data_filepath_stereo = f'analytics/subject_{subject_ID}/{trial_ID}/optimization/grid_search/imu_stereocamera/trial_data.csv'
        # extra safeguard: if file missing, skip quietly to next trial
        if not os.path.exists(trial_data_filepath_stereo):
            print(f"No grid search trial_data found for IMU+STEREOCAMERA at {trial_data_filepath_stereo}, skipping.")
        else:
            df_trial_results_stereo = pd.read_csv(trial_data_filepath_stereo)

            # prepare storage
            sensor_count_data[trial_ID]["imu_stereocamera"] = {i: [] for i in range(1, 9)}

            # clean trial dataframe (drop rows with missing value)
            orientation_cols_stereo = [col for col in df_trial_results_stereo.columns if col.startswith('orientation_weight_')]
            df_trial_results_stereo = df_trial_results_stereo.copy()
            df_trial_results_stereo['value'] = pd.to_numeric(df_trial_results_stereo['value'], errors='coerce')
            df_trial_results_stereo = df_trial_results_stereo.dropna(subset=['value'])

            best_rows_stereo = []
            for sensor_count in range(1, 9):
                mask = df_trial_results_stereo[orientation_cols_stereo].sum(axis=1).astype(int) == sensor_count
                subset = df_trial_results_stereo[mask]
                if subset.empty:
                    continue
                best_idx = subset['value'].idxmin()
                best_row = subset.loc[best_idx]
                best_rows_stereo.append((sensor_count, best_row))

            results_list_stereo = []
            for sensor_count, best_row in best_rows_stereo:
                # extract orientation weights (0/1)
                best_orientation_weights = [int(best_row.get(f'orientation_weight_{i}', 0)) for i in range(len(orientation_cols_stereo))]

                # IMU+STEREO: enable stereocamera weights, webcam off
                webcam_weights_run = [0.0] * 12
                stereocamera_weights_run = [1.0] * 15
                fusion_run = sensor_fusion_initialization(webcam_weights_run, best_orientation_weights, stereocamera_weights_run,
                                                          constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

                try:
                    output_storage = run_sensor_fusion(fusion_run, webcam_weights_run, best_orientation_weights, stereocamera_weights_run, constraint_var)
                    out_name = f"ik_results_subject_{subject_ID}_trial_{trial_ID}_imu_stereocamera_s{sensor_count}.mot"
                    output_storage.printResult(output_storage, out_name.replace('.mot',''), results_folder_opt_imu_stereo, -1, ".mot")
                    output_file = os.path.join(results_folder_opt_imu_stereo, out_name)

                    # read output and compute RMSE
                    out_header, out_columns, out_df = read_mot_file(output_file)
                    time_vector = out_df['time'].values
                    gt_down = downsample(ground_truth_df, time_vector)
                    error_df = compare_joint_angles(gt_down, out_df)
                    rmse_series = np.sqrt((error_df.drop(columns=['time'])**2).sum(axis=0) / len(out_df))
                    total_rmse = 0.0
                    for j in joints_of_interest:
                        if j in rmse_series.index:
                            total_rmse += float(rmse_series[j])

                    # save a CSV summary row
                    row_result = {
                        'sensor_count': sensor_count,
                        'total_rmse': total_rmse
                    }
                    for i, w in enumerate(best_orientation_weights):
                        row_result[f'orientation_weight_{i}'] = w
                    results_list_stereo.append(row_result)

                    # save detailed rmse per joint
                    rmse_file = os.path.join(results_folder_opt_imu_stereo, f"rmse_per_joint_s{sensor_count}.csv")
                    rmse_series.to_frame(name='rmse').to_csv(rmse_file)

                    # remove output to save space
                    try:
                        os.remove(output_file)
                    except OSError:
                        pass

                except Exception as e:
                    print(f"Failed to run IMU+STEREOCAMERA fusion for sensor_count {sensor_count}: {e}")
                    continue

            # write summary CSV
            if results_list_stereo:
                df_results_summary_stereo = pd.DataFrame(results_list_stereo)
                summary_file_stereo = os.path.join(results_folder_opt_imu_stereo, "best_weights_and_rmse_by_sensor_count.csv")
                df_results_summary_stereo.to_csv(summary_file_stereo, index=False)
                print(f"Saved IMU+STEREOCAMERA best weights and RMSE summary to {summary_file_stereo}")

            # store in sensor_count_data
            sensor_count_data[trial_ID]["imu_stereocamera"] = {int(r['sensor_count']): r for r in results_list_stereo}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subject data.")
    parser.add_argument("--subject_ID", type=str, required=True, help="Subject ID")
    parser.add_argument("--trial_ID", type=str, required=True, help="Trial ID (movement name)")
    parser.add_argument("--subject_mass", type=str, required=True, help="Subject mass (kg)")
    parser.add_argument("--subject_height", type=str, required=True, help="Subject height (mm)")
    parser.add_argument("--subject_age", type=str, required=True, help="Subject age (years)")
    parser.add_argument("--subject_sex", type=str, required=True, help="Subject sex (M/F)")
    parser.add_argument("--lag", type=int, required=True, help="Lag between IMU and vicon for this trial" )
    parser.add_argument("--rmse", type=bool, default=True, help="Enable RMSE calculation")
    args = parser.parse_args()

    main(args.subject_ID, args.trial_ID, args.subject_mass, args.subject_height, args.subject_age, args.subject_sex, args.lag, args.rmse)