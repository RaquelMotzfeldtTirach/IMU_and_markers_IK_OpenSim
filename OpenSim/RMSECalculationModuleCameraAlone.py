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
        # ---- WEBCAM-ONLY (no IMU orientations used for fusion) ----
        try:
            print("Running WEBCAM-ONLY sensor fusion and saving IK results...")
            webcam_only_folder = os.path.join(results_folder_rmse, "webcam_only")
            os.makedirs(webcam_only_folder, exist_ok=True)

            webcam_weights_run = [1.0] * 12
            orientation_weights_run = [0.0] * 8
            stereocamera_weights_run = [0.0] * 15

            fusion_run = sensor_fusion_initialization(webcam_weights_run, orientation_weights_run, stereocamera_weights_run,
                                                      constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

            output_storage = run_sensor_fusion(fusion_run, webcam_weights_run, orientation_weights_run, stereocamera_weights_run, constraint_var)
            out_name = f"ik_results_subject_{subject_ID}_trial_{trial_ID}_webcam_only.mot"
            output_storage.printResult(output_storage, out_name.replace('.mot', ''), webcam_only_folder, -1, ".mot")
            output_file = os.path.join(webcam_only_folder, out_name)

            # read output and use IMU timestamps to downsample / align ground truth
            out_header, out_columns, out_df = read_mot_file(output_file)
            imu_sto_path = f'recordings/subject{subject_ID}/imu_{trial_ID}/{trial_ID}_orientations_updatedTime.sto'
            if os.path.exists(imu_sto_path):
                _, _, imu_df = read_sto_file(imu_sto_path)
                imu_times = imu_df['time'].values
            else:
                # fallback to IK output times if IMU file not found
                imu_times = out_df['time'].values

            gt_down = downsample(ground_truth_df, imu_times)
            out_down = downsample(out_df, imu_times)

            error_df = compare_joint_angles(gt_down, out_down)
            rmse_series = np.sqrt((error_df.drop(columns=['time']) ** 2).sum(axis=0) / len(out_down))

            total_rmse = 0.0
            for j in joints_of_interest:
                if j in rmse_series.index:
                    total_rmse += float(rmse_series[j])
            total_rmse = float(total_rmse)/len(joints_of_interest)

            # save detailed rmse and summary
            rmse_file = os.path.join(webcam_only_folder, "rmse_per_joint.csv")
            rmse_series.to_frame(name='rmse').to_csv(rmse_file)
            summary_file = os.path.join(webcam_only_folder, "summary.csv")
            pd.DataFrame([{'mode': 'webcam_only', 'total_rmse': total_rmse}]).to_csv(summary_file, index=False)

            print(f"Saved webcam-only IK and RMSE to {webcam_only_folder}")

        except Exception as e:
            print(f"Webcam-only fusion failed: {e}")


        # ---- STEREOCAMERA-ONLY (no IMU orientations used for fusion) ----
        try:
            print("Running STEREOCAMERA-ONLY sensor fusion and saving IK results...")
            stereo_only_folder = os.path.join(results_folder_rmse, "stereocamera_only")
            os.makedirs(stereo_only_folder, exist_ok=True)

            webcam_weights_run = [0.0] * 12
            orientation_weights_run = [0.0] * 8
            stereocamera_weights_run = [1.0] * 15

            fusion_run = sensor_fusion_initialization(webcam_weights_run, orientation_weights_run, stereocamera_weights_run,
                                                      constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)

            output_storage = run_sensor_fusion(fusion_run, webcam_weights_run, orientation_weights_run, stereocamera_weights_run, constraint_var)
            out_name = f"ik_results_subject_{subject_ID}_trial_{trial_ID}_stereocamera_only.mot"
            output_storage.printResult(output_storage, out_name.replace('.mot', ''), stereo_only_folder, -1, ".mot")
            output_file = os.path.join(stereo_only_folder, out_name)

            # read output and use IMU timestamps to downsample / align ground truth
            out_header, out_columns, out_df = read_mot_file(output_file)
            imu_sto_path = f'recordings/subject{subject_ID}/imu_{trial_ID}/{trial_ID}_orientations_updatedTime.sto'
            if os.path.exists(imu_sto_path):
                _, _, imu_df = read_sto_file(imu_sto_path)
                imu_times = imu_df['time'].values
            else:
                imu_times = out_df['time'].values

            gt_down = downsample(ground_truth_df, imu_times)
            out_down = downsample(out_df, imu_times)

            error_df = compare_joint_angles(gt_down, out_down)
            rmse_series = np.sqrt((error_df.drop(columns=['time']) ** 2).sum(axis=0) / len(out_down))

            total_rmse = 0.0
            for j in joints_of_interest:
                if j in rmse_series.index:
                    total_rmse += float(rmse_series[j])
            total_rmse = float(total_rmse)/len(joints_of_interest)

            # save detailed rmse and summary
            rmse_file = os.path.join(stereo_only_folder, "rmse_per_joint.csv")
            rmse_series.to_frame(name='rmse').to_csv(rmse_file)
            summary_file = os.path.join(stereo_only_folder, "summary.csv")
            pd.DataFrame([{'mode': 'stereocamera_only', 'total_rmse': total_rmse}]).to_csv(summary_file, index=False)

            print(f"Saved stereocamera-only IK and RMSE to {stereo_only_folder}")

        except Exception as e:
            print(f"Stereocamera-only fusion failed: {e}")


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