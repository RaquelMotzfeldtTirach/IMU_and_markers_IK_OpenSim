import os
import shutil
import numpy as np
import pandas as pd
import optuna
from SensorFusion_automatic import main as sensor_fusion
from ViconIK import main as vicon_ik
import matplotlib.pyplot as plt
import statistics as stats

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
    vicon_df = vicon_df.drop(columns=['Frame#'])
    # Normalization, but not the time column # ToDo: look into standardisation instead
    vicon_df = (vicon_df - vicon_df.min()) / (vicon_df.max() - vicon_df.min())
    vicon_df['Time'] = imu_df['time'].iloc[0] + 0.01 * np.arange(len(vicon_df))  # assuming 100Hz

    # For debugging, print the first few rows
    print(vicon_df.head())
    print(imu_df.head())
    # Now select some pairs of columns to compare
    torso_imu_quat = imu_df['torso_imu'].tolist()
    pelvis_imu_quat = imu_df['pelvis_imu'].tolist()
    humerus_r_imu_quat = imu_df['humerus_r_imu'].tolist()
    humerus_l_imu_quat = imu_df['humerus_l_imu'].tolist()
    radius_r_imu_quat = imu_df['radius_r_imu'].tolist()
    radius_l_imu_quat = imu_df['radius_l_imu'].tolist()
    hand_r_imu_quat = imu_df['hand_r_imu'].tolist()
    hand_l_imu_quat = imu_df['hand_l_imu'].tolist()
    C7_vicon = vicon_df[['X1', 'Y1', 'Z1']].values.tolist()
    pelvis_vicon = vicon_df[['X23', 'Y23', 'Z23']].values.tolist()
    humerus_r_vicon = vicon_df[['X14', 'Y14', 'Z14']].values.tolist()
    humerus_l_vicon = vicon_df[['X7', 'Y7', 'Z7']].values.tolist()
    radius_r_vicon = vicon_df[['X16', 'Y16', 'Z16']].values.tolist()
    radius_l_vicon = vicon_df[['X9', 'Y9', 'Z9']].values.tolist()
    hand_r_vicon = vicon_df[['X19', 'Y19', 'Z19']].values.tolist()
    hand_l_vicon = vicon_df[['X12', 'Y12', 'Z12']].values.tolist()
    
    # Then find the time lag be using cross-correlation between the vicon data and the imu data
    nb_lags = len(vicon_df)
    lags = np.arange(-nb_lags + 1, nb_lags)
    best_lag_hum_l = 0
    best_corr_lag_hum_l = 0
    best_lag_hum_r = 0
    best_corr_lag_hum_r = 0
    best_lag_rad_l = 0
    best_corr_lag_rad_l = 0
    best_lag_rad_r = 0
    best_corr_lag_rad_r = 0
    best_lag_torso = 0
    best_corr_lag_torso = 0
    best_lag_hand_l = 0
    best_corr_lag_hand_l = 0
    best_lag_hand_r = 0
    best_corr_lag_hand_r = 0
    # ToDo: are we sure this cross-correlation is the same for vectors and for time series, look into the pearson correlation coefficient instead?
    for lag in lags:
        corr = 0
        for i in range(len(humerus_l_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(humerus_l_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = humerus_l_imu_quat[i] * (-1) + [1, 1, 1, 1]
            v = humerus_l_vicon[j]
            # Simple dot product as correlation measure
            corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
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
            # Simple dot product as correlation measure
            corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        if corr > best_corr_lag_hum_r:
            best_corr_lag_hum_r = corr
            best_lag_hum_r = lag
        corr = 0
        for i in range(len(radius_l_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(radius_l_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = radius_l_imu_quat[i] * (-1) + [1, 1, 1, 1]
            v = radius_l_vicon[j]
            # Simple dot product as correlation measure
            corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
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
            # Simple dot product as correlation measure
            corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        if corr > best_corr_lag_rad_r:
            best_corr_lag_rad_r = corr
            best_lag_rad_r = lag
        corr = 0
        for i in range(len(hand_l_imu_quat)):
            j = i + lag
            if j < 0 or j >= len(hand_l_vicon):
                continue
            # Convert quaternion to vector (just use the vector part)
            q = hand_l_imu_quat[i] * (-1) + [1, 1, 1, 1]
            v = hand_l_vicon[j]
            # Simple dot product as correlation measure
            corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
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
            # Simple dot product as correlation measure
            corr += abs(q[1]*v[0]) + abs(q[2]*v[1]) + abs(q[3]*v[2])
        if corr > best_corr_lag_hand_r:
            best_corr_lag_hand_r = corr
            best_lag_hand_r = lag

    print(f'Best lag hum left: {best_lag_hum_l} samples and best lag hum right: {best_lag_hum_r} samples')
    print(f'Best lag rad left: {best_lag_rad_l} samples and best lag rad right: {best_lag_rad_r} samples')
    print(f'Best lag hand left: {best_lag_hand_l} samples and best lag hand right: {best_lag_hand_r} samples')
    best_lag = stats.median([best_lag_hum_l, best_lag_hum_r, best_lag_rad_l, best_lag_rad_r, best_lag_hand_l, best_lag_hand_r])
    print(f'Overall best lag: {best_lag} samples')
    
    # Now correct the vicon time vector
    best_lag = -100.0  # determined by visual inspection
    time_shift = best_lag / 100.0  # assuming 100Hz
    vicon_df['Time'] = vicon_df['Time'] + time_shift
    # Plot to verify, normalized for better visualisation
    imu_plot = [q[1] for q in humerus_l_imu_quat]
    imu_plot = (imu_plot - np.min(imu_plot)) / (np.max(imu_plot) - np.min(imu_plot))
    vicon_plot = vicon_df['X7'].values
    vicon_plot = (vicon_plot - np.min(vicon_plot)) / (np.max(vicon_plot) - np.min(vicon_plot))
    vicon_plot = vicon_plot * (-1) + 1
    # Plot with subplots for left humerus, right humerus, left radius, right radius, left hand, right hand, torso
    plt.figure(figsize=(12, 10))    
    plt.subplot(4, 2, 1)
    plt.plot(imu_df['time'], imu_plot, label='IMU Humerus L X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Humerus L X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Left Humerus X Component')
    plt.legend()
    plt.subplot(4, 2, 2)
    imu_plot = [q[1] for q in humerus_r_imu_quat]
    imu_plot = (imu_plot - np.min(imu_plot)) / (np.max(imu_plot) - np.min(imu_plot))
    vicon_plot = vicon_df['X14'].values
    vicon_plot = (vicon_plot - np.min(vicon_plot)) / (np.max(vicon_plot) - np.min(vicon_plot))
    plt.plot(imu_df['time'], imu_plot, label='IMU Humerus R X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Humerus R X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Right Humerus X Component')
    plt.legend()
    plt.subplot(4, 2, 3)
    imu_plot = [q[1] for q in radius_l_imu_quat]
    imu_plot = (imu_plot - np.min(imu_plot)) / (np.max(imu_plot) - np.min(imu_plot))
    vicon_plot = vicon_df['X9'].values
    vicon_plot = (vicon_plot - np.min(vicon_plot)) / (np.max(vicon_plot) - np.min(vicon_plot))
    vicon_plot = vicon_plot * (-1) + 1
    plt.plot(imu_df['time'], imu_plot, label='IMU Radius L X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Radius L X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Left Radius X Component')
    plt.legend()
    plt.subplot(4, 2, 4)
    imu_plot = [q[1] for q in radius_r_imu_quat]
    imu_plot = (imu_plot - np.min(imu_plot)) / (np.max(imu_plot) - np.min(imu_plot))
    vicon_plot = vicon_df['X16'].values
    vicon_plot = (vicon_plot - np.min(vicon_plot)) / (np.max(vicon_plot) - np.min(vicon_plot))
    plt.plot(imu_df['time'], imu_plot, label='IMU Radius R X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Radius R X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Right Radius X Component')
    plt.legend()
    plt.subplot(4, 2, 5)
    imu_plot = [q[1] for q in hand_l_imu_quat]
    imu_plot = (imu_plot - np.min(imu_plot)) / (np.max(imu_plot) - np.min(imu_plot))
    vicon_plot = vicon_df['X12'].values
    vicon_plot = (vicon_plot - np.min(vicon_plot)) / (np.max(vicon_plot) - np.min(vicon_plot))
    vicon_plot = vicon_plot * (-1) + 1
    plt.plot(imu_df['time'], imu_plot, label='IMU Hand L X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Hand L X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')  
    plt.title('Left Hand X Component')
    plt.legend()
    plt.subplot(4, 2, 6)    
    imu_plot = [q[1] for q in hand_r_imu_quat]
    imu_plot = (imu_plot - np.min(imu_plot)) / (np.max(imu_plot) - np.min(imu_plot))
    vicon_plot = vicon_df['X19'].values
    vicon_plot = (vicon_plot - np.min(vicon_plot)) / (np.max(vicon_plot) - np.min(vicon_plot))
    plt.plot(imu_df['time'], imu_plot, label='IMU Hand R X', alpha=0.7)
    plt.plot(vicon_df['Time'], vicon_plot, label='Vicon Hand R X', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Right Hand X Component')
    plt.legend()
    plt.subplot(4, 2, 7)
    plt.tight_layout()
    plt.show()

    

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
        header = lines[:3]                 # first three lines as header
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
                if line.strip() == '':
                    continue
                row = [float(x) if x else np.nan for x in line.strip().split()]
                data.append(row)
    df = pd.DataFrame(data, columns=columns)
    return header, columns, df

def compare_joint_angles(df1, df2):
    return df1 - df2

def downsample(groundtruth_df, time_vector): #TODO: check that this is working
    """Downsamples the ground truth DataFrame to match the time vector."""
    downsampled_df = pd.DataFrame()
    # for each time, find the closest time in groundtruth_df and copy the row
    for t in time_vector:
        closest_time = groundtruth_df.iloc[(groundtruth_df['time'] - t).abs().argsort()[:1]]
        downsampled_df = pd.concat([downsampled_df, closest_time], ignore_index=True)
    downsampled_df = downsampled_df.drop(columns=['time']).reset_index(drop=True)
    downsampled_df.insert(0, 'time', time_vector)
    return downsampled_df

def objective(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
    # Define weight parameters to optimize
    webcam_weights = [trial.suggest_float(f'webcam_weight_{i}', 0.0, 10.0) for i in range(12)]  # Example: 12 webcam weights
    orientation_weights = [trial.suggest_float(f'orientation_weight_{i}', 0.0, 10.0) for i in range(8)]  # Example: 8 orientation weights
    stereocamera_weights = [trial.suggest_float(f'stereocamera_weight_{i}', 0.0, 10.0) for i in range(15)]  # Example: 15 stereocamera weights

    # Count weights that are non null
    non_null_weights = sum(1 for w in orientation_weights if w > 0)
    non_null_weights_ratio = 1 # TODO: adjust to the scale of the errors

    # Run IK with updated weights
    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    
    # Read latest output
    latest_ik_results_df = read_mot_file(output)

    # Downsample ground truth to match the time vector of the latest IK results
    time_vector = latest_ik_results_df['time'].values
    ground_truth_df = downsample(ground_truth_df, time_vector)

    # Compare with ground truth
    error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)

    return np.sum(error_df.abs().values) + non_null_weights * non_null_weights_ratio  # Example: minimize the sum of absolute errors and minimise the number of IMUs


def main():
    # Get inputs from user
    subject_ID = input("Enter the subject ID: ")
    trial_ID = input("Enter the trial ID (movement name): ")
    subject_mass = input("Enter the subject mass (kg): ")
    subject_height = input("Enter the subject height (mm): ")
    subject_age = input("Enter the subject age (years): ")
    subject_sex = input("Enter the subject sex (M/F): ")

    # Default weights
    # WEBCAM WEIGHTS: right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, right pinky, left pinky, right index, left index, right hip, left hip
    # ORIENTATION WEIGHTS: torso, pelvis, upper right, lower right, upper left, lower left, hand right, hand left
    # STEREOCAMERA WEIGHTS: neck, right clavicle, left clavicle, right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, spine 3, spine 2, spine 1, pelvis, right hip, left hip

    constraint_var = 1000

    ## THIS IS TO TEST THE SETUP BEFORE WEIGHT TUNING

    # Sensor fusion result with default weights
    webcam_weights = [1.0] * 12
    orientation_weights = [1.0] * 8
    stereocamera_weights = [1.0] * 15
    output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    # Read the output
    latest_ik_results_df = read_mot_file(output)

    # Time synchronise the vicon data
    time_correction(subject_ID, trial_ID)

    # Ground truth IK run - Vicon
    #ground_truth_ik_file = vicon_ik(subject_ID, trial_ID, constraint_var, subject_mass, subject_height, subject_age, subject_sex) 
    #ground_truth_df = read_mot_file(ground_truth_ik_file)

    # Downsample ground truth to match the time vector of the latest IK results
    #time_vector = latest_ik_results_df['time'].values
    #ground_truth_df = downsample(ground_truth_df, time_vector)

    # Compare with ground truth
    #error_df = compare_joint_angles(ground_truth_df, latest_ik_results_df)
    # For each joints plot the error over the timestamps
    #plt.figure()
    #for col in error_df.columns:
    #    if col != 'time':
    #        plt.plot(error_df['time'], error_df[col], label=col)
    #plt.xlabel('Time (s)')
    #plt.ylabel('Joint Angle Error (degrees)')
    #plt.title('Joint Angle Errors with Default Weights')
    #plt.legend()




    # Optimization setup
    #study = optuna.create_study(direction='minimize')
    #study.optimize(lambda trial: objective(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex), n_trials=100)

    # By default, Optuna uses Tree-structured Parzen Estimator algorithm implemented in TPESampler

if __name__ == "__main__":
    main()