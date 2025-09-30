import os
import shutil
import numpy as np
import pandas as pd
import optuna
from SensorFusion_automatic import main as sensor_fusion
from ViconIK import main as vicon_ik

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
                row = [float(x) for x in line.strip().split()]
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

    # Ground truth IK run - Vicon
    ground_truth_ik_file = vicon_ik(subject_ID, trial_ID, constraint_var, subject_mass, subject_height, subject_age, subject_sex) #TODO 
    ground_truth_df = read_mot_file(ground_truth_ik_file)
    
    # Optimization setup
    #study = optuna.create_study(direction='minimize')
    #study.optimize(lambda trial: objective(trial, ground_truth_df, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex), n_trials=100)

    # By default, Optuna uses Tree-structured Parzen Estimator algorithm implemented in TPESampler

if __name__ == "__main__":
    main()