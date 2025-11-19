# This script will ask for the subject ID and will fetch the corresponding TRC files and put them in the right folder
import numpy as np
import pandas as pd
import os
import shutil
import argparse
import glob
from WeightsTuningModule import main as weight_tuning_module
from TimeSynchronization import main as time_synchronization


def parse_subject_info(content):
    """Extract age, height, weight, and sex from subject_info.txt content."""
    info = {}
    for line in content.splitlines():
        if "Age:" in line:
            info['age'] = line.split(":")[1].strip()
        elif "Height" in line:
            # Remove non-digit characters except dot
            info['height'] = line.split(":")[1].strip()
        elif "Weight" in line:
            info['weight'] = line.split(":")[1].strip()
        elif "Sex:" in line:
            info['sex'] = line.split(":")[1].strip()
    return info

def main(ID): 
    if ID != '':
        subject_id = ID
    else:
        subject_id = input("Enter the subject ID (e.g., '29'): ")

    # Fetch the extra data about subject### (txt file)
    file_path ="/home/raquel/Documents/data/subject_"+subject_id+"/subject_info.txt"
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            info = parse_subject_info(content)
            print(info)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Find all the trials for that subject 
    data_path = "/home/raquel/Documents/sensor_fusion/recordings/subject"+subject_id+"/"
    imu_mtb_files = [f for f in os.listdir(data_path) if f.startswith('imu_') and f.endswith('.mtb')]
    trials = []
    for file in imu_mtb_files:
        file = file.removeprefix('imu_')
        file = file.removesuffix('.mtb')
        trials.append(file)
    print("Trials: ", trials)


    # Find time synchronization for each trial (not static for subject 92)
    lags = {}
    lags['static'] = 0  # No lag for static trial
    for trial in trials:
        if trial != 'static':
            lag = time_synchronization(subject_ID=subject_id, trial_ID=trial)
            lags[trial] = lag

    print("lags: ", lags)

    # Run optimization for each trial

    # TODO: should devide the IK in two parts: preparing everything -> only once before optuna, updating the weights and runing the IK -> many times with optuna and parallel computing! => TO BE TESTED!!!!
    # TODO: devide the jobs between only imu, with webcam and with stereocamera (and analytics should have their own folders)
    # TODO: save the raw data in a raw_data folder to always be able to retrieve it, untouchged 
    # TODO: store the results and show the metrics and weights, etc in a nice way! jupiter notebook maybe :) 
    # TODO: think about metrics and optimization equation: RMSE, MAE, R2, etc

    # Start with static trial since used for calibration
    weight_tuning_module(subject_ID=subject_id, trial_ID="static", subject_mass=info['weight'], subject_age=info['age'], subject_height=info['height'], subject_sex=info['sex'], lag=lags["static"], optimization=False)

    # Then run weight tuning for the rest of trials
    for trial in trials:
        if trial != "static":
            weight_tuning_module(subject_ID=subject_id, trial_ID=trial, subject_mass=info['weight'], subject_age=info['age'], subject_height=info['height'], subject_sex=info['sex'], lag=lags[trial])
        else: 
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the optimization for all trials in subject in question")
    parser.add_argument("--SUBJECT_ID", type=str, help="Subject ID number", default='')
    
    args = parser.parse_args()

    main(args.SUBJECT_ID)
        
