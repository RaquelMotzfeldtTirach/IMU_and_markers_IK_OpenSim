# This script will ask for the subject ID and will fetch the corresponding TRC files and put them in the right folder
import numpy as np
import pandas as pd
import os
import shutil
import argparse
import glob
from RMSECalculationModule import main as rmse_calculation_module
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


    # Find time synchronization for each trial
    lags = {}
    for trial in trials:
            lag = time_synchronization(subject_ID=subject_id, trial_ID=trial)
            lags[trial] = lag

    print("lags: ", lags)

    # Then run weight tuning for the rest of trials
    for trial in trials:
        if trial != "static":
            rmse_calculation_module(subject_ID=subject_id, trial_ID=trial, subject_mass=info['weight'], subject_age=info['age'], subject_height=info['height'], subject_sex=info['sex'], lag=lags[trial], rmse=True)

            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the optimization for all trials in subject in question")
    parser.add_argument("--SUBJECT_ID", type=str, help="Subject ID number", default='')
    
    args = parser.parse_args()

    main(args.SUBJECT_ID)
        
