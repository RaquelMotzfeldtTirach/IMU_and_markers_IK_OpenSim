# This script will ask for the subject ID and will fetch the corresponding TRC files and put them in the right folder
import numpy as np
import pandas as pd
import os
import shutil
import argparse

def get_data(ID): 
    if ID != '':
        subject_id = ID
    else:
        subject_id = input("Enter the subject ID (e.g., '29'): ")

    # Make folder in recordings if it doesn't exist
    subject_folder = f"recordings/subject{subject_id}"
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)
        print(f"Created folder: {subject_folder}")
    else:
        print(f"Folder already exists: {subject_folder}")

    # Make a raw_data folder inside subject_folder
    raw_data_folder = os.path.join(subject_folder, "raw_data")
    if not os.path.exists(raw_data_folder):
        os.makedirs(raw_data_folder)
        print(f"Created folder: {raw_data_folder}")
    else:
        print(f"Folder already exists: {raw_data_folder}")
    
    # Fetch the stereocamera trc files
    stereocamera_path = f"/home/raquel/Documents/ZED/body tracking/recordings/subject{subject_id}/" 
    stereocamera_trc_files = [f for f in os.listdir(stereocamera_path) if f.endswith('.trc')]
    if not stereocamera_trc_files:
        print(f"No TRC files found in {stereocamera_path}")
    else:
        for trc_file in stereocamera_trc_files:
            src = os.path.join(stereocamera_path, trc_file)
            dst = os.path.join(subject_folder, trc_file)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
                print(f"Copied {trc_file} to {subject_folder}")
            else:
                print(f"File already exists: {dst}")

    # Fetch the webcam trc files
    webcam_path = f"/home/raquel/Documents/mediapipe_test/recordings/subject{subject_id}/" 
    webcam_trc_files = [f for f in os.listdir(webcam_path) if f.endswith('.trc')]
    if not webcam_trc_files:
        print(f"No TRC files found in {webcam_path}")
    else:
        for trc_file in webcam_trc_files:
            src = os.path.join(webcam_path, trc_file)
            dst = os.path.join(subject_folder, trc_file)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
                print(f"Copied {trc_file} to {subject_folder}")
            else:
                print(f"File already exists: {dst}")

    # Fetch imu files
    imu_path = f"/home/raquel/Documents/Xsens/xda_python/recordings/subject{subject_id}/"
    # Copy contents of imu_path into subject_folder/ (no imu_data subfolder)
    if os.path.exists(imu_path):
        imu_dst = subject_folder  # copy directly into subject_folder (no imu_data subfolder)
        for root, dirs, files in os.walk(imu_path):
            rel = os.path.relpath(root, imu_path)
            dest_dir = imu_dst if rel == "." else os.path.join(imu_dst, rel)
            os.makedirs(dest_dir, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)
                # overwrite existing files; change to skip if you prefer
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} -> {dst_file}")
        print(f"Copied IMU contents from {imu_path} to {imu_dst}")
    else:
        print(f"IMU source folder doesn't exist: {imu_path}")

    # Copy subject content into raw_data folder
    for item in os.listdir(subject_folder):
        if item != "raw_data":
            s = os.path.join(subject_folder, item)
            d = os.path.join(raw_data_folder, item)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    shutil.copytree(s, d)
                    print(f"Copied directory {s} to {d}")
                else:
                    print(f"Directory already exists: {d}")
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d)
                    print(f"Copied file {s} to {d}")
                else:
                    print(f"File already exists: {d}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetches all the necessary data for subject in question")
    parser.add_argument("--SUBJECT_ID", type=str, help="Subject ID number", default='')
    
    args = parser.parse_args()

    get_data(args.SUBJECT_ID)
        
