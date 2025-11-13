# This script will ask for the subject ID and will fetch the corresponding TRC files and put them in the right folder
import numpy as np
import pandas as pd
import os
import shutil
import argparse
import glob
import re
import matplotlib.pyplot as plt

def read_imu_txt_file(file_path):
    start_timestamp = 0
    frequency = 0 

    with open(file_path, 'r') as f: 
        imu_data = f.read()
    
    # Regular expressions to extract start time and update rate
    start_time_pattern = r"// Start Time: ([0-9]+\.[0-9]+)"
    update_rate_pattern = r"// Update Rate: ([0-9]+H[z])"

    # Search for start time
    start_time_match = re.search(start_time_pattern, imu_data)
    if start_time_match:
        start_timestamp = float(start_time_match.group(1))

    # Search for update rate
    update_rate_match = re.search(update_rate_pattern, imu_data)
    if update_rate_match:
        frequency = int(update_rate_match.group(1).replace('Hz', ''))
    
    # Reading imu acceleration in x, y and z
    count = 0
    threshold = 6
    imu_accel = pd.DataFrame(columns=['Time', 'Acc_x', 'Acc_y', 'Acc_z'])
    with open(file_path, 'r') as f: 
        for line in f:
            if count >= threshold:
                data = line.split('\t')
                new_row = {'Time': int(data[0]), 'Acc_x': float(data[14]), 'Acc_y': float(data[15]), 'Acc_z': float(data[16])}
                imu_accel = pd.concat([imu_accel, pd.DataFrame([new_row])], ignore_index=True)
            count += 1

    # Changing the Time column to actual time in timestamps
    imu_accel['Time'] = imu_accel['Time'] / frequency + start_timestamp
    
    return imu_accel

def read_vicon_trc_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

        info_line = lines[2].strip().split('\t')
        frequency = float(info_line[0])

        datalines = lines[5:]
        
        # Columns
        columns = ['Time', 'Clap0_x', 'Clap0_y', 'Clap0_z', 'ClapX_x', 'ClapX_y', 'ClapX_z', 'ClapY_x', 'ClapY_y', 'ClapY_z']

        # Only clap data 
        for line in datalines:
            # slit by tab
            split_line = line.strip().split('\t')
            data.append([float(split_line[0]), float(split_line[70]), float(split_line[71]), float(split_line[72]), float(split_line[73]), float(split_line[74]), float(split_line[75]), float(split_line[76]), float(split_line[77]), float(split_line[78])])

        df = pd.DataFrame(data, columns=columns)

    # Changing the Time column to actual time in timestamps, but it will start at 0
   
    start_frame = df['Time'][0]
    df['Time'] = (df['Time'] - start_frame) / frequency   

    return df

def detect_spike(series, window_size=3, threshold=2):
    def relative_madness( x ):
        return abs( x[1] - np.median(x) ) - np.median( abs( x - np.median(x) ) )
    madness = series.rolling(window_size).apply(relative_madness, raw=True)
    spikes = abs(madness) > threshold
    return spikes

def find_clap(df, min_common):
    spike_indicators = pd.DataFrame(index=df.index)
    for col in df.columns:
        spike_indicators[col] = detect_spike(df[col])
    spike_count = spike_indicators.sum(axis=1)
    spike_count.plot()
    plt.show()
    common_spike_timestamps = spike_count[spike_count >= min_common].index
    if len(common_spike_timestamps.values) > 0:
        # Find the largest spike value(s)
        largest_spike_value = spike_count[common_spike_timestamps.values].max()  # Get the maximum spike count
        largest_spikes = spike_count[spike_count == largest_spike_value]
        
        # Calculate the median of these spikes if there are multiple
        median_spike = largest_spikes.median()

        return median_spike
    else:
        return 0

def main(subject_ID, trial_ID): 
    print("Looking for lag corresponding to trial ", trial_ID)
    # Initialization
    lag = 0

    # IMU data -> IMU 00B4A523
    imu_nb = "00B4A523"
    imu_file_name =  f'recordings/subject{subject_ID}/imu_{trial_ID}/{trial_ID}_{imu_nb}.txt'
    imu_accel_df = read_imu_txt_file(imu_file_name)

    # Vicon data -> clapO, clapX and clapY
    vicon_file_name = f'recordings/subject{subject_ID}/vicon_{trial_ID}.trc'
    vicon_clap_df = read_vicon_trc_file(vicon_file_name)

    # Find timestamp for clap
    imu_clap_time = find_clap(imu_accel_df, min_common=2)
    vicon_clap_time = find_clap(vicon_clap_df, min_common=3)

    lag = int(imu_clap_time - vicon_clap_time)
    #TODO !! NOT FINISHED, SOMETHING IS WRONG HERE 

    return -lag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs time synchronization by use of a cinematographic clap")
    parser.add_argument("subject_ID", type=str, help="Subject ID number")
    parser.add_argument("trial_ID", type=str, help="Trial description ID")
    
    args = parser.parse_args()

    main(args.subject_ID, args.trial_ID)
        
