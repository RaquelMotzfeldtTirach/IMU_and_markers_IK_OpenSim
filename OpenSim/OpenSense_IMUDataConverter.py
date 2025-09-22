# ----------------------------------------------------------------------- #
# The OpenSim API is a toolkit for musculoskeletal modeling and           #
# simulation. See http://opensim.stanford.edu and the NOTICE file         #
# for more information. OpenSim is developed at Stanford University       #
# and supported by the US National Institutes of Health (U54 GM072970,    #
# R24 HD065690) and by DARPA through the Warrior Web program.             #
#                                                                         #
# Copyright (c) 2005-2019 Stanford University and the Authors             #
# Author(s): James Dunne                                                  #
#                                                                         #
# Licensed under the Apache License, Version 2.0 (the "License")         #
# you may not use this file except in compliance with the License.        #
# You may obtain a copy of the License at                                 #
# http://www.apache.org/licenses/LICENSE-2.0.                             #
#                                                                         #
# Unless required by applicable law or agreed to in writing, software     #
# distributed under the License is distributed on an "AS IS" BASIS,       #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or         #
# implied. See the License for the specific language governing            #
# permissions and limitations under the License.                          #
# ----------------------------------------------------------------------- #

# Example code for reading, and converting, XSENS IMU sensor data to
# OpenSense friendly format.
# Run this script from the OpenSenseExampleFiles directory. 

# Import the opensim libraries
import opensim as osim
import pandas as pd
import os 
import random
import xml.etree.ElementTree as ET
import argparse

# Build an Xsens Settings Object. 
# Instantiate the Reader Settings Class

def main(subject_ID, trial_ID, defaultMapping = False):
    if not defaultMapping:
        usualMapping = input('Are you using the usual IMU Mappings file? (y/n): ')
        if usualMapping.lower() == 'y':
            mappingFile = "OpenSim/defaultIMUMappings.xml"
        else:
            mappingFile = input('Please enter the IMU Mappings file path (e.g. IMUMappings.xml): ')
    else:
        mappingFile = "OpenSim/defaultIMUMappings.xml"

    # edit IMU Mappings file to set the trial prefix
    tree = ET.parse(mappingFile)
    root = tree.getroot()
    for tag in root.iter('XsensDataReaderSettings'):
        tag.find('trial_prefix').text = trial_ID

    new_file_path = 'recordings/subject'+ subject_ID + '/imu_' + trial_ID + '/IMUMapping.xml'
    tree.write(new_file_path, encoding='utf-8', xml_declaration=True)

    xsensSettings = osim.XsensDataReaderSettings(new_file_path)
  
    # Instantiate an XsensDataReader
    xsens = osim.XsensDataReader(xsensSettings)
 
    # Read in seprate tables of data from the specified IMU file(s)
    # Ask the user for the IMU Data files
    dataFiles = 'recordings/subject'+ subject_ID + '/imu_' + trial_ID + '/'
    tables = xsens.read(dataFiles)
    # get the trial name from the settings
    trial = xsensSettings.get_trial_prefix()
    # Get Orientation Data as quaternions
    quatTable = xsens.getOrientationsTable(tables)
    # Write to file
    osim.STOFileAdapterQuaternion.write(quatTable, dataFiles + trial + '_orientations.sto')
    # Get Acceleration Data
    accelTable = xsens.getLinearAccelerationsTable(tables)
    # Write to file
    osim.STOFileAdapterVec3.write(accelTable, dataFiles + trial + '_linearAccelerations.sto')
    # Get Magnetic (North) Heading Data
    magTable = xsens.getMagneticHeadingTable(tables)
    # Write to file
    osim.STOFileAdapterVec3.write(magTable, dataFiles + trial + '_magneticNorthHeadings.sto')
    # Get Angular Velocity Data
    angVelTable = xsens.getAngularVelocityTable(tables)
    # Write to file
    osim.STOFileAdapterVec3.write(angVelTable, dataFiles + trial + '_angularVelocities.sto')

    # Updating time to actual time 
    print('Updating the time in the orientation file using the start time timestamp')
    inputFile = dataFiles + trial + '_orientations.sto'
    outputFile = dataFiles + trial + '_orientations_updatedTime.sto'

    # Get the start time from the header of one of the files
    for file in os.listdir(dataFiles):
        if file.endswith('.txt'):
            randomFile = file
            break

    firstLine = pd.read_csv(dataFiles + randomFile, nrows=0, sep='\s+')
    startTime = firstLine.columns[3]
    startTime = float(startTime)

    # Use the start time to update the time in the orientation file
    header = pd.read_csv(inputFile, nrows=4)
    data = pd.read_csv(inputFile, sep='\s+', skiprows=5)
    data['time'] += startTime
    with open(outputFile, 'w') as f:
            # Write the header back
            f.write(header.to_csv(sep="\t", index=False, header=False))
            # Write the modified data
            data.to_csv(f, sep="\t", index=False)
    return outputFile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenSense IMU Data Converter.")
    parser.add_argument("subject_ID", type=str, help="Subject ID")
    parser.add_argument("trial_ID", type=str, help="Trial name")
    args = parser.parse_args()

    main(args.subject_ID, args.trial_ID)