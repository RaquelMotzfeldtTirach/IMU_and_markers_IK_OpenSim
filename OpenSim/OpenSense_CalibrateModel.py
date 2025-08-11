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

# Import OpenSim libraries
import opensim as osim
from math import pi
import argparse

def main(modelFileName, modelName, orientationsFileName, subject_ID, trial_ID):

    # Set variables to use
    sensor_to_opensim_rotations = osim.Vec3(-pi/2, 0, 0);# The rotation of IMU data to the OpenSim world frame # MAYBE -90 in degrees!
    baseIMUName = 'pelvis_imu';                     # The base IMU is the IMU on the base body of the model that dictates the heading (forward) direction of the model.
    baseIMUHeading = '-z';                           # The Coordinate Axis of the base IMU that points in the heading direction. 
    visulizeCalibration = True;                     # Boolean to Visualize the Output model

    # Instantiate an IMUPlacer object
    imuPlacer = osim.IMUPlacer();

    # Set properties for the IMUPlacer
    imuPlacer.set_model_file(modelFileName);
    imuPlacer.set_orientation_file_for_calibration(orientationsFileName);
    imuPlacer.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations);
    imuPlacer.set_base_imu_label(baseIMUName);
    imuPlacer.set_base_heading_axis(baseIMUHeading);

    # Run the IMUPlacer
    imuPlacer.run(visulizeCalibration);

    # Get the model with the calibrated IMU
    model = imuPlacer.getCalibratedModel();

    # Print the calibrated model to file.
    directory = modelFileName.rpartition('/')[0];
    savingFileName = directory +'/Calibrated_' + modelName + "_subject" + subject_ID + "_" + trial_ID +'.osim';
    print('Saving calibrated model to: ' + savingFileName);
    model.printToXML(savingFileName);
    return savingFileName

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenSense Model Calibration Script.")
    parser.add_argument("modelFileName", type=str, help="model file path")
    parser.add_argument("modelName", type=str, help="model name")
    parser.add_argument("orientationsFileName", type=str, help="orientation sto file path")
    parser.add_argument("subject_ID", type=str, help="subject ID")
    parser.add_argument("trial_ID", type=str, help="trial ID (movement name)")
    args = parser.parse_args()

    main(args.modelFileName, args.modelName, args.orientationsFileName)