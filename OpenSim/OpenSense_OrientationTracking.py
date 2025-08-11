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
# Licensed under the Apache License, Version 2.0 (the "License");         #
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

# Example code to perform orienation tracking with OpenSense. This
# script uses the OpenSense library functions and is part of the OpenSense
# Example files. 

import opensim as osim
from opensim import Vec3
import numpy as np
import math
from math import pi
import argparse
import os
import time

def main(modelFileName, orientationsFileName):
    print(modelFileName)
    print(orientationsFileName)
    # Set variables to use
    sensor_to_opensim_rotation = osim.Vec3(-pi/2, 0, 0); # The rotation of IMU data to the OpenSim world frame
    visualizeTracking = True;  # Boolean to Visualize the tracking simulation
    resultsDirectory = 'IKResultsTest';

    # Instantiate an InverseKinematicsTool
    #imuIK = osim.IMUInverseKinematicsTool();
    model = osim.Model(modelFileName);  # Load the model to ensure it is valid before running the IK tool
    s = model.initSystem();  # This is crucial for the model to work properly
    coordinates = model.getCoordinateSet();
    imuPlacer = osim.IMUPlacer();
    quatTable = osim.TimeSeriesTableQuaternion(orientationsFileName);
    
    # Compute rotation matrix so that (e.g. "pelvis_imu" + SimTK::ZAxis) lines up with model forward (+X)
    base_imu_label = "pelvis_imu"  # Replace with your base IMU label
    direction_on_imu = osim.CoordinateDirection(osim.CoordinateAxis(2), -1)  # Negative Z-axis direction

    # Apply sensor to OpenSim rotation first (matching C++ IMUPlacer implementation)
    sensor_to_opensim_rotation = osim.Vec3(-pi/2, 0, 0); # The rotation of IMU data to the OpenSim world frame
    sensorToOpenSim = osim.Rotation()
    # Apply the sensor rotation sequence: X, Y, Z rotations
    rotX = osim.Rotation()
    rotX.setRotationFromAngleAboutAxis(sensor_to_opensim_rotation[0], osim.CoordinateAxis(0))
    rotY = osim.Rotation() 
    rotY.setRotationFromAngleAboutAxis(sensor_to_opensim_rotation[1], osim.CoordinateAxis(1))
    rotZ = osim.Rotation()
    rotZ.setRotationFromAngleAboutAxis(sensor_to_opensim_rotation[2], osim.CoordinateAxis(2))
    
    # Apply sensor rotations to quatTable first
    osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotX)
    osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotY)
    osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotZ)
    print(f"Applied sensor to OpenSim rotations: {sensor_to_opensim_rotation[0]*180/pi:.1f}, {sensor_to_opensim_rotation[1]*180/pi:.1f}, {sensor_to_opensim_rotation[2]*180/pi:.1f} degrees")

    # Compute heading correction using OpenSense utilities (matching C++ implementation)
    # Check if heading correction should be performed
    base_heading_axis = "-z"  # Equivalent to base_heading_axis property in C++
    
    # Parse heading axis specification (matching C++ logic from lines 104-118)
    imu_axis = base_heading_axis.lower()
    direction = 1
    if imu_axis.startswith('-'):
        direction = -1
    
    axis_char = imu_axis[-1]  # Get last character (x, y, or z)
    if axis_char == 'x':
        direction_on_imu = osim.CoordinateDirection(osim.CoordinateAxis(0), direction)  # XAxis
    elif axis_char == 'y':
        direction_on_imu = osim.CoordinateDirection(osim.CoordinateAxis(1), direction)  # YAxis
    elif axis_char == 'z':
        direction_on_imu = osim.CoordinateDirection(osim.CoordinateAxis(2), direction)  # ZAxis
    else:
        raise ValueError(f"Invalid heading axis specification: {base_heading_axis}")
    
    print(f"Using heading axis: {base_heading_axis} -> CoordinateDirection({axis_char.upper()}Axis, {direction})")

    heading_rotation_vec3 = osim.OpenSenseUtilities.computeHeadingCorrection(
        model, s, quatTable, base_imu_label, direction_on_imu)
    
    # Create rotation using SpaceRotationSequence (matching C++ lines 123-126)
    heading_rotation = osim.Rotation()
    # Apply all three rotation components as in C++ implementation
    heading_rotation.setRotationFromAngleAboutAxis(heading_rotation_vec3.get(0), osim.CoordinateAxis(0))
    temp_rotation_y = osim.Rotation()
    temp_rotation_y.setRotationFromAngleAboutAxis(heading_rotation_vec3.get(1), osim.CoordinateAxis(1))
    temp_rotation_z = osim.Rotation()
    temp_rotation_z.setRotationFromAngleAboutAxis(heading_rotation_vec3.get(2), osim.CoordinateAxis(2))
    
    # Manually compose rotations (X * Y * Z sequence)
    # Since we can't multiply directly, we'll apply them in sequence to the quaternion table
    # But apply reduced corrections to preserve legitimate motion
    
    # Calculate the magnitude of correction needed
    correction_magnitude = math.sqrt(heading_rotation_vec3.get(0)**2 + 
                                   heading_rotation_vec3.get(1)**2 + 
                                   heading_rotation_vec3.get(2)**2)
    
    # If correction is too large (> 0.3 radians ~17 degrees), apply only partial correction
    max_correction = 0.3  # Maximum correction in radians (~17 degrees)
    if correction_magnitude > max_correction:
        scale_factor = max_correction / correction_magnitude
        print(f"Large heading correction detected ({correction_magnitude*180/pi:.1f}°), applying scaled correction ({scale_factor:.2f})")
        
        # Scale down each component
        scaled_heading_rotation = osim.Rotation()
        scaled_heading_rotation.setRotationFromAngleAboutAxis(
            heading_rotation_vec3.get(0) * scale_factor, osim.CoordinateAxis(0))
        scaled_temp_rotation_y = osim.Rotation()
        scaled_temp_rotation_y.setRotationFromAngleAboutAxis(
            heading_rotation_vec3.get(1) * scale_factor, osim.CoordinateAxis(1))
        scaled_temp_rotation_z = osim.Rotation()
        scaled_temp_rotation_z.setRotationFromAngleAboutAxis(
            heading_rotation_vec3.get(2) * scale_factor, osim.CoordinateAxis(2))
        
        # Apply scaled corrections
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, scaled_heading_rotation)  # X rotation
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, scaled_temp_rotation_y)  # Y rotation  
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, scaled_temp_rotation_z)  # Z rotation
        
        print(f"Applied scaled heading correction: X={heading_rotation_vec3.get(0)*scale_factor*180/pi:.1f}°, "
              f"Y={heading_rotation_vec3.get(1)*scale_factor*180/pi:.1f}°, "
              f"Z={heading_rotation_vec3.get(2)*scale_factor*180/pi:.1f}°")
    else:
        # Apply full correction for small corrections
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, heading_rotation)  # X rotation
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, temp_rotation_y)  # Y rotation  
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, temp_rotation_z)  # Z rotation
        print(f"Applied full heading correction: X={heading_rotation_vec3.get(0)*180/pi:.1f}°, "
              f"Y={heading_rotation_vec3.get(1)*180/pi:.1f}°, "
              f"Z={heading_rotation_vec3.get(2)*180/pi:.1f}°")
    orientationData = osim.OpenSenseUtilities.convertQuaternionsToRotations(quatTable);
    oRefs = osim.OrientationsReference(orientationData);
    mRefs = osim.MarkersReference();
    coordinateReferences = osim.SimTKArrayCoordinateReference();
    
    # Don't add pelvis constraints - let the IMU data drive the pelvis motion naturally
    # The heading correction should handle coordinate system alignment
    print("Using IMU-driven pelvis motion without artificial constraints")
    
    constraint_var = .001  # Increased constraint weight for better stability
    print('Argument types: ', type(model), type(oRefs), type(mRefs), type(coordinateReferences), type(constraint_var))
    ikSolver = osim.InverseKinematicsSolver(model, mRefs, oRefs, coordinateReferences, constraint_var);
    # Set tool properties
    #imuIK.set_model_file(modelFileName);
    #imuIK.set_orientations_file(orientationsFileName);
    #imuIK.set_sensor_to_opensim_rotations(sensor_to_opensim_rotation)

    directory = orientationsFileName.rpartition('/')[0];
    resultsDirectory = directory + '/' + resultsDirectory;
    
    # Create results directory if it doesn't exist
    os.makedirs(resultsDirectory, exist_ok=True);
    
    # Set solver accuracy to match C++ implementation (line 234 in IMUPlacer.cpp)
    accuracy = 1e-4  # Same as C++ implementation
    ikSolver.setAccuracy(accuracy);
    
    # Print heading correction information
    print(f"Computed heading correction: X={heading_rotation_vec3.get(0):.4f}, Y={heading_rotation_vec3.get(1):.4f}, Z={heading_rotation_vec3.get(2):.4f} radians")
    print(f"Heading correction in degrees: X={heading_rotation_vec3.get(0) * 180.0 / pi:.2f}, Y={heading_rotation_vec3.get(1) * 180.0 / pi:.2f}, Z={heading_rotation_vec3.get(2) * 180.0 / pi:.2f}")
    
    # Set solver properties for better stability
    try:
        ikSolver.setConstraintWeight(constraint_var)
        print(f"Set constraint weight to: {constraint_var}")
    except:
        print("Could not set constraint weight - method may not be available")
        
    # Realize position to ensure model is properly initialized (matching C++ line 94)
    model.realizePosition(s)

    # Get the actual time range from the quaternion table
    times = quatTable.getIndependentColumn();
    startTime = times[0];  # Access first element of tuple
    endTime = times[-1];   # Access last element of tuple
    numTimeSteps = len(times);
    print(f"Data time range: {startTime} to {endTime} seconds");
    print(f"Number of time steps: {numTimeSteps}");
    
    # Create storage for results
    storage = osim.Storage();
    storage.setName("Coordinates");
    # Choose degrees for the output
    storage.setInDegrees(True);

    
    # Get coordinate names for the header
    coordSet = model.getCoordinateSet();
    numCoords = coordSet.getSize();
    
    # Set column labels
    labels = osim.ArrayStr();
    labels.append("time");
    for i in range(numCoords):
        labels.append(coordSet.get(i).getName());
    storage.setColumnLabels(labels);
    
    # Initialize the solver at the first time point (matching C++ implementation)
    s.setTime(startTime);
    
    # Realize position before assembly (important for stability)
    model.realizePosition(s)
    
    # Assemble the model at the first time point (matching C++ line 243)
    ikSolver.assemble(s);  # Only assemble once at the beginning
    
    print(f"IK Solver initialized and assembled. Starting processing...");
    
    # Remove coordinate smoothing to get raw IK results first
    # We'll focus on getting the coordinate system alignment correct
    
    # Start timing
    start_time = time.time();
    
    # Process each time frame
    print(f"Processing {numTimeSteps} frames...")
    
    for i, time_val in enumerate(times):
        # Show progress every 10 frames or at the end
        if i % 10 == 0 or i == numTimeSteps - 1:
            print(f"Processing frame {i+1}/{numTimeSteps}: {time_val:.4f}s");
        
        # Set the state to current time
        s.setTime(time_val);

        # Track for this time step (assemble is called internally by track)
        ikSolver.track(s);
        
        # Get coordinate values from the state and convert rotational coordinates to degrees
        coordValues = osim.Vector(numCoords, 0.0);  # Initialize with size and default value
        for j in range(numCoords):
            coord = coordSet.get(j);
            value = coord.getValue(s);
            coord_name = coord.getName()
            
            # Convert rotational coordinates from radians to degrees
            if coord.getMotionType() == osim.Coordinate.Rotational:
                value = value * 180.0 / pi;  # Convert radians to degrees
            
            coordValues.set(j, value);
        
        # Append to storage
        storage.append(time_val, coordValues);
    
    # End timing
    end_time = time.time();
    elapsed_time = end_time - start_time;
    
    print(f"IK processing completed for {numTimeSteps} frames in {elapsed_time:.2f} seconds.");

    # Save results as .mot file
    motFileName = resultsDirectory + "/inverse_kinematics_results.mot";
    storage.printResult(storage, "inverse_kinematics_results", resultsDirectory, -1, ".mot");
    print(f"Results saved to: {motFileName}");

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenSense Inverse Kinematics.")
    parser.add_argument("modelFileName", type=str, help="Calibrated model file path")
    parser.add_argument("orientationsFileName", type=str, help="Orientation File path")
    args = parser.parse_args()

    main(args.modelFileName, args.orientationsFileName)