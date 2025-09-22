import opensim as osim
from opensim import Vec3
import numpy as np
import math
from math import pi
import argparse
import os
import shutil
import time
from OpenSense_CalibrateModel import main as imu_calibrate_model
from OpenSense_IMUDataConverter import main as convert_imu_data
from modelScalingWebcam import main as model_scaling_webcam
from modelScalingStereocamera import main as model_scaling_stereocamera
from webcam_ref_frame_correction import main as correct_webcam_ref_frame
from stereocamera_ref_frame_correction import main as correct_stereocamera_ref_frame
from model_add_markers import main as model_add_markers

class OpenSimSensorFusion:
    def __init__(self, model_path, model_name, subject_ID, trial_ID, imu_to_opensim_rotation, subject_mass, subject_height, subject_age, subject_sex):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.s = None
        self.subject_ID = subject_ID
        self.trial_ID = trial_ID
        self.subject_mass = subject_mass
        self.subject_height = subject_height
        self.subject_age = subject_age
        self.subject_sex = subject_sex
        self.webcam_weights = None
        self.orientation_weights = None
        self.stereocamera_weights = None
        self.constraint_var = None
        self.imu_to_opensim_rotation = imu_to_opensim_rotation
        self.base_heading_axis = "-z"  
        self.resultsDirectory = None
        self.webcamMarkerTable = None
        self.webcamMarkerLabels = None
        self.orientationQuatTable = None
        self.webcamMarkersFileName = None
        self.orientationLabels = None
        self.stereocameraMarkerTable = None
        self.stereocameraMarkerLabels = None
        self.stereocameraMarkersFileName = None
        self.stereocameraMarkerWeights = None
        self.oRefs = None
        self.mRefs = None
        self.combined_marker_table = None
        self.combinedMarkerWeights = None
        self.times = None 
        self.orientation_file = None  

    def get_imu_data(self):
        # Convert IMU data
        self.orientation_file = convert_imu_data(self.subject_ID, self.trial_ID, defaultMapping = True)
        print(f"IMU data converted and saved to: {self.orientation_file}")

        # Get the orientation labels from the converted file
        self.orientationQuatTable = osim.TimeSeriesTableQuaternion(self.orientation_file)
        self.orientationLabels = self.orientationQuatTable.getColumnLabels()
        
        imuPlacer = osim.IMUPlacer() 

    def correct_webcam_data(self):
        correct_webcam_ref_frame(self.subject_ID, self.trial_ID)
        
    def get_webcam_data(self, webcam_path = None):
        # Set up the markers file
        if webcam_path is not None:
            self.webcamMarkersFileName = webcam_path
        else:
            self.webcamMarkersFileName = "recordings/subject" + str(self.subject_ID) + "/webcam_" + str(self.trial_ID) + ".trc"

        # Load marker data 
        self.webcamMarkerTable = osim.TimeSeriesTableVec3(self.webcamMarkersFileName)
        print(f"Loaded marker data from: {self.webcamMarkersFileName}")
        webcamMarkerTimes = self.webcamMarkerTable.getIndependentColumn()
        print(f"Marker data time range: {webcamMarkerTimes[0]:.4f} to {webcamMarkerTimes[-1]:.4f} seconds")
        print(f"Number of markers: {self.webcamMarkerTable.getNumColumns()}")
        print(f"Marker data points: {len(webcamMarkerTimes)}")

        # Show first few timestamps for debugging
        print(f"First 5 marker timestamps: {[f'{t:.4f}' for t in webcamMarkerTimes[:5]]}")
        print(f"Last 5 marker timestamps: {[f'{t:.4f}' for t in webcamMarkerTimes[-5:]]}")
        
        # Get marker names
        self.webcamMarkerLabels = self.webcamMarkerTable.getColumnLabels()
        print(f"Available markers: {[str(label) for label in self.webcamMarkerLabels]}")
        
        # Validate marker data for missing/invalid positions
        print("Validating webcam marker data...")
        extreme_count = 0
        nan_count = 0
        max = 5*1000 # 5 meters in mm
        for i in range(self.webcamMarkerTable.getNumRows()):
            row = self.webcamMarkerTable.getRowAtIndex(i)
            for j in range(row.size()):
                marker_pos = row.getElt(0, j)
                # Check for extremely large values
                if (abs(marker_pos.get(0)) > max or abs(marker_pos.get(1)) > max or abs(marker_pos.get(2)) > max):
                    extreme_count += 1
                if (math.isnan(marker_pos.get(0)) or math.isnan(marker_pos.get(1)) or math.isnan(marker_pos.get(2))):
                    nan_count += 1

        if extreme_count > 0:
            print(f"WARNING: Found {extreme_count} extreme marker positions in webcam data (>10m from origin)")
        elif nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN marker positions in webcam data")
        else:
            print("✓ All webcam marker positions appear reasonable")
    
    def correct_stereocamera_data(self):
        correct_stereocamera_ref_frame(self.subject_ID, self.trial_ID)

    def get_stereocamera_data(self, stereocamera_path = None):
        # Set up the markers file
        if stereocamera_path is not None:
            self.stereocameraMarkersFileName = stereocamera_path
        else:
            self.stereocameraMarkersFileName = "recordings/subject" + str(self.subject_ID) + "/stereocamera_" + str(self.trial_ID) + ".trc"

        # Load marker data 
        self.stereocameraMarkerTable = osim.TimeSeriesTableVec3(self.stereocameraMarkersFileName)
        print(f"Loaded stereocamera marker data from: {self.stereocameraMarkersFileName}")
        stereocameraMarkerTimes = self.stereocameraMarkerTable.getIndependentColumn()
        print(f"Stereocamera marker data time range: {stereocameraMarkerTimes[0]:.4f} to {stereocameraMarkerTimes[-1]:.4f} seconds")
        print(f"Number of markers: {self.stereocameraMarkerTable.getNumColumns()}")
        print(f"Stereocamera marker data points: {len(stereocameraMarkerTimes)}")

        # Show first few timestamps for debugging
        print(f"First 5 stereocamera marker timestamps: {[f'{t:.4f}' for t in stereocameraMarkerTimes[:5]]}")
        print(f"Last 5 stereocamera marker timestamps: {[f'{t:.4f}' for t in stereocameraMarkerTimes[-5:]]}")
        
        # Get marker names
        self.stereocameraMarkerLabels = self.stereocameraMarkerTable.getColumnLabels()
        print(f"Available stereocamera markers: {[str(label) for label in self.stereocameraMarkerLabels]}")
        
        # Validate marker data for missing/invalid positions
        print("Validating stereocamera marker data...")
        nan_count = 0
        extreme_count = 0
        for i in range(self.stereocameraMarkerTable.getNumRows()):
            row = self.stereocameraMarkerTable.getRowAtIndex(i)
            for j in range(row.size()):
                marker_pos = row.getElt(0, j)
                # Check for NaN or extremely large values
                max = 5*1000 # 5 meters in mm
                if (abs(marker_pos.get(0)) > max or abs(marker_pos.get(1)) > max or abs(marker_pos.get(2)) > max):
                    extreme_count += 1
                    if extreme_count < 5:  # Only print first few
                        print(f"  Extreme marker position at row {i}, marker {j}: ({marker_pos.get(0):.3f}, {marker_pos.get(1):.3f}, {marker_pos.get(2):.3f})")
                if (math.isnan(marker_pos.get(0)) or math.isnan(marker_pos.get(1)) or math.isnan(marker_pos.get(2))):
                    nan_count += 1

        if extreme_count > 0:
            print(f"WARNING: Found {extreme_count} extreme marker positions in stereocamera data (>10m from origin)")
            print("This could cause the billion-scale errors you're seeing!")
        elif nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN marker positions in stereocamera data")
        else:
            print("✓ All stereocamera marker positions appear reasonable")
        

    def calibrate_model(self): 
        calibrated_model_path = 'OpenSim/Models/Rajagopal/Calibrated_Rajagopal_subject' + str(self.subject_ID) +'_' + str(self.trial_ID) + '.osim'
        # Webcam scaling and marker placement
        if (max(self.webcam_weights) > 0 and max(self.stereocamera_weights) == 0):
            calibrated_model_path = model_scaling_webcam(self.subject_ID, self.trial_ID, self.subject_mass, self.subject_height, self.subject_age, self.subject_sex, self.model_path)
        else:
            # If no webcam markers are used, copy the original model
            calibrated_model_path = self.model_path
        # Stereocamera scaling and marker placement
        if (max(self.stereocamera_weights) > 0 and max(self.webcam_weights) == 0):
            calibrated_model_path = model_scaling_stereocamera(self.subject_ID, self.trial_ID, self.subject_mass, self.subject_height, self.subject_age, self.subject_sex, calibrated_model_path)
        # If both Webcam and Stereocamera are being used
        if (max(self.stereocamera_weights) > 0 and max(self.webcam_weights) > 0):
            # first calibrate model with stereocamera data
            calibrated_model_path = model_scaling_stereocamera(self.subject_ID, self.trial_ID, self.subject_mass, self.subject_height, self.subject_age, self.subject_sex, calibrated_model_path)
            # then add webcam markers to model
            calibrated_model_path = model_add_markers(calibrated_model_path, camera_type="webcam")
        # IMU calibration
        if (max(self.orientation_weights) > 0):
            calibrated_model_path = imu_calibrate_model(calibrated_model_path, self.model_name, self.orientation_file, self.subject_ID, self.trial_ID)
        

        self.model = osim.Model(calibrated_model_path)
        self.s = self.model.initSystem()  
        print("Model Mass:", self.model.getTotalMass(self.s))

        nb_markers = self.model.getMarkerSet().getSize()
        print(f"Model calibrated and scaled: {calibrated_model_path}")
        print(f"Number of markers in the model: {nb_markers}")

        return calibrated_model_path

    def set_weights(self, webcam_weights, orientation_weights, stereocamera_weights, constraint_var):
        self.webcam_weights = webcam_weights
        self.orientation_weights = orientation_weights
        self.stereocamera_weights = stereocamera_weights
        self.constraint_var = constraint_var # will actually be infinity

        # Create orientation weights
        self.orientationWeights = osim.OrientationWeightSet()
        for i, label in enumerate(self.orientationLabels):
            orientationWeight = osim.OrientationWeight()
            orientationWeight.setName(str(label))
            orientationWeight.setWeight(self.orientation_weights[i])
            self.orientationWeights.cloneAndAppend(orientationWeight)

        # Create marker weights using the correct OpenSim API pattern
        # Give markers higher weight to prioritize marker data over IMU orientations
        self.combinedMarkerWeights = osim.SetMarkerWeights()
        if max(self.webcam_weights) > 0 :
            for i, label in enumerate(self.webcamMarkerLabels):
                markerWeight = osim.MarkerWeight()
                markerWeight.setName(str(label))
                markerWeight.setWeight(self.webcam_weights[i])
                self.combinedMarkerWeights.cloneAndAppend(markerWeight)
        # Create stereocamera weights
        if max(self.stereocamera_weights) > 0:
            for i, label in enumerate(self.stereocameraMarkerLabels):
                markerWeight = osim.MarkerWeight()
                markerWeight.setName(str(label))
                markerWeight.setWeight(self.stereocamera_weights[i])
                self.combinedMarkerWeights.cloneAndAppend(markerWeight)

        print(f"Webcam marker weight: {self.webcam_weights}")
        print(f"Stereocamera marker weight: {self.stereocamera_weights}")
        print(f"IMU orientation weight: {self.orientation_weights}")

    def manual_downsampling(self):
        # Downsampling the orientation data to match marker timestamps test
        imu_times = self.orientationQuatTable.getIndependentColumn()
        webcam_times = self.webcamMarkerTable.getIndependentColumn()
        stereocamera_times = self.stereocameraMarkerTable.getIndependentColumn()

        print(f"IMU times: {len(imu_times)} samples")
        print(f"Webcam times: {len(webcam_times)} samples")
        print(f"Stereocamera times: {len(stereocamera_times)} samples")

        # Find the shortest timestamps to downsample
        times = []
        # first exclude data with zero weights - so data that is not being used
        if max(self.webcam_weights) > 0 and len(webcam_times) > 0:
            times.append(webcam_times)
            print("Including webcam times for downsampling")
        if max(self.stereocamera_weights) > 0 and len(stereocamera_times) > 0:
            times.append(stereocamera_times)
            print("Including stereocamera times for downsampling")
        if max(self.orientation_weights) > 0 and len(imu_times) > 0:
            times.append(imu_times)
            print("Including IMU times for downsampling")
        
        # Find the latest start time and the earliest end time
        if len(times) > 0:
            start_time = max([t[0] for t in times])
            end_time = min([t[-1] for t in times])
            print(f"Common time range for downsampling: {start_time:.7f} to {end_time:.7f} seconds")
            # Filter each time array to the common time range
            for i in range(len(times)):
                times[i] = [t for t in times[i] if t >= start_time and t <= end_time]
                print(f"Filtered times array {i} to {len(times[i])} samples")
        
        # Then find the shortest timestamp array
        shortest_times = []
        self.new_fps = 0
        self.nb_frames = 0
        if len(times) > 1:
            shortest_times = min(times, key=len)
            print(f"Using shortest timestamps for downsampling: {len(shortest_times)} samples")
            self.nb_frames = len(shortest_times)
            self.new_fps = 1 / (shortest_times[-1] - shortest_times[0]) * self.nb_frames
            #new_fps = new_fps 
        elif len(times) == 1:
            shortest_times = times[0]
            self.nb_frames = len(shortest_times)
            self.new_fps = 1 / (shortest_times[-1] - shortest_times[0]) * self.nb_frames
            #self.new_fps = self.new_fps
        else:
            print("No data available, downsampling aborted")
            return
        
        ### Create new downsampled files
        # IMU data
        if max(self.orientation_weights) != 0 and self.nb_frames != len(imu_times):
            print("Downsampling the IMU data")
            new_imu_path = self.orientation_file.replace('.sto', '_downsampled_'+ str(len(shortest_times)) +'.sto')  
            # Check if this has already been done in the past
            if os.path.isfile(new_imu_path):
                print(f"The file '{new_imu_path}' exists.")
                imu_downsampling = False
                imu_downsampled_exists = True
            else:
                print(f"The file '{new_imu_path}' does not exist.")
                imu_downsampling = True
                # delete all data except for the header
                with open(self.orientation_file, 'r') as f:
                    lines = f.readlines()
                with open(new_imu_path, 'w') as f:
                    f.writelines(lines[:5])  
                imu_downsampled_exists = True
        else:
            imu_downsampling = False
            imu_downsampled_exists = False
        # Webcam data
        if max(self.webcam_weights) != 0 and self.nb_frames != len(webcam_times):
            print("Downsampling the Webcam data")
            new_webcam_path = self.webcamMarkersFileName.replace('.trc', '_downsampled_'+ str(len(shortest_times)) +'.trc')
            # Check if this has already been done in the past
            if os.path.isfile(new_webcam_path):
                print(f"The file '{new_webcam_path}' exists.")
                webcam_downsampling = False
                webcam_downsampled_exists = True
            else:
                print(f"The file '{new_webcam_path}' does not exist.")
                webcam_downsampling = True
                # rewrite the header
                header = (
                    f"PathFileType\t4\t(X/Y/Z)\t{new_webcam_path}\n"
                    f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
                    f"{self.new_fps}\t{self.new_fps}\t{self.nb_frames}\t12\tmm\t{self.new_fps}\t1\t{self.nb_frames}\n"
                )
                with open(self.webcamMarkersFileName, 'r') as f:
                    lines = f.readlines()
                with open(new_webcam_path, 'w') as f:
                    f.write(header)
                    f.writelines(lines[3:6])  
                webcam_downsampled_exists = True
        else: 
            webcam_downsampling = False
            webcam_downsampled_exists = False
        # Stereocamera data
        if max(self.stereocamera_weights) != 0 and self.nb_frames != len(stereocamera_times):
            print("Downsampling the Stereocamera data")
            new_stereocamera_path = self.stereocameraMarkersFileName.replace('.trc', '_downsampled_'+ str(len(shortest_times)) +'.trc')
            # Check if this has already been done in the past
            if os.path.isfile(new_stereocamera_path):
                print(f"The file '{new_stereocamera_path}' exists.")
                stereocamera_downsampling = False
                stereocomera_downsampled_exists = True
            else:
                print(f"The file '{new_stereocamera_path}' does not exist.")
                stereocamera_downsampling = True
                # delete all data except for the header
                with open(self.stereocameraMarkersFileName, 'r') as f:
                    lines = f.readlines()
                # rewrite header
                header = (
                    f"PathFileType\t4\t(X/Y/Z)\t{new_stereocamera_path}\n"
                    f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
                    f"{self.new_fps}\t{self.new_fps}\t{self.nb_frames}\t15\tmm\t{self.new_fps}\t1\t{self.nb_frames}\n"
                )
                with open(new_stereocamera_path, 'w') as f:
                    f.write(header)
                    f.writelines(lines[3:6])  
                stereocomera_downsampled_exists = True
        else: 
            stereocamera_downsampling = False
            stereocomera_downsampled_exists = False
        ## Downsampling
        # Header line indexes
        index_imu = 5
        index_webcam = 6
        index_stereocamera = 6
        index = 1
        # Iterate through shortest timestamps and find closest IMU orientation and closest Marker positions
        for time in shortest_times:
            # Find the closest IMU orientation if IMU are being downsampled
            if imu_downsampling:
                # Read old file
                with open(self.orientation_file, 'r') as f:
                    lines = f.readlines()
                # Find the closest IMU orientation for that time, knowing that the file is sorted by time
                closest_orientation = None
                closest_diff = float('inf')
                for line in lines[index_imu:]:  # Skip header and skip processed lines
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        try:
                            timestamp = float(parts[0])
                            if timestamp == time:
                                closest_orientation = line
                                break
                            diff = abs(timestamp - time)
                            # if the difference is growing, we can stop searching
                            if diff > closest_diff:
                                break
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_orientation = line
                                index_imu = lines.index(line)
                        except ValueError:
                            continue
                if closest_orientation:
                    # change the timestamp to the current time
                    parts = closest_orientation.split('\t')
                    parts[0] = f"{time:.7f}"
                    closest_orientation = '\t'.join(parts)
                    with open(new_imu_path, 'a') as f:
                        f.write(closest_orientation)

            # Find the closest webcam marker positions if webcam data is being downsampled
            if webcam_downsampling:
                # Read old file
                with open(self.webcamMarkersFileName, 'r') as f:
                    lines = f.readlines()
                # Find the closest webcam marker positions for that time, knowing that the file is sorted by time
                closest_markers = None
                closest_diff = float('inf')
                for line in lines[index_webcam:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        try:
                            timestamp = float(parts[1])
                            if timestamp == time:
                                closest_markers = line
                                break
                            diff = abs(timestamp - time)
                            # if the difference is growing, we can stop searching
                            if diff > closest_diff:
                                break
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_markers = line
                                index_webcam = lines.index(line)
                        except ValueError:
                            continue
                if closest_markers:
                    # change the frame number and the timestamp
                    parts = closest_markers.split('\t')
                    parts[0] = f"{index}"
                    parts[1] = f"{time:.6f}"
                    closest_markers = '\t'.join(parts)
                    with open(new_webcam_path, 'a') as f:
                        f.write(closest_markers)

            # Find the closest stereocamera marker positions if stereocamera data is being downsampled
            if stereocamera_downsampling:
                # Read old file
                with open(self.stereocameraMarkersFileName, 'r') as f:
                    lines = f.readlines()
                # Find the closest stereocamera marker positions for that time, knowing that the file is sorted by time
                closest_markers = None
                closest_diff = float('inf')
                for line in lines[index_stereocamera:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        try:
                            timestamp = float(parts[1])
                            if timestamp == time:
                                closest_markers = line
                                break
                            diff = abs(timestamp - time)
                            # if the difference is growing, we can stop searching
                            if diff > closest_diff:
                                break
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_markers = line
                                index_stereocamera = lines.index(line)
                        except ValueError:
                            continue
                if closest_markers:
                    # change the frame number and the timestamp
                    parts = closest_markers.split('\t')
                    parts[0] = f"{index}"
                    parts[1] = f"{time:.6f}"
                    closest_markers = '\t'.join(parts)
                    with open(new_stereocamera_path, 'a') as f:
                        f.write(closest_markers)

            index += 1


        # Replace tables with the downsampled tables by reloading them
        if imu_downsampled_exists:
            self.orientation_file = new_imu_path
            self.orientationQuatTable = osim.TimeSeriesTableQuaternion(self.orientation_file)
            print("New downsampled IMU data loaded")
        if webcam_downsampled_exists:
            self.webcamMarkersFileName = new_webcam_path
            self.get_webcam_data(self.webcamMarkersFileName)
            print("New downsampled webcam data loaded")
        if stereocomera_downsampled_exists:
            self.stereocameraMarkersFileName = new_stereocamera_path
            self.get_stereocamera_data(self.stereocameraMarkersFileName)
            print("New downsampled stereocamera data loaded")

        print(f"Downsampling complete: {self.orientationQuatTable.getNumRows()} rows for orientation data, {self.stereocameraMarkerTable.getNumRows()} rows for stereocamera data, {self.webcamMarkerTable.getNumRows()} rows for webcam data")

        self.times = shortest_times


    def heading_correction_imu_data(self, quatTable):
        # Compute rotation matrix so that (e.g. "pelvis_imu" + - Z Axis) lines up with model forward (+X)
        base_imu_label = "pelvis_imu"  # Replace with your base IMU label
        direction_on_imu = osim.CoordinateDirection(osim.CoordinateAxis(2), -1)  # Negative Z-axis direction

        # Apply the sensor rotation sequence: X, Y, Z rotations
        rotX = osim.Rotation()
        rotX.setRotationFromAngleAboutAxis(self.imu_to_opensim_rotation[0], osim.CoordinateAxis(0))
        rotY = osim.Rotation() 
        rotY.setRotationFromAngleAboutAxis(self.imu_to_opensim_rotation[1], osim.CoordinateAxis(1))
        rotZ = osim.Rotation()
        rotZ.setRotationFromAngleAboutAxis(self.imu_to_opensim_rotation[2], osim.CoordinateAxis(2))
        
        # Apply sensor rotations to quatTable first
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotX)
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotY)
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotZ)
        print(f"Applied sensor to OpenSim rotations: {self.imu_to_opensim_rotation[0]*180/pi:.1f}, {self.imu_to_opensim_rotation[1]*180/pi:.1f}, {self.imu_to_opensim_rotation[2]*180/pi:.1f} degrees")

    
        # Parse heading axis specification (matching C++ logic from lines 104-118)
        imu_axis = self.base_heading_axis.lower()
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
            raise ValueError(f"Invalid heading axis specification: {self.base_heading_axis}")

        print(f"Using heading axis: {self.base_heading_axis} -> CoordinateDirection({axis_char.upper()}Axis, {direction})")

        heading_rotation_vec3 = osim.OpenSenseUtilities.computeHeadingCorrection(
            self.model, self.s, quatTable, base_imu_label, direction_on_imu)

        # Create heading rotation using individual axis rotations (Python OpenSim approach)
        # Apply the rotations in sequence: X, then Y, then Z
        rotX_heading = osim.Rotation()
        rotX_heading.setRotationFromAngleAboutAxis(heading_rotation_vec3.get(0), osim.CoordinateAxis(0))
        rotY_heading = osim.Rotation() 
        rotY_heading.setRotationFromAngleAboutAxis(heading_rotation_vec3.get(1), osim.CoordinateAxis(1))
        rotZ_heading = osim.Rotation()
        rotZ_heading.setRotationFromAngleAboutAxis(heading_rotation_vec3.get(2), osim.CoordinateAxis(2))
        
        # Apply rotations in sequence to achieve the same effect as SpaceRotationSequence
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotX_heading)
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotY_heading)
        osim.OpenSenseUtilities.rotateOrientationTable(quatTable, rotZ_heading)
        
        # Calculate the magnitude of correction for logging
        correction_magnitude = math.sqrt(heading_rotation_vec3.get(0)**2 + 
                                    heading_rotation_vec3.get(1)**2 + 
                                    heading_rotation_vec3.get(2)**2)
        
        print(f"Applied heading correction: X={heading_rotation_vec3.get(0)*180/pi:.1f}°, "
              f"Y={heading_rotation_vec3.get(1)*180/pi:.1f}°, "
              f"Z={heading_rotation_vec3.get(2)*180/pi:.1f}° "
              f"(magnitude: {correction_magnitude*180/pi:.1f}°)")
        
        # Convert orientations for further processing
        orientationData = osim.OpenSenseUtilities.convertQuaternionsToRotations(quatTable)
        self.orientationQuatTable = orientationData

    def load_references(self, is_webcam_used, is_imu_used, is_stereocamera_used):
        # Create OrientationsReference 
        if not is_imu_used:
            # Create an empty OrientationsReference to completely disable orientation constraints
            print("Creating EMPTY OrientationsReference to disable all IMU orientation constraints")
            self.oRefs = osim.OrientationsReference()  # Empty reference - no orientation data
            weights = osim.OrientationWeightSet() # Empty weights
            self.oRefs.setOrientationWeightSet(weights)
            print(f"Orientation weights: {[weights.get(i).getWeight() for i in range(weights.getSize())]}")
        else:
            self.oRefs = osim.OrientationsReference(self.orientationQuatTable, self.orientationWeights) # it's called QuatTable but it is actually Rotations 
            print(f"OrientationsReference created")
            self.oRefs.setOrientationWeightSet(self.orientationWeights)
            print(f"Orientation weights: {[self.orientationWeights.get(i).getWeight() for i in range(self.orientationWeights.getSize())]}")
        # Create MarkersReference from loaded marker data 
        if not is_webcam_used and not is_stereocamera_used:
            # Create an empty MarkersReference to completely disable marker constraints
            print("Creating EMPTY MarkersReference to disable all marker constraints")
            self.mRefs = osim.MarkersReference()  # Empty reference - no marker data
            weights = self.mRefs.getMarkerWeightSet() # Empty weights
            print(f"Combined marker weights: {[weights.get(i).getWeight() for i in range(weights.getSize())]}")
        elif is_webcam_used and not is_stereocamera_used:
            # Create MarkersReference from webcam data only
            self.combined_marker_table = self.webcamMarkerTable
            self.mRefs = osim.MarkersReference(self.combined_marker_table, self.combinedMarkerWeights)
            print(f"MarkersReference created from webcam markers")
            weights = self.mRefs.getMarkerWeightSet()
            print(f"Combined marker weights: {[weights.get(i).getWeight() for i in range(weights.getSize())]}")
        elif not is_webcam_used and is_stereocamera_used:
            # Create MarkersReference from stereocamera data only
            self.combined_marker_table = self.stereocameraMarkerTable
            self.mRefs = osim.MarkersReference(self.combined_marker_table, self.combinedMarkerWeights)
            print(f"MarkersReference created from stereocamera markers")
            weights = self.mRefs.getMarkerWeightSet()
            print(f"Combined marker weights: {[weights.get(i).getWeight() for i in range(weights.getSize())]}")
        else: 
            # Combine webcam and stereocamera markers into a single MarkersReference
            self.combined_marker_table = osim.TimeSeriesTableVec3()
            self.combined_marker_table.setColumnLabels(
                list(self.webcamMarkerTable.getColumnLabels()) + 
                list(self.stereocameraMarkerTable.getColumnLabels())
            )
            for time in self.webcamMarkerTable.getIndependentColumn():
                # Get rows from both tables
                webcam_row = self.webcamMarkerTable.getRowAtIndex(
                    self.webcamMarkerTable.getNearestRowIndexForTime(time))
                stereo_row = self.stereocameraMarkerTable.getRowAtIndex(
                    self.stereocameraMarkerTable.getNearestRowIndexForTime(time))
                
                # Create a proper RowVector for the combined data
                total_markers = webcam_row.size() + stereo_row.size()
                combined_row = osim.RowVectorVec3(total_markers)
                
                # Copy webcam markers first
                for i in range(webcam_row.size()):
                    combined_row[i] = webcam_row.getElt(0, i)
                
                # Copy stereocamera markers after webcam markers
                for i in range(stereo_row.size()):
                    combined_row[webcam_row.size() + i] = stereo_row.getElt(0, i)
                
                # Append to combined table using proper RowVector
                self.combined_marker_table.appendRow(time, combined_row)
            
            # Set required metadata 
            self.combined_marker_table.addTableMetaDataString("DataRate", str(self.new_fps))
            self.combined_marker_table.addTableMetaDataString("CameraRate", str(self.new_fps))
            self.combined_marker_table.addTableMetaDataString("NumFrames", str(self.nb_frames))
            self.combined_marker_table.addTableMetaDataString("NumMarkers", str(webcam_row.size() + stereo_row.size()))
            self.combined_marker_table.addTableMetaDataString("Units", "mm")
            self.combined_marker_table.addTableMetaDataString("OrigDataRate", str(self.new_fps))
            self.combined_marker_table.addTableMetaDataString("OrigDataStartFrame", "1")
            self.combined_marker_table.addTableMetaDataString("OrigNumFrames", str(self.nb_frames))
            # Create the combined MarkersReference
            self.mRefs = osim.MarkersReference(self.combined_marker_table, self.combinedMarkerWeights)
            print(f"MarkersReference created by combining webcam and stereocamera markers")
            weights = self.mRefs.getMarkerWeightSet()
            print(f"Combined marker weights: {[weights.get(i).getWeight() for i in range(weights.getSize())]}")


def main(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
    # Put log to level debug and show in terminal
    osim.Logger.setLevel(osim.Logger.Level_Info)

    model_path = 'OpenSim/Models/Rajagopal/Rajagopal_2015.osim'
    model_name = 'Rajagopal'
    imu_to_opensim_rotation = Vec3(-pi/2, 0, 0) 
    sensor_fusion = OpenSimSensorFusion(model_path, model_name, subject_ID, trial_ID, imu_to_opensim_rotation, subject_mass, subject_height, subject_age, subject_sex)
    sensor_fusion.resultsDirectory = '../'+ trial_ID +'_FusionIKResults'

    # Get IMU data
    sensor_fusion.get_imu_data()

    # Get webcam marker data and correct its coordinate system
    sensor_fusion.correct_webcam_data()
    sensor_fusion.get_webcam_data()

    # Get stereocamera marker data and correct its coordinate system
    sensor_fusion.correct_stereocamera_data()
    sensor_fusion.get_stereocamera_data()

    # Weights for sensor fusion are set by the weight tuning module
    # Webcam marker weight, IMU orientation weight, Stereocamera marker weight, Constraint variable
    # Notes: for now changing weights within a single sensor works and changes the output, but changing weights between sensors does not seem to have an effect
    
    sensor_fusion.set_weights(webcam_weights, orientation_weights, stereocamera_weights, constraint_var)

    is_webcam_used = max(webcam_weights) > 0  
    is_imu_used = max(orientation_weights) > 0 
    is_stereocamera_used = max(stereocamera_weights) > 0  

    # Manual file downsampling and then reloading tables
    sensor_fusion.manual_downsampling()

    # Calibrate the model
    calibrated_model_path = sensor_fusion.calibrate_model()
    print(f"Calibrated model saved to: {calibrated_model_path}")

    # Correct IMU orientations
    sensor_fusion.heading_correction_imu_data(sensor_fusion.orientationQuatTable)

   
    # Load the orientation and marker references
    sensor_fusion.load_references(is_webcam_used, is_imu_used, is_stereocamera_used)
    

    # Create the solver
    coordinateReferences = osim.SimTKArrayCoordinateReference()
    if sensor_fusion.constraint_var < 10000:
        ikSolver = osim.InverseKinematicsSolver(sensor_fusion.model, sensor_fusion.mRefs, sensor_fusion.oRefs, coordinateReferences, sensor_fusion.constraint_var) 
    else : 
        # No constraint variable means Infinite weight, so the joint limits have to be applied, this is the default value for the InverseKinematicsSolver
        ikSolver = osim.InverseKinematicsSolver(sensor_fusion.model, sensor_fusion.mRefs, sensor_fusion.oRefs, coordinateReferences)


    accuracy = 1e-9  # Same as C++ implementation, probably default value
    ikSolver.setAccuracy(accuracy)
    print(f"\n=== SOLVER CONFIGURATION ANALYSIS ===")
    # Check what the solver actually sees
    print(f"Solver configuration:")
    print(f"  - Markers reference: {sensor_fusion.mRefs.getNumFrames() if hasattr(sensor_fusion.mRefs, 'getNumFrames') else 'unknown'} frames")
    print(f"  - Orientations reference: {len(sensor_fusion.oRefs.getTimes()) if hasattr(sensor_fusion.oRefs, 'getTimes') else 'unknown'} frames")
    print(f"  - Constraint variable: {sensor_fusion.constraint_var}")
    print(f"  - Accuracy: {accuracy}")
    print("="*60)


    # Initialize the model  
    # Realize position to ensure model is properly initialized
    sensor_fusion.model.realizePosition(sensor_fusion.s)

    # Define the overlapping time range for combined mode
    # Get the actual time range - use overlapping time range for combined mode
    times = sensor_fusion.times
    startTime = times[0]  
    endTime = times[-1]

    numTimeSteps = len(times) 
    print(f"Data time range: {startTime} to {endTime} seconds") 
    print(f"Number of time steps: {numTimeSteps}") 


    ## For saving the results
    directory = sensor_fusion.orientation_file.rpartition('/')[0] 
    resultsDirectory = directory + '/' + sensor_fusion.resultsDirectory 
    # Create results directory if it doesn't exist
    os.makedirs(resultsDirectory, exist_ok=True) 

    # Create storage for results
    storage = osim.Storage() 
    storage.setName("Coordinates") 
    # Choose degrees for the output
    storage.setInDegrees(True) 
    
    # Get coordinate names for the header
    coordSet = sensor_fusion.model.getCoordinateSet() 
    numCoords = coordSet.getSize() 
    
    # Set column labels
    labels = osim.ArrayStr() 
    labels.append("time") 
    for i in range(numCoords):
        labels.append(coordSet.get(i).getName()) 
    storage.setColumnLabels(labels) 

    ## For saving the errors
    # Create storage objects for error tracking 
    marker_errors_storage = None
    orientation_errors_storage = None
    combined_errors_storage = None
    
    # Initialize marker error storage if markers are used
    if is_webcam_used or is_stereocamera_used:
        marker_errors_storage = osim.Storage()
        marker_errors_storage.setName("Model Marker Errors from IK")
        marker_errors_storage.setInDegrees(False)  # Errors are in meters
        
        # Set column labels following OpenSim C++ pattern
        marker_labels = osim.ArrayStr()
        marker_labels.append("time")
        marker_labels.append("webcam_total_squared_error")
        marker_labels.append("webcam_error_RMS")
        marker_labels.append("webcam_error_max")
        marker_labels.append("stereocamera_total_squared_error")
        marker_labels.append("stereocamera_error_RMS")
        marker_labels.append("stereocamera_error_max")
        
        # Add individual marker error columns
        for label in sensor_fusion.combined_marker_table.getColumnLabels():
            marker_labels.append(f"{str(label)}_error")
            
        marker_errors_storage.setColumnLabels(marker_labels)
        
    # Initialize orientation error storage if orientations are used
    if is_imu_used:
        orientation_errors_storage = osim.Storage()
        orientation_errors_storage.setName("Model Orientation Errors from IK")
        orientation_errors_storage.setInDegrees(True)  # Convert radians to degrees
        
        # Set column labels
        orient_labels = osim.ArrayStr()
        orient_labels.append("time")
        orient_labels.append("total_squared_error")
        orient_labels.append("orientation_error_RMS")
        orient_labels.append("orientation_error_max")
        
        # Add individual orientation error columns
        for label in sensor_fusion.orientationLabels:
            orient_labels.append(f"{str(label)}_error")
            
        orientation_errors_storage.setColumnLabels(orient_labels)
    
    # Initialize combined error storage
    combined_errors_storage = osim.Storage()
    combined_errors_storage.setName("Combined IK Errors")
    combined_errors_storage.setInDegrees(False)
    
    combined_labels = osim.ArrayStr()
    combined_labels.append("time")
    combined_labels.append("total_cost")
    combined_labels.append("webcam_total_squared_error")
    combined_labels.append("stereocamera_total_squared_error")
    combined_labels.append("orientation_total_squared_error")
    combined_labels.append("webcam_rms_error")
    combined_labels.append("webcam_max_error")
    combined_labels.append("stereocamera_rms_error")
    combined_labels.append("stereocamera_max_error")
    combined_labels.append("orientation_rms_error_deg")
    combined_labels.append("orientation_max_error_deg")
    
    combined_errors_storage.setColumnLabels(combined_labels)

    # Initialise nb of markers
    if is_webcam_used and is_stereocamera_used:
        num_webcam_markers = len(sensor_fusion.webcam_weights)
        num_stereocamera_markers = len(sensor_fusion.stereocamera_weights) 
    elif is_webcam_used and not is_stereocamera_used:
        num_webcam_markers = len(sensor_fusion.webcam_weights)
        num_stereocamera_markers = 0
    elif not is_webcam_used and is_stereocamera_used:
        num_webcam_markers = 0
        num_stereocamera_markers = len(sensor_fusion.stereocamera_weights)
    else:
        num_webcam_markers = 0
        num_stereocamera_markers = 0
    
    # Intialise nb of orientationa
    if is_imu_used:
        num_orientations = len(sensor_fusion.orientation_weights)
    else:
        num_orientations = 0


    # Initialize the solver at the first time point 
    # Use the first time from our filtered times array
    first_time = times[0]
    sensor_fusion.s.setTime(first_time) 

    # Realize position before assembly (important for stability)
    sensor_fusion.model.realizePosition(sensor_fusion.s)
    
    # Assemble the model at the first time point
    ikSolver.assemble(sensor_fusion.s)   # Only assemble once at the beginning
    print(f"IK Solver initialized and assembled. Starting processing...") 
    
    # Start timing
    start_time = time.time() 
    
    # Process each time frame
    print(f"Processing {numTimeSteps} frames...")
    
    for i, time_val in enumerate(times):
        # Show progress every 10 frames or at the end
        if i % 10 == 0 or i == numTimeSteps - 1:
            print(f"Processing frame {i+1}/{numTimeSteps}: {time_val:.4f}s") 
            
        # Set the state to current time
        sensor_fusion.s.setTime(time_val) 

        # Track for this time step (assemble is called internally by track)
        ikSolver.track(sensor_fusion.s) 

        # Initialise marker errors
        webcam_total_squared_error = 0.0
        webcam_rms = 0.0
        webcam_max = 0.0
        stereocamera_total_squared_error = 0.0
        stereocamera_rms = 0.0
        stereocamera_max = 0.0
        individual_marker_errors = []
        max_webcam_squared_error = 0.0
        max_stereocamera_squared_error = 0.0

        # Orientation errors
        orientation_total_squared_error = 0.0
        orientation_rms = 0.0
        orientation_max = 0.0
        individual_orientation_errors = []
        max_squared_error = 0.0

        if is_webcam_used or is_stereocamera_used:
            try:
                marker_errors = osim.SimTKArrayDouble()
                ikSolver.computeCurrentSquaredMarkerErrors(marker_errors)
                
                if marker_errors.size() > 0:
                    
                    # Validate that we have enough error values for all markers
                    expected_markers = num_webcam_markers + num_stereocamera_markers
                    if marker_errors.size() < expected_markers:
                        print(f"WARNING: Not enough marker errors! Expected {expected_markers}, got {marker_errors.size()}")
                        # Adjust counts to prevent index out of bounds
                        available_for_stereo = max(0, marker_errors.size() - num_webcam_markers)
                        num_stereocamera_markers = min(num_stereocamera_markers, available_for_stereo)
                    
                    # Calculate statistics 
                    for j in range(min(num_webcam_markers, marker_errors.size())):
                        webcam_squared_error = marker_errors.getElt(j)
                        # Check for invalid/extreme values
                        if webcam_squared_error < 0 or webcam_squared_error > 1e6:  # 1 million square meters is extreme
                            print(f"WARNING: Extreme webcam marker error at index {j}: {webcam_squared_error}")
                            webcam_squared_error = min(webcam_squared_error, 1e6)  # Cap at reasonable value
                        
                        webcam_total_squared_error += webcam_squared_error
                        individual_marker_errors.append(math.sqrt(webcam_squared_error))
                        
                        if webcam_squared_error > max_webcam_squared_error:
                            max_webcam_squared_error = webcam_squared_error

                    for j in range(num_webcam_markers, min(num_webcam_markers + num_stereocamera_markers, marker_errors.size())):
                        stereocamera_squared_error = marker_errors.getElt(j)
                        # Check for invalid/extreme values  
                        if stereocamera_squared_error < 0 or stereocamera_squared_error > 1e6:  # 1 million square meters is extreme
                            print(f"WARNING: Extreme stereocamera marker error at index {j}: {stereocamera_squared_error}")
                            stereocamera_squared_error = min(stereocamera_squared_error, 1e6)  # Cap at reasonable value
                            
                        stereocamera_total_squared_error += stereocamera_squared_error
                        individual_marker_errors.append(math.sqrt(stereocamera_squared_error))
                        
                        if stereocamera_squared_error > max_stereocamera_squared_error:
                            max_stereocamera_squared_error = stereocamera_squared_error
                    
                    # Calculate actual number of markers processed for RMS calculation
                    actual_webcam_markers = min(num_webcam_markers, marker_errors.size())
                    actual_stereocamera_markers = min(num_stereocamera_markers, max(0, marker_errors.size() - num_webcam_markers))
                    
                    webcam_rms = math.sqrt(webcam_total_squared_error / actual_webcam_markers) if actual_webcam_markers > 0 else 0
                    webcam_max = math.sqrt(max_webcam_squared_error)
                    stereocamera_rms = math.sqrt(stereocamera_total_squared_error / actual_stereocamera_markers) if actual_stereocamera_markers > 0 else 0
                    stereocamera_max = math.sqrt(max_stereocamera_squared_error)
                    
                    # Save marker error data
                    if marker_errors_storage:
                        marker_data = osim.ArrayDouble()
                        marker_data.append(webcam_total_squared_error)
                        marker_data.append(webcam_rms)
                        marker_data.append(webcam_max)
                        marker_data.append(stereocamera_total_squared_error)
                        marker_data.append(stereocamera_rms)
                        marker_data.append(stereocamera_max)

                        # Add individual marker errors
                        for error in individual_marker_errors:
                            marker_data.append(error)
                        
                        marker_errors_storage.append(time_val, marker_data)
                        
            except Exception as e:
                print(f"Warning: Could not compute marker errors at time {time_val}: {e}")
        
        
        if is_imu_used:
            try:
                orientation_errors = osim.SimTKArrayDouble()
                ikSolver.computeCurrentOrientationErrors(orientation_errors)
                
                if orientation_errors.size() > 0:
                    # Calculate statistics
                    for j in range(orientation_errors.size()):
                        error = orientation_errors.getElt(j)
                        squared_error = error * error
                        orientation_total_squared_error += squared_error
                        individual_orientation_errors.append(error)
                        
                        if squared_error > max_squared_error:
                            max_squared_error = squared_error
                    
                    orientation_rms = math.sqrt(orientation_total_squared_error / num_orientations) if num_orientations > 0 else 0
                    orientation_max = math.sqrt(max_squared_error)
                    
                    # Save orientation error data
                    if orientation_errors_storage:
                        orient_data = osim.ArrayDouble()
                        orient_data.append(orientation_total_squared_error)
                        orient_data.append(orientation_rms * 180.0 / pi)  # Convert to degrees
                        orient_data.append(orientation_max * 180.0 / pi)  # Convert to degrees
                        
                        # Add individual orientation errors (in degrees)
                        for error in individual_orientation_errors:
                            orient_data.append(error * 180.0 / pi)
                            
                        orientation_errors_storage.append(time_val, orient_data)
                        
            except Exception as e:
                print(f"Warning: Could not compute orientation errors at time {time_val}: {e}")
        
        # Combined errors
        total_cost = webcam_total_squared_error + orientation_total_squared_error + stereocamera_total_squared_error
        
        combined_data = osim.ArrayDouble()
        combined_data.append(total_cost)
        combined_data.append(webcam_total_squared_error)
        combined_data.append(stereocamera_total_squared_error)
        combined_data.append(orientation_total_squared_error)
        combined_data.append(webcam_rms)
        combined_data.append(webcam_max)
        combined_data.append(stereocamera_rms)
        combined_data.append(stereocamera_max)
        combined_data.append(orientation_rms * 180.0 / pi if orientation_rms > 0 else 0.0)  # Convert to degrees
        combined_data.append(orientation_max * 180.0 / pi if orientation_max > 0 else 0.0)  # Convert to degrees
        
        combined_errors_storage.append(time_val, combined_data)
        
        # Get coordinate values from the state and convert rotational coordinates to degrees
        coordValues = osim.Vector(numCoords, 0.0)   # Initialize with size and default value
        for j in range(numCoords):
            coord = coordSet.get(j) 
            value = coord.getValue(sensor_fusion.s) 
            coord_name = coord.getName()
            
            # Convert rotational coordinates from radians to degrees
            if coord.getMotionType() == osim.Coordinate.Rotational:
                value = value * 180.0 / pi   # Convert radians to degrees
            
            coordValues.set(j, value) 
        
        # Append to storage
        storage.append(time_val, coordValues) 
    
    # End timing
    end_time = time.time() 
    elapsed_time = end_time - start_time 
    print(f"IK processing completed for {numTimeSteps} frames in {elapsed_time:.2f} seconds.")
    
    # Save all error files (following OpenSim C++ pattern)
    trial_name = f"subject{sensor_fusion.subject_ID}_{sensor_fusion.trial_ID}"
    
    try:
        # Save marker errors
        if marker_errors_storage and (is_webcam_used or is_stereocamera_used):
            marker_filename = f"{trial_name}_ik_marker_errors"
            osim.Storage.printResult(marker_errors_storage, marker_filename, resultsDirectory, -1, ".mot")
            print(f"✓ Saved marker errors to: {resultsDirectory}/{marker_filename}.mot")
        
        # Save orientation errors  
        if orientation_errors_storage and is_imu_used:
            orientation_filename = f"{trial_name}_ik_orientation_errors"
            osim.Storage.printResult(orientation_errors_storage, orientation_filename, resultsDirectory, -1, ".mot")
            print(f"✓ Saved orientation errors to: {resultsDirectory}/{orientation_filename}.mot")
        
        # Save combined errors
        combined_filename = f"{trial_name}_ik_combined_errors"
        osim.Storage.printResult(combined_errors_storage, combined_filename, resultsDirectory, -1, ".mot")
        print(f"✓ Saved combined errors to: {resultsDirectory}/{combined_filename}.mot")
        
        # Print summary statistics (following OpenSim C++ logging pattern)
        
        if (is_webcam_used or is_stereocamera_used ):
            # Use the counts from the sensor fusion object, not the potentially undefined local variables
            total_markers = len(sensor_fusion.webcam_weights) + len(sensor_fusion.stereocamera_weights)
            print(f"\n=== FINAL ERROR SUMMARY ===")
            print(f"Total markers: {total_markers} ({len(sensor_fusion.webcam_weights)} webcam + {len(sensor_fusion.stereocamera_weights)} stereocamera)")
            print(f"  - Final webcam RMS error: {webcam_rms:.6f} m")
            print(f"  - Final webcam max error: {webcam_max:.6f} m")
            print(f"  - Total webcam squared error: {webcam_total_squared_error:.8f}")
            print(f"  - Final stereocamera RMS error: {stereocamera_rms:.6f} m")
            print(f"  - Final stereocamera max error: {stereocamera_max:.6f} m")
            print(f"  - Total stereocamera squared error: {stereocamera_total_squared_error:.8f}")

        if is_imu_used:
            print(f"Orientations: {num_orientations} tracked")  
            print(f"  - Final RMS error: {orientation_rms*180/pi:.4f}°")
            print(f"  - Final max error: {orientation_max*180/pi:.4f}°")
            print(f"  - Total squared error: {orientation_total_squared_error:.8f}")
            
        print(f"Total final cost: {total_cost:.8f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error saving error files: {e}")

    # Save main results as .mot file
    motFileName = resultsDirectory + "/inverse_kinematics_results_subject_"+str(sensor_fusion.subject_ID)+"_trial_"+str(sensor_fusion.trial_ID)+".mot"
    storage.printResult(storage, "inverse_kinematics_results_subject_"+str(sensor_fusion.subject_ID)+"_trial_"+str(sensor_fusion.trial_ID), resultsDirectory, -1, ".mot")
    print(f"Results saved to: {motFileName}")
    
    print("Script execution completed successfully.")

    
    return motFileName



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SensorFusion based inverse kinematics using OpenSim")
    parser.add_argument("webcam_weights", type=lambda s: [float(item) for item in s.split(',')], help="Comma-separated list of webcam marker weights")
    parser.add_argument("orientation_weights", type=lambda s: [float(item) for item in s.split(',')], help="Comma-separated list of IMU orientation weights")
    parser.add_argument("stereocamera_weights", type=lambda s: [float(item) for item in s.split(',')], help="Comma-separated list of stereocamera marker weights")
    parser.add_argument("constraint_var", type=float, help="Constraint variable (use a large number like 10000 for infinite weight)")
    parser.add_argument("subject_ID", type=str, help="Subject ID")
    parser.add_argument("trial_ID", type=str, help="Trial name")
    parser.add_argument("subject_mass", type=float, help="Subject mass in kg")
    parser.add_argument("subject_height", type=float, help="Subject height in mm")
    parser.add_argument("subject_age", type=int, help="Subject age in years")
    parser.add_argument("subject_sex", type=str, choices=['M', 'F'], help="Subject sex (M/F)")

    args = parser.parse_args()

    main(args.subject_ID, args.trial_ID, args.webcam_weights, args.orientation_weights, args.stereocamera_weights, args.constraint_var, args.subject_mass, args.subject_height, args.subject_age, args.subject_sex)