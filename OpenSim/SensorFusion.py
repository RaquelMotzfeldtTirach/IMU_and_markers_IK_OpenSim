# TODO:
# add support for another camera and set of markers
# change code so another script can be used to run the sensor fusion for different parameters (weights + orientation!!)! 
# find in the assembly code the weight for better understanding
# does the order of calibration matter? (IMU, webcam, stereocamera)


import opensim as osim
from opensim import Vec3
import numpy as np
import math
from math import pi
import argparse
import os
import time
from OpenSense_CalibrateModel import main as imu_calibrate_model
from OpenSense_IMUDataConverter import main as convert_imu_data
from modelScalingWebcam import main as model_scaling_webcam
from modelScalingStereocamera import main as model_scaling_stereocamera

class OpenSimSensorFusion:
    def __init__(self, model_path, model_name, subject_ID, trial_ID, imu_to_opensim_rotation, webcam_to_opensim_rotation, stereocamera_to_opensim_rotation, subject_mass, subject_height, subject_age, subject_sex):
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
        self.webcam_to_opensim_rotation = webcam_to_opensim_rotation
        self.stereocamera_to_opensim_rotation = stereocamera_to_opensim_rotation
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

    def get_imu_data(self):
        # Convert IMU data
        orientation_file = convert_imu_data(self.subject_ID, self.trial_ID)
        print(f"IMU data converted and saved to: {orientation_file}")

        # Get the orientation labels from the converted file
        orientationTable = osim.TimeSeriesTableQuaternion(orientation_file)
        self.orientationLabels = orientationTable.getColumnLabels()
        return orientation_file

        
    def get_webcam_data(self):
        # Set up the markers file
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

    def get_stereocamera_data(self):
        # Set up the markers file
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
        

    def calibrate_model(self, orientation_file): 
        ## TODO : DOES THE CALIBRATION ORDER MATTER??? we try first webcam, then stereocamera, then IMU
        calibrated_model_path = 'OpenSim/Models/Rajagopal/Calibrated_Rajagopal_subject' + str(self.subject_ID) +'_' + str(self.trial_ID) + '.osim'
        # Webcam scaling and marker placement
        if (max(self.webcam_weights) > 0):
            calibrated_model_path = model_scaling_webcam(self.subject_ID, self.trial_ID, self.subject_mass, self.subject_height, self.subject_age, self.subject_sex, self.model_path)
            calibrated_model_path = calibrated_model_path.removeprefix("../../")
        # Stereocamera scaling and marker placement
        if (max(self.stereocamera_weights) > 0):
            calibrated_model_path = model_scaling_stereocamera(self.subject_ID, self.trial_ID, self.subject_mass, self.subject_height, self.subject_age, self.subject_sex, calibrated_model_path)
            calibrated_model_path = calibrated_model_path.removeprefix("../../")
        # IMU calibration
        if (max(self.orientation_weights) > 0):
            # calibrated_model_path = imu_calibrate_model(calibrated_model_path, self.model_name, orientation_file, self.subject_ID, self.trial_ID) # TODO: check if this is needed
            calibrated_model_path = imu_calibrate_model(calibrated_model_path, self.model_name, orientation_file, self.subject_ID, self.trial_ID)

        self.model = osim.Model(calibrated_model_path)
        self.s = self.model.initSystem()  

        nb_markers = self.model.getMarkerSet().getSize()
        print(f"Model calibrated and scaled: {calibrated_model_path}")
        print(f"Number of markers in the model: {nb_markers}")

        return calibrated_model_path

    def set_weights(self, webcam_weights, orientation_weights, stereocamera_weights, constraint_var):
        self.webcam_weights = webcam_weights
        self.orientation_weights = orientation_weights
        self.stereocamera_weights = stereocamera_weights
        self.constraint_var = constraint_var # will actually be infinity

        # Create marker weights using the correct OpenSim API pattern
        # Give markers higher weight to prioritize marker data over IMU orientations
        self.webcamMarkerWeights = osim.SetMarkerWeights()
        for i, label in enumerate(self.webcamMarkerLabels):
            markerWeight = osim.MarkerWeight()
            markerWeight.setName(str(label))
            markerWeight.setWeight(self.webcam_weights[i])
            self.webcamMarkerWeights.cloneAndAppend(markerWeight)

        # Create orientation weights
        self.orientationWeights = osim.OrientationWeightSet()
        for i, label in enumerate(self.orientationLabels):
            orientationWeight = osim.OrientationWeight()
            orientationWeight.setName(str(label))
            orientationWeight.setWeight(self.orientation_weights[i])
            self.orientationWeights.cloneAndAppend(orientationWeight)

        # Create stereocamera weights
        self.stereocameraMarkerWeights = osim.SetMarkerWeights()
        for i, label in enumerate(self.stereocameraMarkerLabels):
            markerWeight = osim.MarkerWeight()
            markerWeight.setName(str(label))
            markerWeight.setWeight(self.stereocamera_weights[i])
            self.stereocameraMarkerWeights.cloneAndAppend(markerWeight)

        print(f"Webcam marker weight: {self.webcam_weights}")
        print(f"IMU orientation weight: {self.orientation_weights}")
        print(f"Stereocamera marker weight: {self.stereocamera_weights}")


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
        
        heading_rotation = osim.Rotation()
        # Apply all three rotation components
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
        
        ## NOT SURE WE NEED THIS PART
        # If correction is too large (> 0.3 radians ~17 degrees), apply only partial correction
        # max_correction = 0.1  # Maximum correction in radians (~17 degrees)
        # if correction_magnitude > max_correction:
        #     scale_factor = max_correction / correction_magnitude
        #     print(f"Large heading correction detected ({correction_magnitude*180/pi:.1f}°), applying scaled correction ({scale_factor:.2f})")
            
        #     # Scale down each component
        #     scaled_heading_rotation = osim.Rotation()
        #     scaled_heading_rotation.setRotationFromAngleAboutAxis(
        #         heading_rotation_vec3.get(0) * scale_factor, osim.CoordinateAxis(0))
        #     scaled_temp_rotation_y = osim.Rotation()
        #     scaled_temp_rotation_y.setRotationFromAngleAboutAxis(
        #         heading_rotation_vec3.get(1) * scale_factor, osim.CoordinateAxis(1))
        #     scaled_temp_rotation_z = osim.Rotation()
        #     scaled_temp_rotation_z.setRotationFromAngleAboutAxis(
        #         heading_rotation_vec3.get(2) * scale_factor, osim.CoordinateAxis(2))
            
        #     # Apply scaled corrections
        #     osim.OpenSenseUtilities.rotateOrientationTable(quatTable, scaled_heading_rotation)  # X rotation
        #     osim.OpenSenseUtilities.rotateOrientationTable(quatTable, scaled_temp_rotation_y)  # Y rotation  
        #     osim.OpenSenseUtilities.rotateOrientationTable(quatTable, scaled_temp_rotation_z)  # Z rotation
            
        #     print(f"Applied scaled heading correction: X={heading_rotation_vec3.get(0)*scale_factor*180/pi:.1f}°, "
        #         f"Y={heading_rotation_vec3.get(1)*scale_factor*180/pi:.1f}°, "
        #         f"Z={heading_rotation_vec3.get(2)*scale_factor*180/pi:.1f}°")
        # else:
        #     # Apply full correction for small corrections
        #     osim.OpenSenseUtilities.rotateOrientationTable(quatTable, heading_rotation)  # X rotation
        #     osim.OpenSenseUtilities.rotateOrientationTable(quatTable, temp_rotation_y)  # Y rotation  
        #     osim.OpenSenseUtilities.rotateOrientationTable(quatTable, temp_rotation_z)  # Z rotation
        #     print(f"Applied full heading correction: X={heading_rotation_vec3.get(0)*180/pi:.1f}°, "
        #         f"Y={heading_rotation_vec3.get(1)*180/pi:.1f}°, "
        #         f"Z={heading_rotation_vec3.get(2)*180/pi:.1f}°")
        orientationData = osim.OpenSenseUtilities.convertQuaternionsToRotations(quatTable)

        self.orientationQuatTable = orientationData
    
    def camera_correction(self, camera_type):
        if camera_type == "webcam":
            # Apply rotation to webcam markers
            rotX = osim.Rotation()
            rotX.setRotationFromAngleAboutAxis(self.webcam_to_opensim_rotation[0], osim.CoordinateAxis(0))
            rotY = osim.Rotation() 
            rotY.setRotationFromAngleAboutAxis(self.webcam_to_opensim_rotation[1], osim.CoordinateAxis(1))
            rotZ = osim.Rotation()
            rotZ.setRotationFromAngleAboutAxis(self.webcam_to_opensim_rotation[2], osim.CoordinateAxis(2))

            osim.OpenSenseUtilities.rotateMarkerTable(self.webcamMarkerTable, rotX)
            osim.OpenSenseUtilities.rotateMarkerTable(self.webcamMarkerTable, rotY)
            osim.OpenSenseUtilities.rotateMarkerTable(self.webcamMarkerTable, rotZ)

            print(f"Applied webcam to OpenSim rotations: {self.webcam_to_opensim_rotation[0]*180/pi:.1f}, {self.webcam_to_opensim_rotation[1]*180/pi:.1f}, {self.webcam_to_opensim_rotation[2]*180/pi:.1f} degrees")
        elif camera_type == "stereocamera":
            # Apply rotation to stereocamera markers
            rotX = osim.Rotation()
            rotX.setRotationFromAngleAboutAxis(self.stereocamera_to_opensim_rotation[0], osim.CoordinateAxis(0))
            rotY = osim.Rotation() 
            rotY.setRotationFromAngleAboutAxis(self.stereocamera_to_opensim_rotation[1], osim.CoordinateAxis(1))
            rotZ = osim.Rotation()
            rotZ.setRotationFromAngleAboutAxis(self.stereocamera_to_opensim_rotation[2], osim.CoordinateAxis(2))

            osim.OpenSenseUtilities.rotateMarkerTable(self.stereocameraMarkerTable, rotX)
            osim.OpenSenseUtilities.rotateMarkerTable(self.stereocameraMarkerTable, rotY)
            osim.OpenSenseUtilities.rotateMarkerTable(self.stereocameraMarkerTable, rotZ)

            print(f"Applied stereocamera to OpenSim rotations: {self.stereocamera_to_opensim_rotation[0]*180/pi:.1f}, {self.stereocamera_to_opensim_rotation[1]*180/pi:.1f}, {self.stereocamera_to_opensim_rotation[2]*180/pi:.1f} degrees")
        else:
            raise ValueError(f"Invalid camera type: {camera_type}. Expected 'webcam' or 'stereocamera'.")
        
    def downsampling(self):
        # Downsampling the orientation data to match marker timestamps test
        imu_times = self.orientationQuatTable.getIndependentColumn()
        webcam_times = self.webcamMarkerTable.getIndependentColumn()
        stereocamera_times = self.stereocameraMarkerTable.getIndependentColumn()

        print(f"IMU times: {len(imu_times)} samples")
        print(f"Webcam times: {len(webcam_times)} samples")
        print(f"Stereocamera times: {len(stereocamera_times)} samples")

        # Create a new table for downsampled orientations
        downsampled_orientations = osim.TimeSeriesTableQuaternion()
        downsampled_orientations.setColumnLabels(self.orientationQuatTable.getColumnLabels())
        
        # Iterate through shortest timestamps and find closest IMU orientation and closest Marker positions
        if len(webcam_times) <= len(stereocamera_times):
            # Create a new table for downsampled stereocamera markers
            downsampled_stereocamera = osim.TimeSeriesTableVec3()
            downsampled_stereocamera.setColumnLabels(self.stereocameraMarkerTable.getColumnLabels())

            for marker_time in webcam_times:
                closest_time_o = min(imu_times, key=lambda t: abs(t - marker_time))
                row_o = self.orientationQuatTable.getRowAtIndex(self.orientationQuatTable.getNearestRowIndexForTime(closest_time_o))
                downsampled_orientations.appendRow(marker_time, row_o)
                closest_time_s = min(stereocamera_times, key=lambda t: abs(t - marker_time))
                row_s = self.stereocameraMarkerTable.getRowAtIndex(self.stereocameraMarkerTable.getNearestRowIndexForTime(closest_time_s))
                downsampled_stereocamera.appendRow(marker_time, row_s)
            # Replace the original table with the downsampled one
            self.stereocameraMarkerTable = downsampled_stereocamera
        else:
            # Create a new table for downsampled webcam markers
            downsampled_webcam = osim.TimeSeriesTableVec3()
            downsampled_webcam.setColumnLabels(self.webcamMarkerTable.getColumnLabels())

            for marker_time in stereocamera_times:
                closest_time_o = min(imu_times, key=lambda t: abs(t - marker_time))
                row_o = self.orientationQuatTable.getRowAtIndex(self.orientationQuatTable.getNearestRowIndexForTime(closest_time_o))
                downsampled_orientations.appendRow(marker_time, row_o)
                closest_time_w = min(webcam_times, key=lambda t: abs(t - marker_time))
                row_w = self.webcamMarkerTable.getRowAtIndex(self.webcamMarkerTable.getNearestRowIndexForTime(closest_time_w))
                downsampled_webcam.appendRow(marker_time, row_w)
            # Replace the original table with the downsampled one
            self.webcamMarkerTable = downsampled_webcam

        # Replace quatTable with the downsampled orientations
        self.orientationQuatTable = downsampled_orientations

        print(f"Downsampling complete: {self.orientationQuatTable.getNumRows()} rows for orientation data, {self.stereocameraMarkerTable.getNumRows()} rows for stereocamera data, {self.webcamMarkerTable.getNumRows()} rows for webcam data")

    def load_references(self, max_orientation_weight, max_webcam_weight, max_stereocamera_weight):
        # Create OrientationsReference 
        if max_orientation_weight == 0:
            # Create an empty OrientationsReference to completely disable orientation constraints
            print("Creating EMPTY OrientationsReference to disable all IMU orientation constraints")
            self.oRefs = osim.OrientationsReference()  # Empty reference - no orientation data
        else:
            self.oRefs = osim.OrientationsReference(self.orientationQuatTable, self.orientationWeights) 
            print(f"OrientationsReference created")
        # Create MarkersReference from loaded marker data (following C++ InverseKinematicsTool pattern)
        if max_webcam_weight == 0 and max_stereocamera_weight == 0:
            # Create an empty MarkersReference to completely disable marker constraints
            print("Creating EMPTY MarkersReference to disable all marker constraints")
            self.mRefs = osim.MarkersReference()  # Empty reference - no marker data
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
            
            # Create combined marker weights
            combined_marker_weights = osim.SetMarkerWeights()
            for i, label in enumerate(self.webcamMarkerLabels):
                markerWeight = osim.MarkerWeight()
                markerWeight.setName(str(label))
                markerWeight.setWeight(self.webcam_weights[i])
                combined_marker_weights.cloneAndAppend(markerWeight)
            for i, label in enumerate(self.stereocameraMarkerLabels):
                markerWeight = osim.MarkerWeight()
                markerWeight.setName(str(label))
                markerWeight.setWeight(self.stereocamera_weights[i])
                combined_marker_weights.cloneAndAppend(markerWeight)
            # Create the combined MarkersReference
            self.mRefs = osim.MarkersReference(self.combined_marker_table, combined_marker_weights)
            print(f"MarkersReference created by combining webcam and stereocamera markers")

def main():
    # Put log to level debug and show in terminal
    #osim.Logger.setLevel(osim.Logger.Level_Debug)

    subject_ID = input("Enter the subject ID: ")
    trial_ID = input("Enter the trial ID (movement name): ")
    subject_mass = input("Enter the subject mass (kg): ")
    subject_height = input("Enter the subject height (mm): ")
    subject_age = input("Enter the subject age (years): ")
    subject_sex = input("Enter the subject sex (M/F): ")
    model_path = 'OpenSim/Models/Rajagopal/Rajagopal_2015.osim'
    model_name = 'Rajagopal'
    imu_to_opensim_rotation = Vec3(-pi/2, 39.5*pi/180, 0)  # Added +40 degrees to counter -40 degree pelvis rotation
    webcam_to_opensim_rotation = Vec3(140*pi/180, 0, 0)  # Webcam rotation to OpenSim world frame
    stereocamera_to_opensim_rotation = Vec3(0, 0, 0)  # Stereocamera rotation to OpenSim world frame
    sensor_fusion = OpenSimSensorFusion(model_path, model_name, subject_ID, trial_ID, imu_to_opensim_rotation, webcam_to_opensim_rotation, stereocamera_to_opensim_rotation, subject_mass, subject_height, subject_age, subject_sex)
    sensor_fusion.resultsDirectory = 'FusionIKResults'

    # Get IMU data
    orientationsFileName = sensor_fusion.get_imu_data()
    imuPlacer = osim.IMUPlacer() 
    sensor_fusion.orientationQuatTable = osim.TimeSeriesTableQuaternion(orientationsFileName) 

    # Get webcam marker data
    sensor_fusion.get_webcam_data()

    # Get stereocamera marker data
    sensor_fusion.get_stereocamera_data()

    # Set weights for sensor fusion
    # Webcam marker weight, IMU orientation weight, Stereocamera marker weight, Constraint variable
    #webcam_weights = [10, 10, 10, 10, 1, 1, 1, 1, 0, 0, 10, 10]
    #orientation_weights = [100, 100, 100, 100, 100, 100, 10, 10]
    #stereocamera_weights = [50, 50, 50, 50, 50, 50, 50, 5, 5, 5, 5, 5, 50, 50, 50]
    webcam_weights = [1 for _ in range(12)] 
    orientation_weights = [0 for _ in range(8)]  # No IMU orientation weight
    stereocamera_weights = [0 for _ in range(15)]  # No Stereocamera marker weight
    constraint_var = 1000  #low for fusion
    sensor_fusion.set_weights(webcam_weights, orientation_weights, stereocamera_weights, constraint_var)

    max_webcam_weight = max(webcam_weights)  # Use the minimum weight for all markers
    max_orientation_weight = max(orientation_weights)  # Use the minimum weight for all orientations
    max_stereocamera_weight = max(stereocamera_weights)  # Use the minimum weight for all stereocamera markers

    # Calibrate the model
    calibrated_model_path = sensor_fusion.calibrate_model(orientationsFileName)
    print(f"Calibrated model saved to: {calibrated_model_path}")

    # Time synchronization and resampling
    sensor_fusion.downsampling()

    # Correct IMU orientations
    sensor_fusion.heading_correction_imu_data(sensor_fusion.orientationQuatTable)
   
    
    # Load the orientation and marker references
    sensor_fusion.load_references(max_orientation_weight, max_webcam_weight, max_stereocamera_weight)
    

    # Create the solver
    coordinateReferences = osim.SimTKArrayCoordinateReference()
    if sensor_fusion.constraint_var < 10000:
        ikSolver = osim.InverseKinematicsSolver(sensor_fusion.model, sensor_fusion.mRefs, sensor_fusion.oRefs, coordinateReferences, sensor_fusion.constraint_var) 
    else : 
        # No constraint variable means Infinite weight, so the joint limits ahev to be applied
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
    # Realize position to ensure model is properly initialized (matching C++ line 94)
    sensor_fusion.model.realizePosition(sensor_fusion.s)

    # Define the overlapping time range for combined mode
    # Get the actual time range - use overlapping time range for combined mode
    times = sensor_fusion.orientationQuatTable.getIndependentColumn() 
    
    if (max_webcam_weight > 0 or max_orientation_weight > 0) and max_orientation_weight > 0:
        # since we downsampled the orientation data to match webcam markers, we can use the webcam marker time range
        webcam_marker_times = sensor_fusion.webcamMarkerTable.getIndependentColumn()
        startTime = webcam_marker_times[0]  # Use full webcam marker range
        endTime = webcam_marker_times[-1]   # Use full webcam marker range  
    else:
        if (max_webcam_weight > 0 or max_orientation_weight > 0):
            print("Using Markers mode only (IMU orientation weight is zero)") 
            # Use the full IMU time range as the primary time grid for IMU-only mode
            startTime = times[0]  # Use full IMU range
            endTime = times[-1]   # Use full IMU range
        elif max_orientation_weight > 0:
            print("Using IMU Orientation mode only (Webcam and Stereocamera marker weight is zero)") 
            # Use the full webcam marker time range as the primary time grid for Webcam-only mode
            webcam_marker_times = sensor_fusion.webcamMarkerTable.getIndependentColumn()
            startTime = webcam_marker_times[0]  # Use full webcam marker range
            endTime = webcam_marker_times[-1]   # Use full webcam marker range

    numTimeSteps = len(times) 
    print(f"Data time range: {startTime} to {endTime} seconds") 
    print(f"Number of time steps: {numTimeSteps}") 


    # For saving the results
    directory = orientationsFileName.rpartition('/')[0] 
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


    # Initialize the solver at the first time point (matching C++ implementation)
    # Use the first time from our filtered times array
    first_time = times[0]
    sensor_fusion.s.setTime(first_time) 

    # Realize position before assembly (important for stability)
    sensor_fusion.model.realizePosition(sensor_fusion.s)
    
    # Assemble the model at the first time point (matching C++ line 243)
    ikSolver.assemble(sensor_fusion.s)   # Only assemble once at the beginning
    print(f"IK Solver initialized and assembled. Starting processing...") 

    
    # Start timing
    start_time = time.time() 
    
    # Create storage objects for error tracking (similar to OpenSim C++ implementation)
    marker_errors_storage = None
    orientation_errors_storage = None
    combined_errors_storage = None
    
    # Initialize marker error storage if markers are used
    if max_webcam_weight > 0 or max_stereocamera_weight > 0:
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
    if max_orientation_weight > 0:
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
    
        # === COMPUTE AND STORE ERRORS (following OpenSim C++ pattern) ===
        
        # Marker errors
        webcam_total_squared_error = 0.0
        webcam_rms = 0.0
        webcam_max = 0.0
        stereocamera_total_squared_error = 0.0
        stereocamera_rms = 0.0
        stereocamera_max = 0.0
        individual_marker_errors = []
        
        if max_webcam_weight > 0 or max_stereocamera_weight > 0:
            try:
                marker_errors = osim.SimTKArrayDouble()
                ikSolver.computeCurrentSquaredMarkerErrors(marker_errors)
                
                if marker_errors.size() > 0:
                    num_webcam_markers = len(sensor_fusion.webcam_weights) - 1
                    num_stereocamera_markers = len(sensor_fusion.stereocamera_weights) - 1 
                    max_webcam_squared_error = 0.0
                    max_stereocamera_squared_error = 0.0
                    
                    # Calculate statistics following OpenSim C++ implementation
                    for j in range(num_webcam_markers):
                        webcam_squared_error = marker_errors.getElt(j)
                        webcam_total_squared_error += webcam_squared_error
                        individual_marker_errors.append(math.sqrt(webcam_squared_error))
                        
                        if webcam_squared_error > max_webcam_squared_error:
                            max_webcam_squared_error = webcam_squared_error

                    for j in range(num_webcam_markers, num_webcam_markers + num_stereocamera_markers):
                        stereocamera_squared_error = marker_errors.getElt(j)
                        stereocamera_total_squared_error += stereocamera_squared_error
                        individual_marker_errors.append(math.sqrt(stereocamera_squared_error))
                        
                        if stereocamera_squared_error > max_stereocamera_squared_error:
                            max_stereocamera_squared_error = stereocamera_squared_error
                    
                    webcam_rms = math.sqrt(webcam_total_squared_error / num_webcam_markers) if num_webcam_markers > 0 else 0
                    webcam_max = math.sqrt(max_webcam_squared_error)
                    stereocamera_rms = math.sqrt(stereocamera_total_squared_error / num_stereocamera_markers) if num_stereocamera_markers > 0 else 0
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
        
        # Orientation errors
        orientation_total_squared_error = 0.0
        orientation_rms = 0.0
        orientation_max = 0.0
        num_orientations = 0
        individual_orientation_errors = []
        
        if max_orientation_weight > 0:
            try:
                orientation_errors = osim.SimTKArrayDouble()
                ikSolver.computeCurrentOrientationErrors(orientation_errors)
                
                if orientation_errors.size() > 0:
                    num_orientations = orientation_errors.size()
                    max_squared_error = 0.0
                    
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
        if marker_errors_storage and max_webcam_weight > 0:
            marker_filename = f"{trial_name}_ik_marker_errors"
            osim.Storage.printResult(marker_errors_storage, marker_filename, resultsDirectory, -1, ".mot")
            print(f"✓ Saved marker errors to: {resultsDirectory}/{marker_filename}.mot")
        
        # Save orientation errors  
        if orientation_errors_storage and max_orientation_weight > 0:
            orientation_filename = f"{trial_name}_ik_orientation_errors"
            osim.Storage.printResult(orientation_errors_storage, orientation_filename, resultsDirectory, -1, ".mot")
            print(f"✓ Saved orientation errors to: {resultsDirectory}/{orientation_filename}.mot")
        
        # Save combined errors
        combined_filename = f"{trial_name}_ik_combined_errors"
        osim.Storage.printResult(combined_errors_storage, combined_filename, resultsDirectory, -1, ".mot")
        print(f"✓ Saved combined errors to: {resultsDirectory}/{combined_filename}.mot")
        
        # Print summary statistics (following OpenSim C++ logging pattern)
        
        if (max_webcam_weight > 0 or max_stereocamera_weight > 0 ):
            num_markers = num_webcam_markers + num_stereocamera_markers + 2 
            print(f"\n=== FINAL ERROR SUMMARY ===")
            print(f"Markers: {num_markers} tracked")
            print(f"Webcam marker weights:")
            for weight in sensor_fusion.webcamMarkerWeights:
                print(f"{weight.getWeight()}")
            print(f"Stereocamera marker weights:")
            for weight in sensor_fusion.stereocameraMarkerWeights:
                print(f"{weight.getWeight()}")
            print(f"  - Final webcam RMS error: {webcam_rms:.6f} m")
            print(f"  - Final webcam max error: {webcam_max:.6f} m")
            print(f"  - Total webcam squared error: {webcam_total_squared_error:.8f}")
            print(f"  - Final stereocamera RMS error: {stereocamera_rms:.6f} m")
            print(f"  - Final stereocamera max error: {stereocamera_max:.6f} m")
            print(f"  - Total stereocamera squared error: {stereocamera_total_squared_error:.8f}")

        if max_orientation_weight > 0 and num_orientations > 0:
            print(f"Orientations: {num_orientations} tracked")  
            print(f"Orientation weights:")
            for weight in sensor_fusion.orientationWeights:
                print(f"{weight.getWeight()}")
            print(f"  - Final RMS error: {orientation_rms*180/pi:.4f}°")
            print(f"  - Final max error: {orientation_max*180/pi:.4f}°")
            print(f"  - Total squared error: {orientation_total_squared_error:.8f}")
            
        print(f"Total final cost: {total_cost:.8f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error saving error files: {e}")

    # Save main results as .mot file
    motFileName = resultsDirectory + "/inverse_kinematics_results.mot"
    storage.printResult(storage, "inverse_kinematics_results", resultsDirectory, -1, ".mot")
    print(f"Results saved to: {motFileName}")
    
    print("Script execution completed successfully.")
    
    # Exit immediately to prevent segmentation fault during OpenSim object destruction
    os._exit(0)



if __name__ == "__main__":
    main()