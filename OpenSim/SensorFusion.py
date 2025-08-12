# TODO:
# clean the code
# make the code more modular
# add support for another camera and set of markers
# make the weights configurable
# make the weights individual for each marker and imu

import opensim as osim
from opensim import Vec3
import numpy as np
import math
from math import pi
import argparse
import os
import time
from OpenSense_CalibrateModel import main as calibrate_model
from OpenSense_IMUDataConverter import main as convert_imu_data

class OpenSimSensorFusion:
    def __init__(self, model_path, model_name, subject_ID, trial_ID, sensor_to_opensim_rotation):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.s = None
        self.subject_ID = subject_ID
        self.trial_ID = trial_ID
        self.webcam_marker_weight = None
        self.imu_orientation_weight = None
        self.stereocamera_marker_weight = None
        self.constraint_var = None
        self.imu_to_opensim_rotation = sensor_to_opensim_rotation
        self.base_heading_axis = "-z"  
        self.resultsDirectory = None
        self.webcamMarkerTable = None
        self.webcamMarkerLabels = None
        self.orientationQuatTable = None
        self.webcamMarkersFileName = None
        self.orientationLabels = None

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
        
            

    def calibrate_model(self, orientation_file): 
        # IMU calibration
        calibrated_model_path = calibrate_model(self.model_path, self.model_name, orientation_file, self.subject_ID, self.trial_ID)
        # Webcam scaling and marker placement
        scale_tool = osim.ScaleTool("OpenSim/scaling_setup.xml")
        generic_model_maker = scale_tool.getGenericModelMaker()
        print("Marker Set File Name:", generic_model_maker.getMarkerSetFileName())
        static_trial = scale_tool.getModelScaler()
        print("Model Scaler File Name:", static_trial.getMarkerFileName())
        model_path = scale_tool.getGenericModelMaker().getModelFileName()
        print("Model File Name:", model_path)
        scale_tool.run()

        self.model = osim.Model(calibrated_model_path)
        self.s = self.model.initSystem()  

        return calibrated_model_path

    def set_weights(self, webcam_marker_weight, imu_orientation_weight, stereocamera_marker_weight, constraint_var):
        self.webcam_marker_weight = webcam_marker_weight
        self.imu_orientation_weight = imu_orientation_weight
        self.stereocamera_marker_weight = stereocamera_marker_weight
        self.constraint_var = constraint_var

        # Create marker weights using the correct OpenSim API pattern
        # Give markers higher weight to prioritize marker data over IMU orientations
        self.webcamMarkerWeights = osim.SetMarkerWeights()
        for i, label in enumerate(self.webcamMarkerLabels):
            markerWeight = osim.MarkerWeight()
            markerWeight.setName(str(label))
            ## TO CHANGE LATER
            if i == len(self.webcamMarkerLabels) - 1:  # Last marker gets different weight
                markerWeight.setWeight(2)  # Different weight for last marker
            else:
                markerWeight.setWeight(self.webcam_marker_weight)  # Use configured marker weight
            self.webcamMarkerWeights.cloneAndAppend(markerWeight)

        # Create orientation weights
        self.orientationWeights = osim.OrientationWeightSet()
        for i, label in enumerate(self.orientationLabels):
            orientationWeight = osim.OrientationWeight()
            orientationWeight.setName(str(label))
            ## TO CHANGE LATER
            if i == len(self.orientationLabels) - 1:  # Last orientation gets different weight
                orientationWeight.setWeight(0)  # Different weight for last orientation
            else:
                orientationWeight.setWeight(self.imu_orientation_weight)  # Use configured orientation weight
            self.orientationWeights.cloneAndAppend(orientationWeight)

        print(f"Webcam marker weight: {self.webcam_marker_weight}")
        print(f"IMU orientation weight: {self.imu_orientation_weight}")

        ## WEIGHTS HAVE TO BE DIFFERENT WITHIN A SAME SET FOR THE SOLVER TO TAKE THE WEIGHTS INTO ACCOUNT


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

    def downsampling(self):
        # Downsampling the orientation data to match marker timestamps test
        imu_times = self.orientationQuatTable.getIndependentColumn()
        marker_times = self.webcamMarkerTable.getIndependentColumn()
        
        print(f"IMU times: {len(imu_times)} samples")
        print(f"Marker times: {len(marker_times)} samples")
        
        # Create a new table for downsampled orientations
        downsampled_orientations = osim.TimeSeriesTableQuaternion()
        downsampled_orientations.setColumnLabels(self.orientationQuatTable.getColumnLabels())
        
        # Iterate through marker timestamps and find closest IMU orientation
        for marker_time in marker_times:
            closest_time = min(imu_times, key=lambda t: abs(t - marker_time))
            row = self.orientationQuatTable.getRowAtIndex(self.orientationQuatTable.getNearestRowIndexForTime(closest_time))
            downsampled_orientations.appendRow(marker_time, row)
        
        # Replace quatTable with the downsampled orientations
        self.orientationQuatTable = downsampled_orientations
        
        print(f"Downsampling complete: {self.orientationQuatTable.getNumRows()} rows")


def main():
    # Put log to level debug and show in terminal
    #osim.Logger.setLevel(osim.Logger.Level_Debug)

    subject_ID = input("Enter the subject ID: ")
    trial_ID = input("Enter the trial ID (movement name): ")
    model_path = 'OpenSim/Models/Rajagopal/Rajagopal_2015.osim'
    model_name = 'Rajagopal'
    sensor_to_opensim_rotation = Vec3(-pi/2, 52*pi/180, 0)  # The rotation of IMU data to the OpenSim world frame
    sensor_fusion = OpenSimSensorFusion(model_path, model_name, subject_ID, trial_ID, sensor_to_opensim_rotation)
    sensor_fusion.resultsDirectory = 'FusionIKResults'

    # Get IMU data
    orientationsFileName = sensor_fusion.get_imu_data()
    imuPlacer = osim.IMUPlacer() 
    sensor_fusion.orientationQuatTable = osim.TimeSeriesTableQuaternion(orientationsFileName) 

    # Get webcam marker data
    sensor_fusion.get_webcam_data()

    # Set weights for sensor fusion
    # Webcam marker weight, IMU orientation weight, Stereocamera marker weight, Constraint variable
    sensor_fusion.set_weights(1,0,1,1)

    # Calibrate the model
    calibrated_model_path = sensor_fusion.calibrate_model(orientationsFileName)
    print(f"Calibrated model saved to: {calibrated_model_path}")

    # Time synchronization and resampling
    sensor_fusion.downsampling()

    # Correct IMU orientations
    sensor_fusion.heading_correction_imu_data(sensor_fusion.orientationQuatTable)
    

    # Load the orientation and marker references
    # Create OrientationsReference 
    if sensor_fusion.imu_orientation_weight == 0:
        # Create an empty OrientationsReference to completely disable orientation constraints
        print("Creating EMPTY OrientationsReference to disable all IMU orientation constraints")
        oRefs = osim.OrientationsReference()  # Empty reference - no orientation data
    else:
        oRefs = osim.OrientationsReference(sensor_fusion.orientationQuatTable, sensor_fusion.orientationWeights) 
        print(f"OrientationsReference created")
    # Create MarkersReference from loaded marker data (following C++ InverseKinematicsTool pattern)
    if sensor_fusion.webcam_marker_weight == 0:
        # Create an empty MarkersReference to completely disable marker constraints
        print("Creating EMPTY MarkersReference to disable all marker constraints")
        mRefs = osim.MarkersReference()  # Empty reference - no marker data
    else:
        mRefs = osim.MarkersReference()
        mRefs.initializeFromMarkersFile(sensor_fusion.webcamMarkersFileName, sensor_fusion.webcamMarkerWeights)
        print(f"MarkersReference created")
       

        
    # Create the solver
    coordinateReferences = osim.SimTKArrayCoordinateReference()
    ikSolver = osim.InverseKinematicsSolver(sensor_fusion.model, mRefs, oRefs, coordinateReferences) # No constraint variable means Infinite weight, so the joint limits ahev to be applied
    accuracy = 1e-9  # Same as C++ implementation, probably default value
    ikSolver.setAccuracy(accuracy)
    print(f"\n=== SOLVER CONFIGURATION ANALYSIS ===")
    # Check what the solver actually sees
    print(f"Solver configuration:")
    print(f"  - Markers reference: {mRefs.getNumFrames() if hasattr(mRefs, 'getNumFrames') else 'unknown'} frames")
    print(f"  - Orientations reference: {len(oRefs.getTimes()) if hasattr(oRefs, 'getTimes') else 'unknown'} frames")
    print("="*60)

    
    # Initialize the model  
    # Realize position to ensure model is properly initialized (matching C++ line 94)
    sensor_fusion.model.realizePosition(sensor_fusion.s)

    # Define the overlapping time range for combined mode
    # Get the actual time range - use overlapping time range for combined mode
    times = sensor_fusion.orientationQuatTable.getIndependentColumn() 
    
    if sensor_fusion.webcam_marker_weight > 0 and sensor_fusion.imu_orientation_weight > 0:
        # For combined mode, use the overlapping time range to avoid extrapolation errors
        webcam_marker_times = sensor_fusion.webcamMarkerTable.getIndependentColumn()
        webcam_marker_start_time = webcam_marker_times[0]
        webcam_marker_end_time = webcam_marker_times[-1]
        
        orientation_start_time = times[0]
        orientation_end_time = times[-1]
        
        # Use the overlapping time range
        startTime = max(webcam_marker_start_time, orientation_start_time)  # Start when both data sources are available
        endTime = min(webcam_marker_end_time, orientation_end_time)        # End when either data source ends
        
        print(f"Using combined IMU + Webcam Marker mode:")
        print(f"  - Overlap time range: {startTime:.4f} to {endTime:.4f} seconds")
        print(f"  - Duration: {endTime - startTime:.2f} seconds")
        
        # Filter IMU times to the overlapping range
        times_filtered = [t for t in times if startTime <= t <= endTime]
        times = times_filtered
        print(f"  - Using {len(times)} IMU frames within overlap period")
        
    else:
        if sensor_fusion.webcam_marker_weight > 0:
            print("Using Webcam Marker mode only (IMU orientation weight is zero)") 
            # Use the full IMU time range as the primary time grid for IMU-only mode
            startTime = times[0]  # Use full IMU range
            endTime = times[-1]   # Use full IMU range
            print(f"Using IMU-only mode: {startTime:.4f} to {endTime:.4f} seconds")
        elif sensor_fusion.imu_orientation_weight > 0:
            print("Using IMU Orientation mode only (Webcam marker weight is zero)") 
            # Use the full webcam marker time range as the primary time grid for Webcam-only mode
            webcam_marker_times = sensor_fusion.webcamMarkerTable.getIndependentColumn()
            startTime = webcam_marker_times[0]  # Use full webcam marker range
            endTime = webcam_marker_times[-1]   # Use full webcam marker range
            print(f"Using Webcam-only mode: {startTime:.4f} to {endTime:.4f} seconds")

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
    if sensor_fusion.webcam_marker_weight > 0:
        marker_errors_storage = osim.Storage()
        marker_errors_storage.setName("Model Marker Errors from IK")
        marker_errors_storage.setInDegrees(False)  # Errors are in meters
        
        # Set column labels following OpenSim C++ pattern
        marker_labels = osim.ArrayStr()
        marker_labels.append("time")
        marker_labels.append("total_squared_error")
        marker_labels.append("marker_error_RMS")
        marker_labels.append("marker_error_max")
        
        # Add individual marker error columns
        for label in sensor_fusion.webcamMarkerLabels:
            marker_labels.append(f"{str(label)}_error")
            
        marker_errors_storage.setColumnLabels(marker_labels)
        
    # Initialize orientation error storage if orientations are used
    if sensor_fusion.imu_orientation_weight > 0:
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
    combined_labels.append("marker_total_squared_error")
    combined_labels.append("orientation_total_squared_error")
    combined_labels.append("marker_rms_error")
    combined_labels.append("marker_max_error")
    combined_labels.append("orientation_rms_error_deg")
    combined_labels.append("orientation_max_error_deg")
    combined_labels.append("num_markers")
    combined_labels.append("num_orientations")
    
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
        marker_total_squared_error = 0.0
        marker_rms = 0.0
        marker_max = 0.0
        num_markers = 0
        individual_marker_errors = []
        
        if sensor_fusion.webcam_marker_weight > 0:
            try:
                marker_errors = osim.SimTKArrayDouble()
                ikSolver.computeCurrentSquaredMarkerErrors(marker_errors)
                
                if marker_errors.size() > 0:
                    num_markers = marker_errors.size()
                    max_squared_error = 0.0
                    
                    # Calculate statistics following OpenSim C++ implementation
                    for j in range(marker_errors.size()):
                        squared_error = marker_errors.getElt(j)
                        marker_total_squared_error += squared_error
                        individual_marker_errors.append(math.sqrt(squared_error))
                        
                        if squared_error > max_squared_error:
                            max_squared_error = squared_error
                    
                    marker_rms = math.sqrt(marker_total_squared_error / num_markers) if num_markers > 0 else 0
                    marker_max = math.sqrt(max_squared_error)
                    
                    # Save marker error data
                    if marker_errors_storage:
                        marker_data = osim.ArrayDouble()
                        marker_data.append(marker_total_squared_error)
                        marker_data.append(marker_rms)
                        marker_data.append(marker_max)
                        
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
        
        if sensor_fusion.imu_orientation_weight > 0:
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
        total_cost = marker_total_squared_error + orientation_total_squared_error
        
        combined_data = osim.ArrayDouble()
        combined_data.append(total_cost)
        combined_data.append(marker_total_squared_error)
        combined_data.append(orientation_total_squared_error)
        combined_data.append(marker_rms)
        combined_data.append(marker_max)
        combined_data.append(orientation_rms * 180.0 / pi if orientation_rms > 0 else 0.0)  # Convert to degrees
        combined_data.append(orientation_max * 180.0 / pi if orientation_max > 0 else 0.0)  # Convert to degrees
        combined_data.append(float(num_markers))
        combined_data.append(float(num_orientations))
        
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
        if marker_errors_storage and sensor_fusion.webcam_marker_weight > 0:
            marker_filename = f"{trial_name}_ik_marker_errors"
            osim.Storage.printResult(marker_errors_storage, marker_filename, resultsDirectory, -1, ".mot")
            print(f"✓ Saved marker errors to: {resultsDirectory}/{marker_filename}.mot")
        
        # Save orientation errors  
        if orientation_errors_storage and sensor_fusion.imu_orientation_weight > 0:
            orientation_filename = f"{trial_name}_ik_orientation_errors"
            osim.Storage.printResult(orientation_errors_storage, orientation_filename, resultsDirectory, -1, ".mot")
            print(f"✓ Saved orientation errors to: {resultsDirectory}/{orientation_filename}.mot")
        
        # Save combined errors
        combined_filename = f"{trial_name}_ik_combined_errors"
        osim.Storage.printResult(combined_errors_storage, combined_filename, resultsDirectory, -1, ".mot")
        print(f"✓ Saved combined errors to: {resultsDirectory}/{combined_filename}.mot")
        
        # Print summary statistics (following OpenSim C++ logging pattern)
        if sensor_fusion.webcam_marker_weight > 0 and num_markers > 0:
            print(f"\n=== FINAL ERROR SUMMARY ===")
            print(f"Markers: {num_markers} tracked")
            print(f"Marker weights:")
            for weight in sensor_fusion.webcamMarkerWeights:
                print(f"{weight.getWeight()}")
            print(f"  - Final RMS error: {marker_rms:.6f} m")
            print(f"  - Final max error: {marker_max:.6f} m")
            print(f"  - Total squared error: {marker_total_squared_error:.8f}")
            
        if sensor_fusion.imu_orientation_weight > 0 and num_orientations > 0:
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