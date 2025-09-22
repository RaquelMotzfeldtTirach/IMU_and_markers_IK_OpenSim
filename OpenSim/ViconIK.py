import opensim as osim
from opensim import Vec3
import numpy as np
import math
from math import pi
import argparse
import os
import shutil
import time
from modelScalingVicon import main as model_scaling_vicon

# TODO: 
# - Change marker xml file for vicon markers
# - Make scaling script for Vicon data
# - Make scaling template xml for vicon data
# - Add hand markers for vicon
# - Do time synchronization and downsampling in the weights tuning module!

class OpenSimVicon:
    def __init__(self, model_path, model_name, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
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
        self.constraint_var = None
        self.resultsDirectory = None
        self.viconMarkerTable = None
        self.viconMarkerLabels = None
        self.viconMarkersFileName = None
        self.viconMarkerWeights = None
        self.mRefs = None
        self.times = None 

    def get_vicon_data(self, vicon_path = None):
        # Set up the markers file
        if vicon_path is not None:
            self.viconMarkersFileName = vicon_path
        else:
            self.viconMarkersFileName = "recordings/subject" + str(self.subject_ID) + "/vicon_" + str(self.trial_ID) + ".trc"

        # Load marker data 
        self.viconMarkerTable = osim.TimeSeriesTableVec3(self.viconMarkersFileName)
        print(f"Loaded vicon marker data from: {self.viconMarkersFileName}")
        viconMarkerTimes = self.viconMarkerTable.getIndependentColumn()
        print(f"Vicon marker data time range: {viconMarkerTimes[0]:.4f} to {viconMarkerTimes[-1]:.4f} seconds")
        print(f"Number of markers: {self.viconMarkerTable.getNumColumns()}")
        print(f"Vicon marker data points: {len(viconMarkerTimes)}")

        # Show first few timestamps for debugging
        print(f"First 5 vicon marker timestamps: {[f'{t:.4f}' for t in viconMarkerTimes[:5]]}")
        print(f"Last 5 vicon marker timestamps: {[f'{t:.4f}' for t in viconMarkerTimes[-5:]]}")

        # Get marker names
        self.viconMarkerLabels = self.viconMarkerTable.getColumnLabels()
        print(f"Available vicon markers: {[str(label) for label in self.viconMarkerLabels]}")

        # Validate marker data for missing/invalid positions
        print("Validating vicon marker data...")
        nan_count = 0
        extreme_count = 0
        for i in range(self.viconMarkerTable.getNumRows()):
            row = self.viconMarkerTable.getRowAtIndex(i)
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
            print(f"WARNING: Found {extreme_count} extreme marker positions in vicon data (>10m from origin)")
            print("This could cause the billion-scale errors you're seeing!")
        elif nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN marker positions in vicon data")
        else:
            print("✓ All vicon marker positions appear reasonable")
        

    def calibrate_model(self): 
        calibrated_model_path = 'OpenSim/Models/Rajagopal/Calibrated_Rajagopal_subject' + str(self.subject_ID) +'_' + str(self.trial_ID) + '.osim'
        # Vicon scaling and marker placement
        calibrated_model_path = model_scaling_vicon(self.subject_ID, self.trial_ID, self.subject_mass, self.subject_height, self.subject_age, self.subject_sex, calibrated_model_path)

        self.model = osim.Model(calibrated_model_path)
        self.s = self.model.initSystem()  
        print("Model Mass:", self.model.getTotalMass(self.s))

        nb_markers = self.model.getMarkerSet().getSize()
        print(f"Model calibrated and scaled: {calibrated_model_path}")
        print(f"Number of markers in the model: {nb_markers}")

        return calibrated_model_path


    def load_references(self):
        # Create MarkersReference from vicon data only
        self.marker_table = self.viconMarkerTable
        self.mRefs = osim.MarkersReference(self.marker_table)
        print(f"MarkersReference created from vicon markers")


def main(constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex):
    # Put log to level debug and show in terminal
    osim.Logger.setLevel(osim.Logger.Level_Info)

    model_path = 'OpenSim/Models/Rajagopal/Rajagopal_2015.osim'
    model_name = 'Rajagopal'
    sensor_fusion = OpenSimVicon(model_path, model_name, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    sensor_fusion.resultsDirectory = '../'+ trial_ID +'_ViconIKResults'
    sensor_fusion.constraint_var = constraint_var

    # Get Vicon data
    sensor_fusion.get_vicon_data()

    # Calibrate the model
    calibrated_model_path = sensor_fusion.calibrate_model()
    print(f"Calibrated model saved to: {calibrated_model_path}")


    # Load the marker references
    sensor_fusion.load_references()
    

    # Create the solver
    coordinateReferences = osim.SimTKArrayCoordinateReference()
    if sensor_fusion.constraint_var < 10000:
        ikSolver = osim.InverseKinematicsSolver(sensor_fusion.model, sensor_fusion.mRefs, coordinateReferences, sensor_fusion.constraint_var) 
    else : 
        # No constraint variable means Infinite weight, so the joint limits have to be applied, this is the default value for the InverseKinematicsSolver
        ikSolver = osim.InverseKinematicsSolver(sensor_fusion.model, sensor_fusion.mRefs, coordinateReferences)


    accuracy = 1e-9  # Same as C++ implementation, probably default value
    ikSolver.setAccuracy(accuracy)
    print(f"\n=== SOLVER CONFIGURATION ANALYSIS ===")
    # Check what the solver actually sees
    print(f"Solver configuration:")
    print(f"  - Markers reference: {sensor_fusion.mRefs.getNumFrames() if hasattr(sensor_fusion.mRefs, 'getNumFrames') else 'unknown'} frames")
    print(f"  - Constraint variable: {sensor_fusion.constraint_var}")
    print(f"  - Accuracy: {accuracy}")
    print("="*60)


    # Initialize the model  
    # Realize position to ensure model is properly initialized
    sensor_fusion.model.realizePosition(sensor_fusion.s)

    # Define the overlapping time range for combined mode
    # Get the actual time range - use overlapping time range for combined mode
    times = sensor_fusion.viconMarkerTable.getIndependentColumn()
    startTime = times[0]  
    endTime = times[-1]

    numTimeSteps = len(times) 
    print(f"Data time range: {startTime} to {endTime} seconds") 
    print(f"Number of time steps: {numTimeSteps}") 


    ## For saving the results
    directory = sensor_fusion.viconMarkersFileName.rpartition('/')[0] 
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

    # Initialize vicon error storage 
    vicon_errors_storage = osim.Storage()
    vicon_errors_storage.setName("Model Vicon Errors from IK")
    vicon_errors_storage.setInDegrees(True)  # Convert radians to degrees

    # Set column labels
    vicon_labels = osim.ArrayStr()
    vicon_labels.append("time")
    vicon_labels.append("total_squared_error")
    vicon_labels.append("vicon_error_RMS")
    vicon_labels.append("vicon_error_max")

    # Add individual vicon error columns
    for label in sensor_fusion.viconLabels:
        vicon_labels.append(f"{str(label)}_error")

    vicon_errors_storage.setColumnLabels(vicon_labels)

    # Initialise nb of markers
    num_markers = len(sensor_fusion.viconMarkerLabels)


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
        vicon_total_squared_error = 0.0
        vicon_rms = 0.0
        vicon_max = 0.0
        individual_marker_errors = []
        max_vicon_squared_error = 0.0

        marker_errors = osim.SimTKArrayDouble()
        ikSolver.computeCurrentSquaredMarkerErrors(marker_errors)
        
        if marker_errors.size() > 0:
            
            # Validate that we have enough error values for all markers
            expected_markers = num_markers
            if marker_errors.size() < expected_markers:
                print(f"WARNING: Not enough marker errors! Expected {expected_markers}, got {marker_errors.size()}")
            
            # Calculate statistics 
            for j in range(min(num_markers, marker_errors.size())):
                vicon_squared_error = marker_errors.getElt(j)
                # Check for invalid/extreme values
                if vicon_squared_error < 0 or vicon_squared_error > 1e6:  # 1 million square meters is extreme
                    print(f"WARNING: Extreme Vicon marker error at index {j}: {vicon_squared_error}")
                    vicon_squared_error = min(vicon_squared_error, 1e6)  # Cap at reasonable value

                vicon_total_squared_error += vicon_squared_error
                individual_marker_errors.append(math.sqrt(vicon_squared_error))

                if vicon_squared_error > max_vicon_squared_error:
                    max_vicon_squared_error = vicon_squared_error
            
            # Calculate actual number of markers processed for RMS calculation
            vicon_rms = math.sqrt(vicon_total_squared_error / marker_errors.size()) if marker_errors.size() > 0 else 0
            vicon_max = math.sqrt(max_vicon_squared_error)

            # Save marker error data
            if marker_errors_storage:
                marker_data = osim.ArrayDouble()
                marker_data.append(vicon_total_squared_error)
                marker_data.append(vicon_rms)
                marker_data.append(vicon_max)

                # Add individual marker errors
                for error in individual_marker_errors:
                    marker_data.append(error)
                
                marker_errors_storage.append(time_val, marker_data)
                        
        
        
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
        if marker_errors_storage:
            marker_filename = f"{trial_name}_ik_marker_errors"
            osim.Storage.printResult(marker_errors_storage, marker_filename, resultsDirectory, -1, ".mot")
            print(f"✓ Saved marker errors to: {resultsDirectory}/{marker_filename}.mot")

        
        # Print summary statistics (following OpenSim C++ logging pattern)
        # Use the counts from the sensor fusion object, not the potentially undefined local variables
        print(f"\n=== FINAL ERROR SUMMARY ===")
        print(f"  - Final vicon RMS error: {vicon_rms:.6f} m")
        print(f"  - Final vicon max error: {vicon_max:.6f} m")
        print(f"  - Total vicon squared error: {vicon_total_squared_error:.8f}")
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
    parser = argparse.ArgumentParser(description="Inverse kinematics using OpenSim for Vicon data")
    parser.add_argument("constraint_var", type=float, help="Constraint variable (use a large number like 10000 for infinite weight)")
    parser.add_argument("subject_ID", type=str, help="Subject ID")
    parser.add_argument("trial_ID", type=str, help="Trial name")
    parser.add_argument("subject_mass", type=float, help="Subject mass in kg")
    parser.add_argument("subject_height", type=float, help="Subject height in mm")
    parser.add_argument("subject_age", type=int, help="Subject age in years")
    parser.add_argument("subject_sex", type=str, choices=['M', 'F'], help="Subject sex (M/F)")

    args = parser.parse_args()

    main(args.subject_ID, args.trial_ID, args.constraint_var, args.subject_mass, args.subject_height, args.subject_age, args.subject_sex)