from OpenSense_CalibrateModel import main as calibrate_model
from OpenSense_IMUDataConverter import main as convert_imu_data
from OpenSense_OrientationTracking import main as track_orientation

def main():
    model_path = 'OpenSense/Models/Rajagopal/Rajagopal_2015.osim'
    model_name = 'Rajagopal'

    # Convert IMU data
    subject_ID = input("Enter the subject ID: ")
    trial_ID = input("Enter the trial ID (movement name): ")
    orientation_file = convert_imu_data(subject_ID, trial_ID)

    # Calibrate the model
    calibrated_model_path = calibrate_model(model_path, model_name, orientation_file, subject_ID, trial_ID)

    # Track orientation
    track_orientation(calibrated_model_path, orientation_file)

if __name__ == "__main__":
    main()