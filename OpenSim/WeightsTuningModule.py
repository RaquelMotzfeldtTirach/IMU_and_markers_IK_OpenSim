from SensorFusion_automatic import main as sensor_fusion
import os
import shutil
from diff_check_mot import compare_mot_files

def main():

    # Get inputs from user
    subject_ID = input("Enter the subject ID: ")
    trial_ID = input("Enter the trial ID (movement name): ")
    subject_mass = input("Enter the subject mass (kg): ")
    subject_height = input("Enter the subject height (mm): ")
    subject_age = input("Enter the subject age (years): ")
    subject_sex = input("Enter the subject sex (M/F): ")

    # Default weights
    # right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, right pinky, left pinky, right index, left index, right hip, left hip
    webcam_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    # torso, pelvis, upper right, lower right, upper left, lower left, hand right, hand left
    orientation_weights = [1, 1, 1, 1, 1, 1, 1, 1] 
    # neck, right clavicle, left clavicle, right shoulder, left shoulder, right elbow, left elbow, left wrist, right wrist, spine 3, spine 2, spine 1, pelvis, right hip, left hip
    stereocamera_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    constraint_var = 1000

    # Easy experiment: vary orientation weights only, one by one
    # output saved in new folder
    folder_path = f'recordings/subject{subject_ID}/tuning_{trial_ID}'
    os.makedirs(folder_path, exist_ok=True)

    output_original = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
    # copy output file to new folder
    shutil.copy(output_original, folder_path + f'/output_original.mot')


    for i in range(len(orientation_weights)):
        orientation_weights[i] = 0.1
        output = sensor_fusion(webcam_weights, orientation_weights, stereocamera_weights, constraint_var, subject_ID, trial_ID, subject_mass, subject_height, subject_age, subject_sex)
        # copy output file to new folder
        shutil.copy(output, folder_path + f'/output_{i}.mot')

    for i in range(len(orientation_weights)):
        compare_mot_files(folder_path + f'/output_original.mot', folder_path + f'/output_{i}.mot')

if __name__ == "__main__":
    main()