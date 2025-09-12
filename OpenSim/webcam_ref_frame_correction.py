import numpy as np
import pandas as pd
import argparse

# For the webcam data it is quite simple
# The orientation of the data is the same as in OpenSim
# The origin (0, 0, 0) is between the two hip markers
# So we only need to translate all the data so that the hip middle point matches the hip middle point of the OpenSim model
# The OpenSim model hip middle point is at (-0.079247, 0.878088, 0.0) meters

def read_trc_file(filepath, columns_of_interest=None):
    """
    Reads a .trc file and returns a DataFrame with the specified columns of interest.
    """
    # Find where the actual data starts (after 'endheader')
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # For all .trc files, the data start on line 7
    header_end = 4
    # Column names are on line 4
    column_names = lines[3].strip().split('\t\t\t')
    column_names[0] = column_names[0].removeprefix('Frame#\tTime\t')
    
    # Read the data starting from after the header
    df = pd.read_csv(filepath, sep='\t', skiprows=header_end)
    # All columns should be put together as vectors of size 3
    # except for the first two columns
    first_two = df.iloc[:, :2]
    first_two.columns = ['Frames', 'Time']
    # Group the remaining columns into vectors of size 3 (X, Y, Z for each marker)
    marker_cols = df.iloc[:, 2:]
    n_markers = marker_cols.shape[1] // 3

    marker_data = {}
    for i in range(n_markers):
        marker_name = column_names[i]
        # For each marker, stack X, Y, Z into a list for each row
        marker_data[marker_name] = marker_cols.iloc[:, i*3:(i+1)*3].values.tolist()

    marker_df = pd.DataFrame(marker_data)
    # Combine the first two columns with the marker columns
    df = pd.concat([first_two, marker_df], axis=1)
    
    # Select only the columns of interest if specified
    if columns_of_interest:
        # Make sure 'time' is always included
        if 'Time' not in columns_of_interest:
            columns_of_interest = ['Time'] + columns_of_interest
        df = df[columns_of_interest]
    
    return df

def find_rotation_matrix(source, target):
    """
    Finds the rotation matrix that aligns the source vector with the target vector.
    """
    R = None
    # Compute the necessary rotation
    v_source = source / np.linalg.norm(source)
    v_target = target / np.linalg.norm(target)

    # Calculate the rotation axis and angle using cross product
    v = np.cross(v_source, v_target)
    c = np.dot(v_source, v_target)

    if np.allclose(v, 0):
        # If the vectors are already aligned, return the identity matrix
        R = np.eye(3)
        return R

    # Calculate the skew-symmetric cross product matrix
    K = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    # Compute the rotation matrix using Rodrigues' rotation formula
    R = np.eye(3) + K + K @ K * (1 / (1 + c))
    return R

def apply_transformation(df, R, t):
    """
    Applies the given rotation matrix R and translation vector t to the marker coordinates in the DataFrame.
    """
    new_df = df.copy()
    for marker_col in df.columns[2:]:
        for coordinate in df[marker_col]:
            coord_array = np.array(coordinate)
            transformed_coord = np.dot(R, coord_array) + t
            coordinate[0] = transformed_coord[0]
            coordinate[1] = transformed_coord[1]
            coordinate[2] = transformed_coord[2]
    return new_df

def write_trc_file(df, original_trc_path, output_trc_path):
    """
    Writes the DataFrame df to a .trc file, preserving the header from the original file.
    """
    # Read the header (first 6 lines) from the original file
    with open(original_trc_path, 'r') as f:
        header_lines = [next(f) for _ in range(6)]
    
    # Prepare data: flatten marker columns back to X, Y, Z columns
    data = df.copy()
    marker_cols = [col for col in data.columns if col not in ['Frames', 'Time']]
    new_cols = ['Frames', 'Time']
    flat_data = [data['Frames'], data['Time']]
    for marker in marker_cols:
        arr = np.array(data[marker].tolist())
        for i, axis in enumerate(['X', 'Y', 'Z']):
            new_cols.append(f"{marker}_{axis}")
            flat_data.append(arr[:, i])
    out_df = pd.DataFrame({col: vals for col, vals in zip(new_cols, flat_data)})
    
    # Write header and data to new file
    with open(output_trc_path, 'w', newline='') as f:
        f.writelines(header_lines)
        out_df.to_csv(f, sep='\t', index=False, header=False, float_format='%.6f')

def main(subject_ID, mvt_ID):
    ## LOADING THE DATA
    # Load Webcam trc file 
    webcam_file_path = "recordings/subject" + subject_ID + "/webcam_" + mvt_ID + ".trc"
    webcam_df = read_trc_file(webcam_file_path)
    # Inspect webcam df
    #print(webcam_df.head())


    ## TRANSLATION MATRIX

    goal_middle_point_hips = np.array([-79.247, 878.088, 0]) #in mm

    webcam_initial_right_hip = np.array(webcam_df['WEBCAM_JOINT_RIGHT_HIP'][0])
    webcam_initial_left_hip = np.array(webcam_df['WEBCAM_JOINT_LEFT_HIP'][0])

    # hip middle point
    webcam_initial_middle_point_hips = (webcam_initial_right_hip + webcam_initial_left_hip) / 2
    print("Webcam Initial Middle Point Hips:", webcam_initial_middle_point_hips) # should be (0, 0, 0)

    # translation vector
    t_webcam = goal_middle_point_hips - webcam_initial_middle_point_hips
    print("Translation vector Webcam:", t_webcam)


    ## APPLY TRANSFORMATION TO STEREOCAMERA
    R_webcam = np.eye(3) # no rotation needed
    transformed_webcam_df = apply_transformation(webcam_df, R_webcam, t_webcam)
    #print(transformed_webcam_df.head())
    #print(webcam_df.head())


    ## WRITE IN A NEW TRC FILE 
    output_trc_path = webcam_file_path
    write_trc_file(transformed_webcam_df, webcam_file_path, output_trc_path)



if __name__ == "__main__":
    # Argparse 
    parser = argparse.ArgumentParser(description='Does reference frame correction for webcam data.')
    parser.add_argument('subject_ID', type=str, help='Subject ID')
    parser.add_argument('mvt_ID', type=str, help='Movement ID')
    args = parser.parse_args()

    main(args.subject_ID, args.mvt_ID)