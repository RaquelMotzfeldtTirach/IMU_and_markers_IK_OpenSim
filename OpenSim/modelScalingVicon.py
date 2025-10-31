import opensim as osim
import xml.etree.ElementTree as ET
import os
import argparse


def load_model(model_path):
    """Load the OpenSim model."""
    model = osim.Model(model_path)
    print("Name of the model:", model.getName())
    return model

def create_config_file(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, output_model_file_scaling, mvt_ID, model_path, initial_time):
    """Create a configuration file for the Scale Tool."""
    tree = ET.parse(config_template_path)
    root = tree.getroot()
    for tag in root.iter('ScaleTool'):
        tag.find('height').text = subject_height  
        tag.find('age').text = subject_age  
        tag.find('notes').text = subject_ID
        tag.find('sex').text = subject_sex
        tag.find('mass').text = subject_mass  
        scaler = tag.find('ModelScaler')
        scaler.find('marker_file').text = static_trial_path  
        scaler.find('output_model_file').text = "../../" + output_model_file_scaling 
        scaler = tag.find('GenericModelMaker')
        scaler.find('model_file').text = "../../" + model_path
        scaler.find('marker_set_file').text = "../../OpenSim/Models/Rajagopal/vicon_markers.xml"
        scaler = tag.find('MarkerPlacer')
        scaler.find('marker_file').text = static_trial_path
        scaler.find('time_range').text = str(initial_time) + " " + str(initial_time + 10.0) # first 10 seconds of static trial
        #scaler.find('output_motion_file').text = "scaling_motion.mot"
        scaler.find('output_model_file').text = "Unassigned"
        #scaler.find('output_marker_file').text = "../../../scaled_markers.xml"


    new_file_path = 'recordings/subject'+ subject_ID +'/vicon_scaling_setup_'+ mvt_ID +'.xml'
    tree.write(new_file_path, encoding='utf-8', xml_declaration=True)
    return new_file_path

def create_config_file_markers(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, mvt_ID, model_path):
    """Create a configuration file for the Scale Tool."""
    tree = ET.parse(config_template_path)
    root = tree.getroot()
    for tag in root.iter('ScaleTool'):
        tag.find('height').text = subject_height  
        tag.find('age').text = subject_age  
        tag.find('notes').text = subject_ID
        tag.find('sex').text = subject_sex
        tag.find('mass').text = subject_mass  
        scaler = tag.find('ModelScaler')
        scaler.find('apply').text = "false"
        scaler.find('marker_file').text = static_trial_path  
        scaler.find('output_model_file').text = "Unassigned"
        scaler = tag.find('GenericModelMaker')
        scaler.find('model_file').text = "../../" + model_path
        scaler.find('marker_set_file').text = "Unassigned"
        scaler = tag.find('MarkerPlacer')
        scaler.find('marker_file').text = static_trial_path
        #scaler.find('output_motion_file').text = "scaling_motion.mot"
        scaler.find('output_model_file').text = "Unassigned"
        #scaler.find('output_marker_file').text = "../../../scaled_markers.xml"


    new_file_path = 'recordings/subject'+ subject_ID +'/vicon_scaling_setup_'+ mvt_ID +'.xml'
    tree.write(new_file_path, encoding='utf-8', xml_declaration=True)
    return new_file_path

def create_scale_tool(config_path):
    """Create a ScaleTool object using the configuration file."""
    scale_tool = osim.ScaleTool(config_path)
    return scale_tool

def print_scale_tool_info(scale_tool):
    """Print relevant information from the ScaleTool object."""
    print("Name:", scale_tool.getName())
    print("ID:", scale_tool.getPropertyByName("notes").toString())
    print("Subject Mass:", scale_tool.getSubjectMass())
    print("Subject Height:", scale_tool.getSubjectHeight())
    print()

def print_marker_file_names(scale_tool):
    """Print the marker set and marker file names."""
    generic_model_maker = scale_tool.getGenericModelMaker()
    print("Marker Set File Name:", generic_model_maker.getMarkerSetFileName())

    static_trial = scale_tool.getModelScaler()
    print("Model Scaler File Name:", static_trial.getMarkerFileName())

    model_path = scale_tool.getGenericModelMaker().getModelFileName()
    print("Model File Name:", model_path)
    
    print()

def run_scaling(scale_tool):
    """Run the Scale Tool."""
    scale_tool.run()
    print("Scaling completed.")

def create_mixed_model(output_model_file_scaling, output_model_file_markers, output_model_file):
    """Create a mixed model by combining the scaled and marker placed models."""
    # Fetch data from marker model
    tree_markers = ET.parse(output_model_file_markers)
    root_markers = tree_markers.getroot()
    for tag in root_markers.iter('Model'):
        markers = tag.find('MarkerSet')
        #marker_jointset = tag.find('JointSet')

    # Write in scaling model
    tree = ET.parse(output_model_file_scaling.removeprefix("../../"))
    root = tree.getroot()
    for tag in root.iter('Model'):
        tag.find('MarkerSet').clear()
        tag.find('MarkerSet').extend(markers)
        #scaled_jointset = tag.find('JointSet')
    #print("MarkerSet swaped!")

    #if scaled_jointset is not None and marker_jointset is not None:
    #    #print("Both JointSet elements found.")
    #    scaled_objects = scaled_jointset.find('objects')
    #    marker_objects = marker_jointset.find('objects')
    #    # Build a mapping from joint name to frames in marker_placed
    #    marker_frames_map = {}
    #    for marker_joint in marker_objects:
    #        name = marker_joint.get('name')
    #        #print("Checking marker joints for <frames>...in the joint: ", name)
    #        frames = marker_joint.find('frames')
    #        if name and frames is not None:
    #            marker_frames_map[name] = frames
    #            #print(f"Found frames for joint: {name}")

    #    # For each joint in scaled, swap frames if present in marker_placed
    #    for scaled_joint in scaled_objects:
    #        name = scaled_joint.get('name')
    #        if name in marker_frames_map:
    #            # Remove existing <frames> if present
    #            for child in list(scaled_joint):
    #                if child.tag == 'frames':
    #                    scaled_joint.remove(child)
    #            # Deep copy and insert the marker's <frames>
    #            marker_frames = marker_frames_map[name]
    #            scaled_joint.append(ET.fromstring(ET.tostring(marker_frames)))
    #            #print(f"Swapped frames for joint: {name}")
    
    tree.write(output_model_file, encoding='utf-8', xml_declaration=True)


def main(subject_ID, mvt_ID, subject_mass, subject_height, subject_age, subject_sex, model_path):
    """Main entry point of the script."""
    config_template_path = 'OpenSim/scaling_setup_template_vicon.xml'

    # Load the model
    model = load_model(model_path)

    # Define paths
    static_trial_path = "../../recordings/subject" + subject_ID + "/vicon_static.trc"
    output_model_file = "OpenSim/Models/Rajagopal/Calibrated_Rajagopal_subject" + subject_ID + "_" + mvt_ID + "_vicon.osim"
    output_model_file_scaling = "../../OpenSim/Models/Rajagopal/Scaled_Rajagopal_subject" + subject_ID + "_" + mvt_ID + "_vicon.osim"
    output_model_file_markers = "OpenSim/Models/Rajagopal/Marker_Placed_Rajagopal_subject" + subject_ID + "_" + mvt_ID + "_vicon.osim"

    # Define initial time for static trial
    # read the .trc file to get the initial time
    with open(static_trial_path.removeprefix("../../"), 'r') as f:
        lines = f.readlines()
        initial_time = float(lines[6].strip().split()[1])  # Assuming time is the 6th row, second column
    print("Initial time of the static trial:", initial_time)

    # Create our own config file
    config_path = create_config_file(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, output_model_file, mvt_ID, model_path, initial_time)
    print("Configuration file created at:", config_path)

    # Create and configure the Scale Tool
    scale_tool = create_scale_tool(config_path)

    # Print Scale Tool information
    print_scale_tool_info(scale_tool)

    # Print Marker file names
    print_marker_file_names(scale_tool)

    # Run the Scale Tool
    run_scaling(scale_tool)

    return output_model_file

    

if __name__ == "__main__":
    # Argparse 
    parser = argparse.ArgumentParser(description='Scale OpenSim model using webcam data.')
    parser.add_argument('subject_ID', type=str, help='Subject ID')
    parser.add_argument('mvt_ID', type=str, help='Movement ID')
    parser.add_argument('subject_mass', type=str, help='Subject mass in kg')
    parser.add_argument('subject_height', type=str, help='Subject height in cm')
    parser.add_argument('subject_age', type=str, help='Subject age in years')
    parser.add_argument('subject_sex', type=str, help='Subject sex (M/F)')
    parser.add_argument('model_path', type=str, help='Path to the OpenSim model file')
    args = parser.parse_args()

    main(args.subject_ID, args.mvt_ID, args.subject_mass, args.subject_height, args.subject_age, args.subject_sex, args.model_path)