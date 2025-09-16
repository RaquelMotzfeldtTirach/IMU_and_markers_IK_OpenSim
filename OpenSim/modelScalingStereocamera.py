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
        scaler.find('output_model_file').text = output_model_file_scaling
        scaler = tag.find('GenericModelMaker')
        scaler.find('model_file').text = "../../" + model_path
        scaler.find('marker_set_file').text = "../../OpenSim/Models/Rajagopal/stereocamera_markers.xml"
        scaler = tag.find('MarkerPlacer')
        scaler.find('marker_file').text = static_trial_path  
        #scaler.find('output_model_file').text = output_model_file # This fails for some reason, so we'll do it manually, later
        scaler.find('time_range').text = str(initial_time) + " " + str(initial_time + 10.0) # first 10 seconds of static trial


    new_file_path = 'recordings/subject'+ subject_ID +'/stereocamera_scaling_setup_'+ mvt_ID +'.xml'
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

    final_model_path = scale_tool.getMarkerPlacer().getOutputModelFileName()
    print("Final Model File Name:", final_model_path)
    
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
        #joints = tag.find('JointSet')
        markers = tag.find('MarkerSet')

    # Write in scaling model
    tree = ET.parse(output_model_file_scaling.removeprefix("../../"))
    root = tree.getroot()
    #frames = []
    #count = 0
    #for joint_set in root.iter('JointSet'):
    #    for obj in joint_set.iter('objects'):
            # Find all 'frame' tags within the 'object' tags
    #        for frame in obj.iter('frames'):
                # Get the text content of the frame tag
    #            frame_content = frame.text.strip() if frame.text else ""
    #           frames.append(frame_content)
    #print("Found so many frames in jointset: ", len(frames))

    for tag in root.iter('Model'):
        #tag.find('JointSet').clear()
        tag.find('MarkerSet').clear()
        #tag.find('JointSet').extend(joints)
        tag.find('MarkerSet').extend(markers)
    #for joint_set in root.iter('JointSet'):
    #    for obj in joint_set.iter('objects'):
    #        # Find all 'frame' tags within the 'object' tags
    #        for frame in obj.iter('frames'):
    #            tag.text = frames[count]
    #            count = count + 1
    
    tree.write(output_model_file, encoding='utf-8', xml_declaration=True)



def main(subject_ID, mvt_ID, subject_mass, subject_height, subject_age, subject_sex, model_path):
    """Main entry point of the script."""
    config_template_path = 'OpenSim/scaling_setup_template_stereocamera.xml'

    # Load the model
    model = load_model(model_path)

    # Define paths
    static_trial_path = "../../recordings/subject" + subject_ID + "/stereocamera_static.trc"
    output_model_file = "OpenSim/Models/Rajagopal/Calibrated_Rajagopal_subject" + subject_ID + "_" + mvt_ID + ".osim"
    output_model_file_scaling = "../../OpenSim/Models/Rajagopal/Scaled_Rajagopal_subject" + subject_ID + "_" + mvt_ID + ".osim"
    output_model_file_markers = "OpenSim/Models/Rajagopal/Marker_Placed_Rajagopal_subject" + subject_ID + "_" + mvt_ID + ".osim"


    # Define initial time for static trial
    # read the .trc file to get the initial time
    with open(static_trial_path.removeprefix("../../"), 'r') as f:
        lines = f.readlines()
        initial_time = float(lines[6].strip().split()[1])  # Assuming time is the 6th row, second column
    print("Initial time of the static trial:", initial_time)

    # Create our own config file
    config_path = create_config_file(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, output_model_file_scaling, mvt_ID, model_path, initial_time)
    print("Configuration file created at:", config_path)

    # Create and configure the Scale Tool
    scale_tool = create_scale_tool(config_path)

    # Print Scale Tool information
    print_scale_tool_info(scale_tool)

    # Print Marker file names
    print_marker_file_names(scale_tool)

    # Run the Scale Tool
    run_scaling(scale_tool)

    # Get the model with the markers placed
    new_model = scale_tool.createModel()
    # Save the new model to a file
    new_model.printToXML(output_model_file_markers)

    # Mix both scaled and marker placed models into one
    create_mixed_model(output_model_file_scaling, output_model_file_markers, output_model_file)

    # Delete intermediate files
    if os.path.exists(output_model_file_scaling.removeprefix("../../")):
        os.remove(output_model_file_scaling.removeprefix("../../"))
    if os.path.exists(output_model_file_markers):
        os.remove(output_model_file_markers)

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