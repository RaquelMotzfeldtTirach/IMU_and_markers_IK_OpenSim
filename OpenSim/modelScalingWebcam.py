import opensim as osim
import xml.etree.ElementTree as ET
import os


def load_model(model_path):
    """Load the OpenSim model."""
    model = osim.Model(model_path)
    print("Name of the model:", model.getName())
    return model

def create_config_file(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, output_model_file):
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
        scaler.find('output_model_file').text = output_model_file  


    new_file_path = 'recordings/subject'+ subject_ID +'/scaling_setup.xml'
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


def extract_timestamps(trial_file_path):
    start_time = None
    end_time = None
    with open(trial_file_path, 'r') as file:
        line_count = 0
        for line in file:
            line = line.strip()
            # Skip first 6 lines
            if line_count < 6:
                line_count += 1
                continue
            
            # Check if the line starts with a frame number (indicates a data line)
            if line[0].isdigit():
                parts = line.split('\t')  # Split by tab; adjust if using a different delimiter
                if len(parts) > 1:
                    timestamp = float(parts[1])  # Convert the timestamp string to float
                    # Initialize start_timestamp and update end_timestamp
                    if start_time is None:
                        start_time = timestamp
                    end_time = timestamp  # Update every time we read a line
            line_count += 1

    return start_time, end_time


def main():
    """Main entry point of the script."""
    model_path = 'OpenSim/models/Rajagopal/Rajagopal_2015.osim'
    config_template_path = 'OpenSim/scaling_setup_template.xml'

    # Load the model
    model = load_model(model_path)

    # Get subject information
    subject_ID = input("Enter subject ID: ")
    subject_mass = input("Enter subject mass (kg): ")
    subject_height = input("Enter subject height (mm): ")
    subject_age = input("Enter subject age (years): ")
    subject_sex = input("Enter subject sex (male/female): ")
    static_trial_path = input("Enter static trial file name (.trc): ")

    output_model_file = "../../OpenSim/models/Rajagopal/Rajagopal_scaled_subject"+subject_ID+".osim"

    # Create our own config file
    config_path = create_config_file(config_template_path, subject_ID, subject_mass, subject_height, subject_sex, subject_age, static_trial_path, output_model_file)
    print("Configuration file created at:", config_path)

    # Create and configure the Scale Tool
    scale_tool = create_scale_tool(config_path)

    # Print Scale Tool information
    print_scale_tool_info(scale_tool)

    # Print Marker file names
    print_marker_file_names(scale_tool)

    # Run the Scale Tool
    run_scaling(scale_tool)

    

if __name__ == "__main__":
    main()