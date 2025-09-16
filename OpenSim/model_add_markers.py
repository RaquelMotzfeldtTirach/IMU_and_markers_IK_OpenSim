import opensim as osim
import xml.etree.ElementTree as ET
import os
import argparse


def load_model(model_path):
    """Load the OpenSim model."""
    model = osim.Model(model_path)
    print("Name of the model:", model.getName())
    return model


def main(model_path, camera_type):
    """Main entry point of the script."""
    markers_file = "OpenSim/Models/Rajagopal/"+ camera_type +"_markers.xml"

    # Load the model
    model = load_model(model_path)
    old_count = model.getNumMarkers()

    # Load the markers
    marker_set = osim.MarkerSet(markers_file)

    # Add markers to mode
    model.updateMarkerSet(marker_set)

    new_count = model.getNumMarkers()

    if (new_count > old_count):
        print("Success!")
        print("there are now so many markers: ", new_count)

    # Save model
    model.printToXML(model_path)

    return model_path

    

if __name__ == "__main__":
    # Argparse 
    parser = argparse.ArgumentParser(description='Add markers to model')
    parser.add_argument('model_path', type=str, help='Path to the OpenSim model file')
    parser.add_argument('camera_type', type=str, help='is it webcam or stereocamera markers?')
    args = parser.parse_args()

    main(args.model_path, args.camera_type)