import numpy as np
import math
from math import pi
import argparse
import os
import shutil
import time

def main(filepath):
    lines = []

    # Read the original file to determine the number of markers and adjust the lines
    with open(filepath, 'r') as infile:
        for line_number, line in enumerate(infile):
            if line_number == 2:  
                parts = line.split('\t')
                num_markers = int(parts[3])  #
                expected_columns = num_markers * 3 + 2
                lines.append(line)
            elif line_number >= 5:
                # Split the line by tabs
                columns = line.split('\t')
                # Replace empty fields with 'NaN'
                columns = ['NaN' if col == '' else col for col in columns]
                # Plus one 'NaN' if it's the end of the row
                # Check if the last column is empty
                if columns[-2] == 'NaN':
                    print("NaN at the end at row: ", line_number)
                    # Add another 'NaN' if the last column is a NaN (originally an empty string)
                    columns[-1] = 'NaN'
                    columns.append('\n')
                # Join the columns back together with tabs
                lines.append('\t'.join(columns))
            else:
                lines.append(line)

    # Write the adjusted lines back to the same file
    with open(filepath, 'w') as outfile:
        outfile.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverse kinematics using OpenSim for Vicon data")
    parser.add_argument("filepath", type=str, help="TRC file path")

    args = parser.parse_args()

    main(args.filepath)