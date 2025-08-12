# OpenSim Sensor Fusion for Human Motion Analysis

This repository implements a multi-modal sensor fusion framework that combines IMU (Inertial Measurement Unit) data with webcam marker tracking for accurate human motion analysis. Using OpenSim's toolkit, it performs weighted inverse kinematics that leverages the strengths of both sensor types: IMUs provide continuous orientation data without occlusion issues, while optical markers offer precise 3D position information.

The framework includes automated data conversion, model calibration, temporal synchronization, and configurable sensor weighting to produce robust motion capture results suitable for biomechanical research and clinical applications.

## Folder Structure
- `OpenSim/`  
  Utilities for converting and calibrating raw data from IMUs, webcam and stereocamera. And inverse kinematics with sensor fusion!
- `recordings/`  
  Example log files, recordings and results

## Requirements
- Linux, ubuntu 24.04 (not tested on previous versions)
- Conda with OpenSim package

## Setup Instructions

### Conda environement for OpenSense library

1. **Install Conda:**
  Follow the instructions on the Anaconda website: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

2. **Create and activate Conda environment:**
  ```sh
  conda create -n opensim_scripting python=3.11 numpy
  conda activate opensim_scripting
  ```
3. **Libraries installation:**
  ```sh
  conda install conda-forge::simbody
  conda install conda-forge::cma
  ```

4. **OpenSim installation:**
  ```sh
  conda install -c opensim-org opensim
  ```


## Usage 

