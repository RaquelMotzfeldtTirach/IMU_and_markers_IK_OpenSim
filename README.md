# OpenSim Sensor Fusion for Human Motion Analysis

This repository implements a multi-modal sensor fusion framework that combines IMU (Inertial Measurement Unit) data with Webcam and Stereocamera marker tracking for accurate human motion analysis. Using OpenSim's toolkit, it performs weighted inverse kinematics that leverages the strengths of both sensor types: IMUs provide continuous orientation data without occlusion issues, while optical markers offer precise 3D position information.

The framework includes automated data conversion, model calibration, temporal synchronization, and configurable sensor weighting to produce robust motion capture results suitable for biomechanical research and clinical applications.

## Folder Structure
- `OpenSim/`  
  Utilities for converting and calibrating raw data from IMUs, webcam and stereocamera. And inverse kinematics with sensor fusion!
- `recordings/`  
  Example log files, recordings and results
- `analytics/`  
  This is were the optimization outcomes are stored, with metrics and plots

## Requirements
- Linux, ubuntu 24.04 (not tested on previous versions)
- Conda with OpenSim package and pandas, numpy, os, optuna, etc

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

## Data pre-processing
This repo solely focuses on combining multi-modal data to then run inverse kinematics from OpenSim. The IK process is being optimized by tuning the weights used in the IK problem, according to a defined cost-function. 

Before one can start with multi-modal IK, one has to collect data or download a dataset. For the first alternative, there are three existing repos:
- Body tracking using a webcam: https://github.com/RaquelMotzfeldtTirach/Mediapipe_OpenSim
- Body tracking using a stereocamera: https://github.com/RaquelMotzfeldtTirach/ZED_stereocamera_OpenSim
- Body tracking with Xsens IMUs: https://github.com/RaquelMotzfeldtTirach/Xsens_mtw_OpenSim
  
Each individual repo also has the possibility to run IK, but only on one data type. For fusion, this is the repo you will need.
Save all the data files in the following structure:
- `recordings/`
  - `SubjectXX/`
    - `imu_trial_ID/` - for the IMU .txt files 
    - `webcam_trial_ID.trc` 
    - `stereocamera_trial_ID.trc`
    - `vicon_trial_ID.trc` for the reference data recording with a MoCap system like Vicon

## Usage 
### Automatic
  ```sh
  ./script.sh
  ```
And give it the suject ID
### Manual
Activate the conda environment
  ```sh
  conda activate opensim_scripting
  ```
From there you can run any script you'd like from the OpenSim folder
