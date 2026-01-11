eval "$(conda shell.bash hook)"
conda activate opensim_scripting

read -p "Enter subject ID: " SUBJECT_ID


#python3 get_data.py --SUBJECT_ID=$SUBJECT_ID


python3 OpenSim/run_optimization.py --SUBJECT_ID=$SUBJECT_ID
