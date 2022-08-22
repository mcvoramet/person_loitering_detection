%%bash
# input data preparation
pip install -r requirements.txt
source pld_env/bin/activate
python3 preproecess_input.py
deactivate
# perform loitering detection
conda env create -f skeleton_based_anomaly_detection/environment.yml
conda activate tbad
python3 loitering_detection.py
conda deactivate