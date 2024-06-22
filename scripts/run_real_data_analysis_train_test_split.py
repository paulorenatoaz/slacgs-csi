import os
import sys

# Add the 'src' directory to the sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
print("Source directory added to sys.path:", src_path)
sys.path.append(src_path)
from slacgscsi import run_real_data_analysis_train_test_split

if __name__ == "__main__":

    lab_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Lab.csv')
    json_output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'analysis_results.json')
    removed_features = ['rssi', 'amp3', 'amp4', 'amp5', 'amp6', 'amp7', 'amp9', 'amp10',
                  'amp11', 'amp12', 'amp13', 'amp14', 'amp15', 'amp16', 'amp17', 'amp18',
                  'amp19', 'amp20', 'amp21', 'amp23', 'amp24', 'amp25', 'amp26', 'amp27',
                  'amp28', 'amp30', 'amp31', 'amp32', 'amp33', 'amp34', 'amp35', 'amp36',
                  'amp37', 'amp38', 'amp39', 'amp40', 'amp41', 'amp42', 'amp43', 'amp44',
                  'amp45', 'amp46', 'amp47', 'amp49', 'amp50', 'amp51', 'amp52']

    for n_features_to_remove in range(1, -1, -1):
        run_real_data_analysis_train_test_split(lab_data_path, removed_features, n_features_to_remove=n_features_to_remove)