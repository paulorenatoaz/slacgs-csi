# Example usage
import os
import sys

# Add the 'src' directory to the sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
print("Source directory added to sys.path:", src_path)
sys.path.append(src_path)

# Print sys.path to verify paths
print("Current sys.path:", sys.path)
from slacgscsi import PROJECT_DIR, run_simulation_train_test

if __name__ == "__main__":
    lab_data_path = os.path.join(PROJECT_DIR, 'Lab.csv')
    removed_features = ['rssi', 'amp3', 'amp4', 'amp5', 'amp6', 'amp7', 'amp9', 'amp10',
                  'amp11', 'amp12', 'amp13', 'amp14', 'amp15', 'amp16', 'amp17', 'amp18',
                  'amp19', 'amp20', 'amp21', 'amp23', 'amp24', 'amp25', 'amp26', 'amp27',
                  'amp28', 'amp30', 'amp31', 'amp32', 'amp33', 'amp34', 'amp35', 'amp36',
                  'amp37', 'amp38', 'amp39', 'amp40', 'amp41', 'amp42', 'amp43', 'amp44',
                  'amp45', 'amp46', 'amp47', 'amp49', 'amp50', 'amp51', 'amp52']
    for n_features_to_remove in range(4, -1, -1):
        run_simulation_train_test(lab_data_path, removed_features, n_features_to_remove=n_features_to_remove, test_mode=True)

