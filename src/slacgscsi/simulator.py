from datetime import datetime
import hashlib
import json
import os
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
import math
from sklearn.utils import compute_class_weight
from tqdm import tqdm
import time
from .config import PROJECT_DIR


def load_data(lab_data_path, removed_features):
    # Load the dataset
    lab_data = pd.read_csv(lab_data_path)

    # Remove corrupted subcarriers
    X = lab_data.drop(columns=['rotulo'] + removed_features)
    y = lab_data['rotulo'].apply(lambda x: 1 if x == 'ofensivo' else 0)

    return X, y, X.columns

def calculate_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def generate_synthetic_data(class1, class2, n_samples, n_components_class0, n_components_class1):

    """
    Generate synthetic data based on the provided class data using Gaussian Mixture Models.

    Parameters:
    - class1: array-like, shape (n_samples, n_features)
      Data points for class 1.
    - class2: array-like, shape (n_samples, n_features)
      Data points for class 2.
    - n_samples: int
      Number of samples to generate for each class.
    - n_components: int
      Number of mixture components to use for GMM.

    Returns:
    - new_class1: array, shape (n_samples, n_features)
      Generated data points for class 1.
    - new_class2: array, shape (n_samples, n_features)
      Generated data points for class 2.
    """

    total_samples = len(class1) + len(class2)
    proportion_class_0 = len(class1) / total_samples
    proportion_class_1 = len(class2) / total_samples

    n_samples_class0 = int(n_samples*proportion_class_0)
    n_samples_class1 = int(n_samples*proportion_class_1)

    # Convert to numpy arrays
    class1 = np.array(class1)
    class2 = np.array(class2)

    # Fit GMM to each class
    gmm1 = GaussianMixture(n_components=n_components_class0, random_state=n_samples_class0)
    gmm1.fit(class1)

    gmm2 = GaussianMixture(n_components=n_components_class1, random_state=n_samples_class1)
    gmm2.fit(class2)

    # Generate new samples
    new_class1, _ = gmm1.sample(n_samples_class0)
    new_class2, _ = gmm2.sample(n_samples_class1)

    if len(new_class1.shape) == 1:
        new_class1 = new_class1.reshape(-1, 1)
        new_class2 = new_class2.reshape(-1, 1)

    X_synthetic = np.vstack((new_class1, new_class2))
    y_synthetic = np.array([0] * n_samples_class0 + [1] * n_samples_class1)

    return X_synthetic, y_synthetic


def estimate_n_components(data, max_components, random_state):
    """
    Estimate the optimal number of components for GMM using BIC.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
      Data points for which to estimate the number of components.
    - max_components: int
      Maximum number of components to consider.

    Returns:
    - optimal_components: int
      Estimated optimal number of components.
    """
    bics = []
    aics = []
    n_components_range = range(1, max_components + 1)

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(data)
        bics.append(gmm.bic(data))
        # aics.append(gmm.aic(data))

    optimal_components_bic = n_components_range[np.argmin(bics)]
    # optimal_components_aic = n_components_range[np.argmin(aics)]

    return optimal_components_bic#, optimal_components_aic


def print_dataset_report(X, corrompidas, y):
    print("Dataset shape:", X.shape)
    print("Corrupted subcarriers removed:", corrompidas)
    print("Class distribution:\n", y.value_counts())
    print("Features:\n", X.columns)


def print_feature_statistics(mean_empirical_0, cov_empirical_0, mean_empirical_1, cov_empirical_1):
    print("Mean of class 0:\n", mean_empirical_0)
    print("\nCovariance of class 0:\n", cov_empirical_0)
    print("\nMean of class 1:\n", mean_empirical_1)
    print("\nCovariance of class 1:\n", cov_empirical_1)

def print_simulation_parameters(X, removed, n_samples_list, pca_retained_variance, max_gmm_component, test_mode):
    print("=" * 50)
    print("\nSimulation Parameters:")
    print(f"Number of Features: {X.shape[1]}")
    print(f"Dataset Size: {X.shape[0]}")
    print(f"Removed features: {removed}")
    print(f"Sample sizes to test: {n_samples_list}")
    print(f"PCA Retained Variance: {pca_retained_variance}")
    print(f"Max GMM Components: {max_gmm_component}")
    print(f"Test Mode: {test_mode}\n")
    print("=" * 50)


def print_summarized_report(n_samples, n_features, rounds, mean_error_rate, mean_training_time,
                            mean_classification_report, mean_conf_matrix):

    print('\n',"=" * 50)
    print(f"Summarized Report for N={n_samples}:")
    print(f"Number of Features: {n_features}")
    print(f'Rounds: {rounds}')
    print(f"Mean Error Rate: {mean_error_rate:.4f}")
    print(f"Mean Training Time: {mean_training_time:.2f} seconds")
    print(f"Mean Classification Report:\n {pd.DataFrame(mean_classification_report).transpose()}")
    print(f"Mean Confusion Matrix: {mean_conf_matrix}")
    print("=" * 50)


def print_simulation_report(simulation_report):
    simulation_report_df = pd.DataFrame(simulation_report)
    print("=" * 50)
    print("Simulation Report:")
    print(simulation_report_df.drop(columns=['mean_classification_report']))
    print("=" * 50)
    return simulation_report_df


def plot_two_classes_from_list(class1_data, class2_data, class1_label='Class 1', class2_label='Class 2'):
    """
    Plots datapoints for two classes.

    Parameters:
    class1_data (array-like): Coordinates of the first class, should be a 2D array or list of tuples.
    class2_data (array-like): Coordinates of the second class, should be a 2D array or list of tuples.
    class1_label (str): Label for the first class.
    class2_label (str): Label for the second class.
    """
    # Extract x and y coordinates for each class
    class1_x, class1_y = zip(*class1_data)
    class2_x, class2_y = zip(*class2_data)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(class1_x, class1_y, color='blue', label=class1_label)
    plt.scatter(class2_x, class2_y, color='red', label=class2_label)

    # Add title and labels
    plt.title('Scatter Plot of Two Classes')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the plot
    plt.show()

def plot_two_classes_from_dfs(X_df_class_0, X_df_class_1, class0_label='Class 0', class1_label='Class 1'):
    """
    Plots datapoints for two classes from separate DataFrames.

    Parameters:
    X_df_class_0 (DataFrame): DataFrame containing coordinates of the first class.
    X_df_class_1 (DataFrame): DataFrame containing coordinates of the second class.
    class0_label (str): Label for the first class.
    class1_label (str): Label for the second class.
    """
    # Get the feature names (assumes the first two columns are the features to be plotted)
    feature_names = X_df_class_0.columns[:2]
    feature1, feature2 = feature_names[0], feature_names[1]

    # Extract coordinates for each class
    class0_x = X_df_class_0[feature1]
    class0_y = X_df_class_0[feature2]
    class1_x = X_df_class_1[feature1]
    class1_y = X_df_class_1[feature2]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(class0_x, class0_y, color='blue', label=class0_label)
    plt.scatter(class1_x, class1_y, color='red', label=class1_label)

    # Add title and labels
    plt.title('Scatter Plot of Two Classes')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()

    # Show the plot
    plt.show()


def plot_metrics(simulation_report, n_features, n_samples_list, id, test_size, pca_retained_variance, max_gmm_component):
    plt.figure(figsize=(10, 6))

    # Plot the error rate
    plt.plot(n_samples_list, list(simulation_report["mean_error_rate"].values()), marker='o', linestyle='--', label='Error Rate', color='blue')

    # Plot the precision for class 0
    plt.plot(n_samples_list, list(simulation_report["mean_precision_class_0"].values()), marker='x', linestyle='--', label='Precision Class 0', color='orange')

    # Plot the f-score for class 1
    plt.plot(n_samples_list, list(simulation_report["mean_fscore_class_0"].values()), marker='s', linestyle='--', label='F-Score Class 0', color='orange')

    # Plot the precision for class 1
    plt.plot(n_samples_list, list(simulation_report["mean_precision_class_1"].values()), marker='x', linestyle='--',
             label='Precision Class 1', color='green')

    # Plot the f-score for class 0
    plt.plot(n_samples_list, list(simulation_report["mean_fscore_class_1"].values()), marker='s', linestyle='--',
             label='F-Score Class 1', color='green')

    crossvalidation = False if simulation_report["mean_num_folds"][0] is None else True
    simulation_method = 'cross-validation' if crossvalidation else 'train_test_split=' + str(test_size)
    plt.title('Error Rate, Precision, and F-Score vs. Number of Samples for ' + str(n_features) + ' features\n' +
              'pca = ' + str(pca_retained_variance) + '; max_gmm_comp = ' + str(max_gmm_component) +
              '; method: ' + simulation_method)
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Metrics')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(10 ** 2, 10 ** 5)
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'graphs',
                               'simulation_'+ str(id) +'_metrics-vs-N_' + str(n_features) + '-feat' + '.png')

    # create directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def store_simulation_results(simulation_data):
    output_file_path = os.path.join(PROJECT_DIR, 'data', 'outputs', 'simulation_results.json')

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Read the existing data to get the current sequence ID
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as file:
            try:
                existing_data = json.load(file)
                if isinstance(existing_data, list) and len(existing_data) > 0:
                    last_id = existing_data[-1]["id"]
                    current_id = last_id + 1
                else:
                    existing_data = []
                    current_id = 1
            except json.JSONDecodeError:
                existing_data = []
                current_id = 1
    else:
        existing_data = []
        current_id = 1

    plot_image_path = plot_metrics(simulation_data['simulation_report'], simulation_data['n_features'],
                                   simulation_data['n_samples_list'], current_id, simulation_data['test_size'],
                                   simulation_data['pca_retained_variance'], simulation_data['max_gmm_component'])

    simulation_data['plot_image_path'] = plot_image_path

    # Add the id to the simulation data
    new_entry = {
        "id": current_id,
        "simulation_data": simulation_data
    }

    # Append the new data
    existing_data.append(new_entry)

    # Write back the updated data to the JSON file
    with open(output_file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

# def store_simulation_results(simulation_data):
#     # Save the simulation results to a JSON file create directory if it does not exist
#     output_file_path = os.path.join(PROJECT_DIR, 'outputs', 'simulation_results.json')
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#
#     with open(output_file_path, 'a') as file:
#         json.dump(simulation_data, file, indent=4)


def print_progress(start_time, running_event, prefix=""):
    while running_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"\r{prefix} Training Time: {elapsed_time:.3f} seconds", end="")
        time.sleep(0.1)


def rank_features(X, y, feature_names, ranking_file_path, ranking_data, dataset_hash):
    print("\nStarting feature Ranking...")
    start_time = time.time()
    running_event = threading.Event()
    running_event.set()
    progress_thread = threading.Thread(target=print_progress, args=(start_time, running_event, "Feature Selection"))
    progress_thread.start()

    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X, y)
    importances = selector.feature_importances_
    indices = np.argsort(importances)[::-1]

    running_event.clear()
    progress_thread.join()

    rank = []
    for i in range(X.shape[1]):
        feature_name = feature_names[indices[i]]
        importance = importances[indices[i]]
        print(f"Feature {feature_name} ({importance:.6f})")
        rank.append([feature_name, importance])

    # Save the ranking to a JSON file
    ranking_data[dataset_hash] = rank
    with open(ranking_file_path, 'w') as file:
        json.dump(ranking_data, file)
    return rank


def select_features(X, y, n_features_to_remove, feature_names, dataset_hash):
    ranking_file_path = os.path.join(PROJECT_DIR, 'data', 'outputs', 'feature_ranking.json')
    print('\n', "=" * 50)
    print("Starting Feature Removal Based on Random Forest Ranking...")
    if os.path.exists(ranking_file_path):
        with open(ranking_file_path, 'r') as file:
            ranking_data = json.load(file)
            if dataset_hash in ranking_data:
                print("Using existing feature ranking.")
                rank = ranking_data[dataset_hash]

                for i in range(len(rank)):
                    feature_name = rank[i][0]
                    importance = rank[i][1]
                    print(f"Feature {feature_name} ({importance:.6f})")

            else:
                rank = rank_features(X, y, feature_names, ranking_file_path, ranking_data, dataset_hash)

    else:
        rank = rank_features(X, y, feature_names, ranking_file_path, {}, dataset_hash)

    # Select features above the threshold from rank list
    deleted_indices = rank[-n_features_to_remove:] if n_features_to_remove > 0 else []

    columns_to_remove = [feature[0] for feature in deleted_indices]
    X_selected = X.drop(columns=columns_to_remove)


    # Print deleted features
    print("\nDeleted features:")
    for i in range(n_features_to_remove):
        print(f"Feature {rank[len(rank) - n_features_to_remove + i][0]} ({rank[len(rank) - n_features_to_remove + i][1]:.6f})")

    print("\nFeature selection completed.")
    return X_selected, deleted_indices


def train_and_evaluate_crossval(X, y, n_splits):
    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # Train the SVM model with RBF kernel and class weights
    svm = SVC(kernel='rbf', class_weight=class_weights_dict, random_state=42)

    start_time = time.time()

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = cross_val_predict(svm, X, y, cv=cv)

    end_time = time.time()

    # Evaluate the model's performance
    report = classification_report(y, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y, predictions)
    error_rate = 1 - accuracy_score(y, predictions)
    training_time = end_time - start_time

    return report, conf_matrix, training_time, error_rate


def train_and_evaluate(X_train, y_train, X_test, y_test, random_state):
    # Train the SVM model with RBF kernel
    svm = SVC(kernel='rbf', random_state=random_state)

    # Measure training time
    start_time = time.time()
    svm.fit(X_train, y_train)
    end_time = time.time()

    # Make predictions on the test set
    predictions = svm.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, predictions)
    error_rate = 1 - accuracy
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)
    training_time = end_time - start_time

    return report, conf_matrix, training_time, error_rate


def run_simulation_train_test_split(lab_data_path, removed_features, n_samples_list=None, n_features_to_remove=0,
                                    pca_retained_variance=0.95, max_gmm_component=10, test_size=0.2, test_mode=False):
    if n_samples_list is None:
        if test_mode:
            n_samples_list = [2 ** 7, 2**10, 2**13]
        else:
            n_samples_list = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15]

    simulation_begin = datetime.now().isoformat()

    # Load and preprocess the data
    X, y, feature_names = load_data(lab_data_path, removed_features)

    # Calculate the hash of the dataset and select features
    dataset_hash = calculate_hash(lab_data_path)
    X, deleted_indices = select_features(X, y, n_features_to_remove, feature_names, dataset_hash)
    n_features = X.shape[1]

    # initialize lists to store the results
    n_reports_list = []
    error_rate_list = []

    # Simulation parameters
    print_simulation_parameters(X, removed_features, n_samples_list, pca_retained_variance, max_gmm_component, test_mode)

    # Run simulations for different sample sizes
    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):
        if test_mode:
            rounds_factor = 2 ** 7
        else:
            rounds_factor = 2 ** 13

        r = math.floor(rounds_factor / np.sqrt(n_samples)) if n_samples < rounds_factor else 4
        errors = []

        # Initialize accumulators for metrics
        total_training_time = 0
        total_conf_matrix = np.zeros((2, 2), dtype=int)
        total_classification_report = None
        total_optimal_n_components_class_0 = 0
        total_optimal_n_components_class_1 = 0
        n_features_after_pca = None
        for round_num in tqdm(range(r), desc=f"Simulations for N={n_samples}", leave=False):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=round_num)

            # Separate the training data based on class
            X_class_0 = X_train[y_train == 0]
            X_class_1 = X_train[y_train == 1]

            # standardize
            scaler = StandardScaler()
            X_class_0 = scaler.fit_transform(X_class_0)
            X_class_1 = scaler.transform(X_class_1)
            X_test = scaler.transform(X_test)

            # Estimate optimal number of components for each class
            optimal_n_components_class_0 = estimate_n_components(X_class_0, max_gmm_component, round_num)
            optimal_n_components_class_1 = estimate_n_components(X_class_1, max_gmm_component, round_num)


            # Generate synthetic data
            X_train, y_train = generate_synthetic_data(X_class_0, X_class_1, n_samples,
                                                       optimal_n_components_class_0,
                                                       optimal_n_components_class_1)

            # Apply PCA to retain the specified variance
            n_features_after_pca = None
            if pca_retained_variance:
                pca = PCA(n_components=pca_retained_variance)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                n_features_after_pca = X_train.shape[1]


            # Train and evaluate the model
            report, conf_matrix, training_time, error_rate = train_and_evaluate(X_train, y_train, X_test, y_test, round_num)

            # Accumulate metrics
            errors.append(error_rate)
            total_training_time += training_time
            total_conf_matrix += conf_matrix
            total_optimal_n_components_class_0 += optimal_n_components_class_0
            total_optimal_n_components_class_1 += optimal_n_components_class_1

            if total_classification_report is None:
                total_classification_report = report
            else:
                for label, metrics in report.items():
                    if isinstance(metrics, dict):  # Ensure metrics is a dictionary
                        if label not in total_classification_report:
                            total_classification_report[label] = metrics
                        else:
                            for metric, value in metrics.items():
                                total_classification_report[label][metric] += value

        # Average error rate for the current sample size
        mean_error_rate = np.mean(errors)

        # Calculate mean values for the metrics
        mean_training_time = total_training_time / r
        mean_conf_matrix = total_conf_matrix / r
        mean_classification_report = {label: {metric: value / r for metric, value in metrics.items()}
                                      for label, metrics in total_classification_report.items() if
                                      isinstance(metrics, dict)}
        mean_optimal_n_components_class_0 = total_optimal_n_components_class_0/r
        mean_optimal_n_components_class_1 = total_optimal_n_components_class_1/r

        # Convert NumPy arrays to lists
        mean_conf_matrix = mean_conf_matrix.tolist()
        for label, metrics in mean_classification_report.items():
            if isinstance(metrics, dict):
                for metric in metrics:
                    mean_classification_report[label][metric] = float(mean_classification_report[label][metric])

        # Print summarized report for the current sample size
        print_summarized_report(n_samples, n_features, r, mean_error_rate, mean_training_time,
                                mean_classification_report, mean_conf_matrix)

        # Extract precision and f-score for class 0
        precision_class_0 = mean_classification_report["0"]["precision"]
        fscore_class_0 = mean_classification_report["0"]["f1-score"]
        precision_class_1 = mean_classification_report["1"]["precision"]
        fscore_class_1 = mean_classification_report["1"]["f1-score"]

        # Store the results
        n_report_summary = {
            "sample_size": n_samples,
            "rounds": r,
            "mean_error_rate": mean_error_rate,
            "mean_training_time": mean_training_time,
            "mean_precision_class_0": precision_class_0,
            "mean_fscore_class_0": fscore_class_0,
            "mean_precision_class_1": precision_class_1,
            "mean_fscore_class_1": fscore_class_1,
            "mean_optimal_n_components_class_0": mean_optimal_n_components_class_0,
            "mean_optimal_n_components_class_1": mean_optimal_n_components_class_1,
            "mean_confusion_matrix": mean_conf_matrix,
            "mean_num_folds": None,
            "num_features_after_pca": n_features_after_pca,
            "mean_classification_report": mean_classification_report
        }
        error_rate_list.append(mean_error_rate)
        n_reports_list.append(n_report_summary)

    # Print and save the final comparison report
    simulation_report = print_simulation_report(n_reports_list)


    # Save the simulation results to a JSON file
    simulation_data = {
        "n_features": n_features,
        "begin_time": simulation_begin,
        "end_time": datetime.now().isoformat(),
        "duration(h)": (datetime.now() - datetime.fromisoformat(simulation_begin)).total_seconds()/3600,
        "pca_retained_variance": pca_retained_variance,
        "test_size": test_size,
        "max_gmm_component": max_gmm_component,
        "n_samples_list": n_samples_list,
        "error_rate_list": error_rate_list,
        "deleted_features": deleted_indices,
        "simulation_report": simulation_report.to_dict(),
    }
    store_simulation_results(simulation_data)





def run_simulation_crossval(lab_data_path, corrompidas, output_file, n_features_to_remove=0, pca_variance=0.99, n_samples_list=None):
    if n_samples_list is None:
        n_samples_list = [2 ** 10, 2 ** 12, 2 ** 13]

    # Load and preprocess the data
    X, y, feature_names = load_data(lab_data_path, corrompidas)

    total_samples = len(y)


    dataset_hash = calculate_hash(lab_data_path)
    X, deleted_indices = select_features(X, y, n_features_to_remove, feature_names, dataset_hash)

    n_features_before_pca = X.shape[1]

    # Separate the training data based on class
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    # Calculate mean and covariance for each class
    mean_empirical_0 = np.mean(X_class_0, axis=0)
    cov_empirical_0 = np.cov(X_class_0, rowvar=False)

    mean_empirical_1 = np.mean(X_class_1, axis=0)
    cov_empirical_1 = np.cov(X_class_1, rowvar=False)

    # Print dataset mean and covariance
    print_feature_statistics(mean_empirical_0, cov_empirical_0, mean_empirical_1, cov_empirical_1)

    final_reports = []
    classification_reports = []

    # Simulation parameters
    print_simulation_parameters(X, corrompidas, X, n_samples_list)

    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):
        r = math.floor(2 ** 14 / np.sqrt(n_samples)) if n_samples < 2 ** 14 else 4
        errors = []

        # Initialize accumulators for metrics
        total_training_time = 0
        total_conf_matrix = np.zeros((2, 2), dtype=int)
        total_classification_report = None

        max_n_splits = 10
        min_n_splits = 2
        n_splits = math.floor((max_n_splits / (
                0.9 * total_samples)) * n_samples) if max_n_splits / total_samples * n_samples > min_n_splits else min_n_splits


        for round_num in tqdm(range(r), desc=f"Simulations for N={n_samples}", leave=False):

            n_samples_to_generate = int((n_splits/(n_splits-1))*n_samples)

            # Generate synthetic data with a specific random state for each round
            X_synthetic, y_synthetic = generate_synthetic_data(mean_empirical_0, cov_empirical_0, mean_empirical_1, cov_empirical_1, n_samples_to_generate, round_num)

            # Apply PCA to retain the specified variance
            pca = PCA(n_components=pca_variance)
            X_synthetic = pca.fit_transform(X_synthetic)

            n_features_after_pca = X_synthetic.shape[1]

            # Train and evaluate the model
            report, conf_matrix, training_time, error_rate = train_and_evaluate_crossval(X_synthetic, y_synthetic, n_splits)

            # Accumulate metrics
            errors.append(error_rate)
            total_training_time += training_time
            total_conf_matrix += conf_matrix
            if total_classification_report is None:
                total_classification_report = report
            else:
                for label, metrics in report.items():
                    if isinstance(metrics, dict):  # Ensure metrics is a dictionary
                        if label not in total_classification_report:
                            total_classification_report[label] = metrics
                        else:
                            for metric, value in metrics.items():
                                total_classification_report[label][metric] += value

        # Average error rate for the current sample size
        mean_error_rate = np.mean(errors)

        # Calculate mean values for the metrics
        mean_training_time = total_training_time / r
        mean_conf_matrix = total_conf_matrix / r
        mean_classification_report = {label: {metric: value / r for metric, value in metrics.items()}
                                      for label, metrics in total_classification_report.items() if
                                      isinstance(metrics, dict)}

        # Convert NumPy arrays to lists
        mean_conf_matrix = mean_conf_matrix.tolist()
        for label, metrics in mean_classification_report.items():
            if isinstance(metrics, dict):
                for metric in metrics:
                    mean_classification_report[label][metric] = float(mean_classification_report[label][metric])

        # Print summarized report for the current sample size
        print_summarized_report(n_samples, n_features_after_pca, r,
                                mean_training_time, mean_classification_report, mean_conf_matrix, mean_error_rate)

        # Extract precision and f-score for class 0
        precision_class_0 = mean_classification_report["0"]["precision"]
        fscore_class_0 = mean_classification_report["0"]["f1-score"]
        precision_class_1 = mean_classification_report["1"]["precision"]
        fscore_class_1 = mean_classification_report["1"]["f1-score"]

        # Store the results
        report_summary = {
            "sample_size": n_samples,
            "number_of_features_before_pca": n_features,
            "number_of_features_after_pca": n_features_after_pca,
            "mean_error_rate": mean_error_rate,
            "mean_training_time": mean_training_time,
            "mean_precision_class_0": precision_class_0,
            "mean_fscore_class_0": fscore_class_0,
            "mean_precision_class_1": precision_class_1,
            "mean_fscore_class_1": fscore_class_1,
            "mean_confusion_matrix": mean_conf_matrix,
            "number_of_folds": n_splits
        }
        final_reports.append(report_summary)
        classification_reports.append(mean_classification_report)

    # Print and save the final comparison report
    final_report_df = print_simulation_report(final_reports)

    # Save the simulation results to a JSON file
    simulation_data = {
        "n_samples_list": n_samples_list,
        "deleted_features": deleted_indices,
        "num_features_after_pca": n_features_after_pca,
        "num_features_before_pca": n_features_before_pca,
        "final_report_df": final_report_df.to_dict(),
        "pca_variance": pca_variance,
        "classification_reports": [report for report in classification_reports]
    }
    store_simulation_results(output_file, simulation_data)

    # Plot the error curve
    plot_metrics(n_samples_list, final_report_df, n_features_before_pca)

