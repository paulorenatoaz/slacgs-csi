import math
import os
import json
import hashlib
import threading
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from .config import PROJECT_DIR


def load_data(lab_data_path, corrompidas):
    # Load the dataset
    lab_data = pd.read_csv(lab_data_path)

    # Remove corrupted subcarriers
    X = lab_data.drop(columns=['rotulo'] + corrompidas)
    y = lab_data['rotulo'].apply(lambda x: 1 if x == 'ofensivo' else 0)

    # Normalize the data



    return X, y, X.columns


def calculate_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def print_progress(start_time, running_event, prefix=""):
    while running_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"\r{prefix} Training Time: {elapsed_time:.3f} seconds", end="")
        time.sleep(0.1)


def select_features(X, y, n_features_to_remove, feature_names, dataset_hash):
    ranking_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'feature_ranking.json')

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


def rank_features(X, y, feature_names, ranking_file_path, ranking_data, dataset_hash):
    print("\nStarting feature selection...")
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


def print_dataset_report(X, corrompidas, y, feature_names):
    print("Dataset shape:", X.shape)
    print("Corrupted subcarriers removed:", corrompidas)
    print("Class distribution:\n", y.value_counts())
    print("Features:\n", feature_names)


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


def print_analysis_report(analysis_report):
    analysis_report_df = pd.DataFrame(analysis_report)
    print("=" * 50)
    print("Simulation Report:")
    print(analysis_report_df.drop(columns=['mean_classification_report']))
    print("=" * 50)
    return analysis_report_df


def plot_metrics(analysis_report, n_features, n_samples_list, id, test_size, pca_retained_variance, max_gmm_components):
    plt.figure(figsize=(10, 6))

    # Plot the error rate
    plt.plot(n_samples_list, list(analysis_report["mean_error_rate"].values()), marker='o', linestyle='--', label='Error Rate', color='blue')

    # Plot the precision for class 0
    plt.plot(n_samples_list, list(analysis_report["mean_precision_class_0"].values()), marker='x', linestyle='--', label='Precision Class 0', color='orange')

    # Plot the f-score for class 1
    plt.plot(n_samples_list, list(analysis_report["mean_fscore_class_0"].values()), marker='s', linestyle='--', label='F-Score Class 0', color='orange')

    # Plot the precision for class 1
    plt.plot(n_samples_list, list(analysis_report["mean_precision_class_1"].values()), marker='x', linestyle='--',
             label='Precision Class 1', color='green')

    # Plot the f-score for class 0
    plt.plot(n_samples_list, list(analysis_report["mean_fscore_class_1"].values()), marker='s', linestyle='--',
             label='F-Score Class 1', color='green')

    crossvalidation = False if analysis_report["num_folds"][0] is None else True
    analysis_method = 'cross-validation' if crossvalidation else 'train_test_split=' + str(test_size)
    plt.title('Error Rate, Precision, and F-Score vs. Number of Samples for ' + str(n_features) + ' features\n' +
              'pca = ' + str(pca_retained_variance) + '; max_gmm_comp = ' + str(max_gmm_components) +
              '; method: ' + analysis_method)
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Metrics')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(10 ** 2, 10 ** 5)
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'graphs',
                               'analysis_'+ str(id) +'_metrics-vs-N_' + str(n_features) + '-feat' + '.png')

    # create directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def store_analysis_results(analysis_data):
    output_file_path = os.path.join(PROJECT_DIR, 'data', 'outputs', 'analysis_results.json')

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

    plot_image_path = plot_metrics(analysis_data['analysis_report'], analysis_data['n_features'],
                                   analysis_data['n_samples_list'], current_id, analysis_data['test_size'],
                                   analysis_data['pca_retained_variance'], analysis_data['max_gmm_components'])

    analysis_data['plot_image_path'] = plot_image_path

    # Add the id to the analysis data
    new_entry = {
        "id": current_id,
        "analysis_data": analysis_data
    }

    # Append the new data
    existing_data.append(new_entry)

    # Write back the updated data to the JSON file
    with open(output_file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


def train_and_evaluate_train_test(X_train, X_test, y_train, y_test):


    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # Train the SVM model with RBF kernel and class weights
    svm = SVC(kernel='rbf', class_weight=class_weights_dict, random_state=42)

    # Display progress
    start_time = time.time()
    svm.fit(X_train, y_train)
    end_time = time.time()

    # Make predictions on the test set
    predictions = svm.predict(X_test)

    # Evaluate the model's performance
    report = classification_report(y_test, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_test, predictions)
    error_rate = 1 - accuracy_score(y_test, predictions)
    training_time = end_time - start_time

    return report, conf_matrix, training_time, error_rate


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



def get_gmm_components_labels(class1, class2, n_components_class1, n_components_class2):
    # Fit GMM to each class
    gmm1 = GaussianMixture(n_components=n_components_class1, random_state=42)
    gmm1.fit(class1)

    gmm2 = GaussianMixture(n_components=n_components_class2, random_state=42)
    gmm2.fit(class2)

    # Predict component membership for each sample
    component_labels1 = gmm1.predict(class1)
    component_labels2 = gmm2.predict(class2)

    return component_labels1, component_labels2


def create_balanced_subset(X_class_0, X_class_1, subset_size, optimal_n_components_class_0, optimal_n_components_class_1):
    """
    Create a balanced subset of the dataset based on the GMM components for each class,
    maintaining the original class proportion.

    Parameters:
    - class1: array-like, shape (n_samples_class1, n_features)
      Data points for class 1.
    - class2: array-like, shape (n_samples_class2, n_features)
      Data points for class 2.
    - subset_size: int
      Size of the subset to generate.
    - n_components: int
      Number of mixture components to use for GMM.
    - random_state: int
      Random seed for reproducibility.

    Returns:
    - X_subset: array, shape (subset_size, n_features)
      Concatenated subset of feature matrix.
    - y_subset: array, shape (subset_size,)
      Concatenated subset of labels.
    """
    component_labels_0, component_labels_1 = get_gmm_components_labels(X_class_0, X_class_1,
                                                                     optimal_n_components_class_0,
                                                                     optimal_n_components_class_1)
    # Convert to numpy arrays
    X_class_0 = np.array(X_class_0)
    X_class_1 = np.array(X_class_1)

    # Calculate original class proportions
    total_samples = len(X_class_0) + len(X_class_1)
    proportion_class1 = len(X_class_0) / total_samples

    X_subset = []
    y_subset = []

    # Calculate the number of samples to draw from each class
    samples_class1 = int(np.round(subset_size * proportion_class1))
    samples_class2 = subset_size - samples_class1


    # Calculate the number of samples to draw from each component for class 1
    for component in range(optimal_n_components_class_0):
        component_indices = np.where(component_labels_0 == component)[0]
        n_samples_component = len(component_indices)
        n_samples_to_draw = int(np.round(samples_class1 * n_samples_component / len(X_class_0)))
        n_samples_to_draw = min(n_samples_to_draw, len(X_class_0))
        sampled_indices = np.random.choice(component_indices, n_samples_to_draw)
        X_subset.extend(X_class_0[sampled_indices])
        y_subset.extend([0] * n_samples_to_draw)  # Label for class 1

    # Calculate the number of samples to draw from each component for class 2
    for component in range(optimal_n_components_class_1):
        component_indices = np.where(component_labels_1 == component)[0]
        n_samples_component = len(component_indices)
        n_samples_to_draw = int(np.round(samples_class2 * n_samples_component / len(X_class_1)))
        n_samples_to_draw = min(n_samples_to_draw, len(X_class_1))
        sampled_indices = np.random.choice(component_indices, n_samples_to_draw)
        X_subset.extend(X_class_1[sampled_indices])
        y_subset.extend([1] * n_samples_to_draw)  # Label for class 2

    # Convert lists to numpy arrays
    X_subset, y_subset = np.array(X_subset), y_subset

    return X_subset, y_subset


def run_real_data_analysis_train_test_split(lab_data_path, removed_features, n_samples_list=None, n_features_to_remove=0,
                                    pca_retained_variance=0.95, max_gmm_components=10, test_size=0.2, test_mode=False):
    if n_samples_list is None:
        if test_mode:
            n_samples_list = [2 ** 7, 2 ** 10, 2 ** 13]
        else:
            n_samples_list = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15]

    analysis_begin = datetime.now().isoformat()

    X, y, feature_names = load_data(lab_data_path, removed_features)
    print_dataset_report(X, removed_features, y, feature_names)

    # Calculate the hash of the dataset and select features
    dataset_hash = calculate_hash(lab_data_path)
    X, deleted_indices = select_features(X, y, n_features_to_remove , feature_names, dataset_hash)

    # store the original number of features before PCA
    n_features = X.shape[1]

    # initialize lists to store the results
    n_reports_list = []
    error_rate_list = []

    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):
        if test_mode:
            rounds_factor = 2 ** 7
        else:
            rounds_factor = 2 ** 12

        # set number of round
        r = math.floor(rounds_factor / np.sqrt(n_samples)) if n_samples < rounds_factor else 4

        # Initialize accumulators for metrics
        errors = []
        total_training_time = 0
        total_conf_matrix = np.zeros((2, 2), dtype=int)
        total_classification_report = None
        total_optimal_n_components_class_0 = 0
        total_optimal_n_components_class_1 = 0
        n_features_after_pca = None

        for round_num in tqdm(range(r), desc=f"Analysis for N={n_samples}", leave=False):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=round_num)

            # Separate the training data based on class
            X_class_0 = X_train[y_train == 0]
            X_class_1 = X_train[y_train == 1]

            # Estimate optimal number of components for each class
            optimal_n_components_class_0 = estimate_n_components(X_class_0, max_gmm_components, round_num)
            optimal_n_components_class_1 = estimate_n_components(X_class_1, max_gmm_components, round_num)

            np.random.seed(round_num)
            X_train, y_train = create_balanced_subset(X_class_0, X_class_1, n_samples,
                                                      optimal_n_components_class_0,
                                                      optimal_n_components_class_1)


            if pca_retained_variance:
                pca = PCA(n_components=pca_retained_variance)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test.to_numpy())
                n_features_after_pca = X_train.shape[1]

            # Train and evaluate the model
            report, conf_matrix, training_time, error_rate = train_and_evaluate_train_test(X_train, X_test,
                                                                                           y_train, y_test.to_numpy())

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
        mean_optimal_n_components_class_0 = total_optimal_n_components_class_0 / r
        mean_optimal_n_components_class_1 = total_optimal_n_components_class_1 / r

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
            "num_folds": None,
            "num_features_after_pca": n_features_after_pca,
            "mean_classification_report": mean_classification_report
        }
        error_rate_list.append(mean_error_rate)
        n_reports_list.append(n_report_summary)

    # Print and save the final comparison report
    analysis_report = print_analysis_report(n_reports_list)

    # Save the analysis results to a JSON file
    analysis_data = {
        "n_features": n_features,
        "begin_time": analysis_begin,
        "end_time": datetime.now().isoformat(),
        "duration(h)": (datetime.now() - datetime.fromisoformat(analysis_begin)).total_seconds() / 3600,
        "pca_retained_variance": pca_retained_variance,
        "test_size": test_size,
        "max_gmm_components": max_gmm_components,
        "n_samples_list": n_samples_list,
        "error_rate_list": error_rate_list,
        "deleted_features": deleted_indices,
        "analysis_report": analysis_report.to_dict(),
    }
    store_analysis_results(analysis_data)


def estimate_n_components(data, max_components, random_state=13):
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


def run_real_data_analysis_crossval(lab_data_path, removed_features, n_samples_list=None, n_features_to_remove=0,
                                    pca_retained_variance=0.95, max_gmm_components=10, test_size=None, test_mode=False):
    if n_samples_list is None:
        if test_mode:
            n_samples_list = [2 ** 7, 2 ** 10, 2 ** 13]
        else:
            n_samples_list = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15]

    analysis_begin = datetime.now().isoformat()

    X, y, feature_names = load_data(lab_data_path, removed_features)
    print_dataset_report(X, removed_features, y, feature_names)

    # Calculate the hash of the dataset and select features
    dataset_hash = calculate_hash(lab_data_path)
    X, deleted_indices = select_features(X, y, n_features_to_remove, feature_names, dataset_hash)

    # Separate the training data based on class
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    # store the original number of features before PCA
    n_features = X.shape[1]

    # Estimate optimal number of components for each class
    optimal_n_components_class_0 = estimate_n_components(X_class_0, max_gmm_components)
    optimal_n_components_class_1 = estimate_n_components(X_class_1, max_gmm_components)

    # initialize lists to store the results
    n_reports_list = []
    error_rate_list = []

    total_samples = len(y)
    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):

        if test_mode:
            rounds_factor = 2 ** 7
        else:
            rounds_factor = 2 ** 12

        # set number of round
        r = math.floor(rounds_factor / np.sqrt(n_samples)) if n_samples < rounds_factor else 4

        max_n_splits = 10
        min_n_splits = 2
        n_folds = math.floor((max_n_splits/(0.9*total_samples)) * n_samples) if max_n_splits/total_samples * n_samples > min_n_splits else min_n_splits
        n_samples_subset = int(n_folds / (n_folds - 1)) * n_samples

        # Initialize accumulators for metrics
        errors = []
        total_training_time = 0
        total_conf_matrix = np.zeros((2, 2), dtype=int)
        total_classification_report = None
        total_optimal_n_components_class_0 = 0
        total_optimal_n_components_class_1 = 0
        n_features_after_pca = None

        for round_num in tqdm(range(r), desc=f"Analysis for N={n_samples}", leave=False):

            np.random.seed(round_num)
            X_train, y_train = create_balanced_subset(X_class_0, X_class_1, n_samples_subset,
                                                      optimal_n_components_class_0,
                                                      optimal_n_components_class_1)

            if pca_retained_variance:
                pca = PCA(n_components=pca_retained_variance)
                X_train = pca.fit_transform(X_train)
                n_features_after_pca = X_train.shape[1]

                # Train and evaluate the model
                report, conf_matrix, training_time, error_rate = train_and_evaluate_crossval(X_train, y_train, n_folds)

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
        mean_optimal_n_components_class_0 = total_optimal_n_components_class_0 / r
        mean_optimal_n_components_class_1 = total_optimal_n_components_class_1 / r

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
            "num_folds": n_folds,
            "num_features_after_pca": n_features_after_pca,
            "mean_classification_report": mean_classification_report
        }
        error_rate_list.append(mean_error_rate)
        n_reports_list.append(n_report_summary)

    # Print and save the final comparison report
    analysis_report = print_analysis_report(n_reports_list)

    # Save the analysis results to a JSON file
    analysis_data = {
        "n_features": n_features,
        "begin_time": analysis_begin,
        "end_time": datetime.now().isoformat(),
        "duration(h)": (datetime.now() - datetime.fromisoformat(analysis_begin)).total_seconds() / 3600,
        "pca_retained_variance": pca_retained_variance,
        "test_size": test_size,
        "max_gmm_components": max_gmm_components,
        "n_samples_list": n_samples_list,
        "error_rate_list": error_rate_list,
        "deleted_features": deleted_indices,
        "analysis_report": analysis_report.to_dict(),
    }
    store_analysis_results(analysis_data)
if __name__ == "__main__":
    lab_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Lab.csv')
    json_output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'analysis_results.json')
    excluded_sub_carr = ['rssi', 'amp3', 'amp4', 'amp5', 'amp6', 'amp7', 'amp9', 'amp10',
                  'amp11', 'amp12', 'amp13', 'amp14', 'amp15', 'amp16', 'amp17', 'amp18',
                  'amp19', 'amp20', 'amp21', 'amp23', 'amp24', 'amp25', 'amp26', 'amp27',
                  'amp28', 'amp30', 'amp31', 'amp32', 'amp33', 'amp34', 'amp35', 'amp36',
                  'amp37', 'amp38', 'amp39', 'amp40', 'amp41', 'amp42', 'amp43', 'amp44',
                  'amp45', 'amp46', 'amp47', 'amp49', 'amp50', 'amp51', 'amp52']
    n_samples_list = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11,  2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15]
    # n_samples_list = [2 ** 8, 2 ** 9, 2 ** 10]
    # n_samples_list = [2 ** 10, 2 ** 11, 2 ** 12]

    # for n_features_to_remove in range(4, -1,-1):
    #     run_analysis_crossval(lab_data_path, json_output_path, excluded_sub_carr, n_features_to_remove, n_samples_list=n_samples_list)

    for n_features_to_remove in range(4, -1,-1):
        run_real_data_analysis_train_test_split(lab_data_path, json_output_path, excluded_sub_carr, n_features_to_remove, n_samples_list)