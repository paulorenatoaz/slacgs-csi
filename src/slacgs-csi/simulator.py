import hashlib
import json
import os
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from scipy.stats import multivariate_normal
import math

from sklearn.utils import compute_class_weight
from tqdm import tqdm
import time


def load_and_normalize_data(lab_data_path, corrompidas):
    # Load the dataset
    lab_data = pd.read_csv(lab_data_path)

    # Remove corrupted subcarriers
    X = lab_data.drop(columns=['rotulo'] + corrompidas)
    y = lab_data['rotulo'].apply(lambda x: 1 if x == 'ofensivo' else 0)



    return X, y, X.columns



def generate_synthetic_data(mean_empirical_0, cov_empirical_0, mean_empirical_1, cov_empirical_1, n_samples, round_num):
    n_samples_per_class = n_samples // 2

    X_synthetic_0 = multivariate_normal.rvs(mean=mean_empirical_0, cov=cov_empirical_0, size=n_samples_per_class, random_state=round_num)
    X_synthetic_1 = multivariate_normal.rvs(mean=mean_empirical_1, cov=cov_empirical_1, size=n_samples_per_class, random_state=round_num + 1)

    X_synthetic = np.vstack((X_synthetic_0, X_synthetic_1))
    y_synthetic = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

    return X_synthetic, y_synthetic


def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Train the SVM model with RBF kernel
    svm = SVC(kernel='rbf')

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

def print_simulation_parameters(X_normalized, corrompidas, X_train_pca, n_samples_list):
    print(f"\nDataset shape: {X_normalized.shape}")
    print(f"Corrupted subcarriers removed: {corrompidas}")
    print(f"Total features before PCA: {X_train_pca.shape[1]}")
    print(f"Sample sizes to test: {n_samples_list}")


def print_summarized_report(n_samples, mean_training_time, mean_classification_report, mean_conf_matrix, average_error_rate, n_features):
    print(f"\nSummarized Report for N={n_samples}:")
    print(f"Mean Training Time: {mean_training_time:.2f} seconds")
    print(f"Mean Number of Features: {n_features}")
    print(f"Mean Classification Report:\n {pd.DataFrame(mean_classification_report).transpose()}")
    print(f"Mean Confusion Matrix:\n {mean_conf_matrix}")
    print(f"Error Rate: {average_error_rate:.4f}")


def print_final_report(final_reports):
    final_report_df = pd.DataFrame(final_reports)
    print("\nFinal Comparison Report:")
    print(final_report_df)
    return final_report_df


def plot_error_curve(n_samples_list, final_report_df, n_features):
    plt.figure(figsize=(10, 6))

    # Plot the error rate
    plt.plot(n_samples_list, final_report_df["Mean Error Rate"], marker='o', linestyle='--', label='Error Rate')

    # Plot the precision for class 0
    plt.plot(n_samples_list, final_report_df["Mean Precision Class 0"], marker='x', linestyle='--',
             label='Precision Class 0')

    # Plot the f-score for class 0
    plt.plot(n_samples_list, final_report_df["Mean F-Score Class 0"], marker='s', linestyle='--', label='F-Score Class 0')

    plt.title('Error Rate, Precision Class 0, and F-Score Class 0 vs. Number of Samples for ' + str(n_features) + ' features')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Metrics')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(10 ** 2, 10 ** 5)
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'graphs',
                               'simulation_error_curve_' + str(n_features) + '_features.png')

    # create directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def store_simulation_results(file_path, simulation_data):
    # Save the simulation results to a JSON file create directory if it does not exist

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'a') as file:
        json.dump(simulation_data, file, indent=4)


def train_and_evaluate_cross_val(X, y, n_splits):
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

def calculate_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def run_simulation(lab_data_path, corrompidas , output_file, n_features_to_remove=0, pca_variance=0.99, n_samples_list=None):
    if n_samples_list is None:
        n_samples_list = [2 ** 10, 2 ** 12, 2 ** 13]

    # Load and preprocess the data
    X, y, feature_names = load_and_normalize_data(lab_data_path, corrompidas)

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
            report, conf_matrix, training_time, error_rate = train_and_evaluate_cross_val(X_synthetic, y_synthetic, n_splits)

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
        average_error_rate = np.mean(errors)

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
        print_summarized_report(n_samples, mean_training_time, mean_classification_report, mean_conf_matrix,
                                average_error_rate, n_features_after_pca)

        # Extract precision and f-score for class 0
        precision_class_0 = mean_classification_report["0"]["precision"]
        fscore_class_0 = mean_classification_report["0"]["f1-score"]

        # Store the results
        report_summary = {
            "Sample Size": n_samples,
            "Number of Features before PCA": n_features_before_pca,
            "Number of Features after PCA": n_features_after_pca,
            "Mean Error Rate": average_error_rate,
            "Mean Training Time (s)": mean_training_time,
            "Mean Precision Class 0": precision_class_0,
            "Mean F-Score Class 0": fscore_class_0,
            "Mean Confusion Matrix": mean_conf_matrix,
            "n_splits": n_splits
        }
        final_reports.append(report_summary)
        classification_reports.append(mean_classification_report)

    # Print and save the final comparison report
    final_report_df = print_final_report(final_reports)

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
    plot_error_curve(n_samples_list, final_report_df, n_features_before_pca)


# Example usage
if __name__ == "__main__":
    lab_data_path = os.path.join(os.path.dirname(__file__), '..\\..\\data', 'Lab.csv')
    output_file_path = os.path.join(os.path.dirname(__file__), '..\\..\\data', 'outputs', 'simulation_results.json')
    corrompidas = [ 'amp2', 'amp8', 'amp22', 'amp29', 'amp48']
    # n_samples_list = [2 ** 10, 2 ** 12, 2 ** 13, 2 ** 14, 26269, 2 ** 15]
    n_samples_list = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15]
    # n_samples_list = [2 ** 8, 2 ** 9, 2 ** 10]
    for n_features_to_remove in range(0, 46):
        run_simulation(lab_data_path, corrompidas, output_file_path, n_features_to_remove=n_features_to_remove, n_samples_list=n_samples_list)
