import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import math
from tqdm import tqdm
import time


def load_and_normalize_data(lab_data_path, corrompidas):
    # Load the dataset
    lab_data = pd.read_csv(lab_data_path)

    # Remove corrupted subcarriers
    X = lab_data.drop(columns=['rotulo'] + corrompidas)
    y = lab_data['rotulo'].apply(lambda x: 1 if x == 'ofensivo' else 0)

    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y, X.columns

def apply_pca(X_train, X_test, variance=0.99):
    # Apply PCA to retain specified variance
    pca = PCA(n_components=variance)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


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
    print(f"Total features after PCA: {X_train_pca.shape[1]}")
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


def plot_error_curve(n_samples_list, final_report_df):
    plt.figure(figsize=(10, 6))

    # Plot the error rate
    plt.plot(n_samples_list, final_report_df["Error Rate"], marker='o', linestyle='--', label='Error Rate')

    # Plot the precision for class 0
    plt.plot(n_samples_list, final_report_df["Precision Class 0"], marker='x', linestyle='--',
             label='Precision Class 0')

    # Plot the f-score for class 0
    plt.plot(n_samples_list, final_report_df["F-Score Class 0"], marker='s', linestyle='--', label='F-Score Class 0')

    plt.title('Error Rate, Precision Class 0, and F-Score Class 0 vs. Number of Samples')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Metrics')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(10 ** 2, 10 ** 5)
    plt.grid(True)
    plt.legend()
    plt.show()


def store_simulation_results(file_path, simulation_data):
    # Save the simulation results to a JSON file create directory if it does not exist

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        json.dump(simulation_data, file, indent=4)


def run_simulation(lab_data_path, corrompidas , output_file, test_size=0.3, pca_variance=0.99, n_samples_list=None):
    if n_samples_list is None:
        n_samples_list = [2 ** 10, 2 ** 12, 2 ** 13]#, 2 ** 14, 26269, 2 ** 15]

    # Load and preprocess the data
    X_normalized, y, feature_names = load_and_normalize_data(lab_data_path, corrompidas)

    # Create a DataFrame with the original indices
    X_normalized_df = pd.DataFrame(X_normalized)

    # Split the data into training and testing sets
    X_train_df, X_test_df, y_train, y_test, train_indices, test_indices = train_test_split(
        X_normalized_df, y, X_normalized_df.index, test_size=test_size, random_state=42, stratify=y)

    # Convert back to numpy arrays for PCA and further processing
    X_train = X_train_df.values
    X_test = X_test_df.values

    # Apply PCA to retain the specified variance
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, variance=pca_variance)

    # Separate the training data based on class
    X_pca_class_0 = X_train_pca[y_train == 0]
    X_pca_class_1 = X_train_pca[y_train == 1]

    # Calculate mean and covariance for each class
    mean_empirical_0 = np.mean(X_pca_class_0, axis=0)
    cov_empirical_0 = np.cov(X_pca_class_0, rowvar=False)

    mean_empirical_1 = np.mean(X_pca_class_1, axis=0)
    cov_empirical_1 = np.cov(X_pca_class_1, rowvar=False)

    # Print dataset mean and covariance
    print_feature_statistics(mean_empirical_0, cov_empirical_0, mean_empirical_1, cov_empirical_1)

    error_rates = []
    final_reports = []
    classification_reports = []

    # Simulation parameters
    print_simulation_parameters(X_normalized, corrompidas, X_train_pca, n_samples_list)

    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):
        r = math.floor(2 ** 10 / np.sqrt(n_samples)) if n_samples < 2 ** 10 else 4
        errors = []

        # Initialize accumulators for metrics
        total_training_time = 0
        total_conf_matrix = np.zeros((2, 2), dtype=int)
        total_classification_report = None

        for round_num in tqdm(range(r), desc=f"Simulations for N={n_samples}", leave=False):
            # Generate synthetic data with a specific random state for each round
            X_synthetic, y_synthetic = generate_synthetic_data(mean_empirical_0, cov_empirical_0, mean_empirical_1, cov_empirical_1, n_samples, round_num)

            # Train and evaluate the model
            report, conf_matrix, training_time, error_rate = train_and_evaluate(X_synthetic, y_synthetic, X_test_pca, y_test)

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
        error_rates.append(average_error_rate)

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
                                average_error_rate, X_train_pca.shape[1])

        # Extract precision and f-score for class 0
        precision_class_0 = mean_classification_report["0"]["precision"]
        fscore_class_0 = mean_classification_report["0"]["f1-score"]

        # Store the results
        report_summary = {
            "Sample Size": n_samples,
            "Mean Training Time (s)": mean_training_time,
            "Number of Features": X_train_pca.shape[1],
            "Error Rate": average_error_rate,
            "Precision Class 0": precision_class_0,
            "F-Score Class 0": fscore_class_0,
            "Mean Confusion Matrix": mean_conf_matrix
        }
        final_reports.append(report_summary)
        classification_reports.append(mean_classification_report)

    # Print and save the final comparison report
    final_report_df = print_final_report(final_reports)

    # Save the simulation results to a JSON file
    simulation_data = {
        "selected_indices": sorted(train_indices.tolist()), # Indices from the original X that are in X_train
        "pca_variance": pca_variance,
        "mean_empirical_0": mean_empirical_0.tolist(),
        "cov_empirical_0": cov_empirical_0.tolist(),
        "mean_empirical_1": mean_empirical_1.tolist(),
        "cov_empirical_1": cov_empirical_1.tolist(),
        "num_features_after_pca": X_train_pca.shape[1],
        "final_report_df": final_report_df.to_dict(),
        "classification_reports": [{**{label: {metric: value for metric, value in metrics.items()} for label, metrics in report.items()}, "Sample Size": n_samples} for report, n_samples in zip(classification_reports, n_samples_list)]
    }
    store_simulation_results(output_file, simulation_data)

    # Plot the error curve
    plot_error_curve(n_samples_list, final_report_df)


# Example usage
if __name__ == "__main__":
    lab_data_path = os.path.join(os.path.dirname(__file__), '..\\..\\data', 'Lab.csv')
    output_file_path = os.path.join(os.path.dirname(__file__), '..\\..\\data', 'outputs', 'simulation_results.json')
    corrompidas = ['rssi', 'amp2', 'amp8', 'amp22', 'amp29', 'amp48']
    n_samples_list = [2 ** 8, 2 ** 10, 2 ** 12, 2 ** 13, 2 ** 14, 26269, 2 ** 15]
    run_simulation(lab_data_path, corrompidas, output_file_path, n_samples_list = n_samples_list)
