import os
import json
import hashlib
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def load_data(lab_data_path, corrompidas):
    # Load the dataset
    lab_data = pd.read_csv(lab_data_path)

    # Remove corrupted subcarriers
    X = lab_data.drop(columns=['rotulo'] + corrompidas)
    y = lab_data['rotulo'].apply(lambda x: 1 if x == 'ofensivo' else 0)

    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y, X.columns

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

def select_features(X, y, threshold, feature_names, dataset_hash):
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
                rank = perform_feature_selection(X, y, feature_names, ranking_file_path, ranking_data, dataset_hash)

    else:
        rank = perform_feature_selection(X, y, feature_names, ranking_file_path, {}, dataset_hash)

    # Select features above the threshold from rank list
    selected_indices = [i for i in range(len(rank)) if float(rank[i][1]) > threshold]
    deleted_indices = [i for i in range(len(rank)) if i not in selected_indices]

    X_selected = X[:, selected_indices]

    # Print deleted features
    print("\nDeleted features:")
    for i in deleted_indices:
        print(f"Feature {rank[i][0]} ({rank[i][1]})")

    print("\nFeature selection completed.")
    return X_selected, selected_indices

def perform_feature_selection(X, y, feature_names, ranking_file_path, ranking_data, dataset_hash):
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

def train_and_evaluate(X_train, X_test, y_train, y_test, use_pca):

    n_features = X_train.shape[1]

    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # Train the SVM model with RBF kernel and class weights
    svm = SVC(kernel='rbf', class_weight=class_weights_dict, random_state=42)

    print("\n")  # Add a newline to avoid overlapping with the progress bar
    print(f"Training SVM model with {'PCA' if use_pca else 'out PCA'}...")
    # Display progress
    start_time = time.time()
    running_event = threading.Event()
    running_event.set()
    progress_thread = threading.Thread(target=print_progress, args=(start_time, running_event))
    progress_thread.start()

    svm.fit(X_train, y_train)

    running_event.clear()
    progress_thread.join()
    end_time = time.time()

    # Make predictions on the test set
    predictions = svm.predict(X_test)

    # Evaluate the model's performance
    report = classification_report(y_test, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_test, predictions)
    error_rate = 1 - accuracy_score(y_test, predictions)
    training_time = end_time - start_time

    return report, conf_matrix, training_time, n_features, predictions, error_rate

def print_summarized_report(n_samples, training_time_pca, n_features_pca, report_pca, conf_matrix_pca, error_rate_pca,
                            training_time_no_pca, n_features_no_pca, report_no_pca, conf_matrix_no_pca, error_rate_no_pca):
    print(f"\n\nReport for N={n_samples}:")
    print(f"Training Time with PCA: {training_time_pca:.2f} seconds")
    print(f"Number of Features with PCA: {n_features_pca}")
    print(f"Classification Report with PCA:\n {pd.DataFrame(report_pca).transpose()}")
    print(f"Confusion Matrix with PCA:\n {conf_matrix_pca}")
    print(f"Error Rate with PCA: {error_rate_pca:.4f}")
    print(f"\nTraining Time without PCA: {training_time_no_pca:.2f} seconds")
    print(f"Number of Features without PCA: {n_features_no_pca}")
    print(f"Classification Report without PCA:\n {pd.DataFrame(report_no_pca).transpose()}")
    print(f"Confusion Matrix without PCA:\n {conf_matrix_no_pca}")
    print(f"Error Rate without PCA: {error_rate_no_pca:.4f}")

def print_final_report(final_report_df):
    print("\nFinal Comparison Report:")
    print(final_report_df)

def plot_error_curve(n_samples_list, final_report_df):
    plt.figure(figsize=(10, 6))

    # Plot the f-score for class 0
    plt.plot(n_samples_list, final_report_df["F-Score Class 0"], marker='s', linestyle='--', label='F-Score Class 0')

    # Plot the precision for class 0
    plt.plot(n_samples_list, final_report_df["Precision Class 0"], marker='x', linestyle='--',
             label='Precision Class 0')

    plt.plot(n_samples_list, final_report_df["Error Rate PCA"], marker='o', linestyle='--', label='PCA')
    plt.plot(n_samples_list, final_report_df["Error Rate No PCA"], marker='x', linestyle='--', label='No PCA')
    plt.title('Error Rate vs. Number of Samples')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Error Rate')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(10 ** 2, 10 ** 5)
    plt.grid(True)
    plt.legend()
    plt.show()


def store_analysis_results(file_path, analysis_data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(analysis_data, file, indent=4)


def run_analysis(lab_data_path, json_output_path, corrompidas, threshold=0.01, n_samples_list=None):
    n_samples_list = [2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12] if n_samples_list is None else n_samples_list

    X_normalized, y, feature_names = load_data(lab_data_path, corrompidas)
    print_dataset_report(X_normalized, corrompidas, y, feature_names)

    dataset_hash = calculate_hash(lab_data_path)
    X_selected, selected_indices = select_features(X_normalized, y, threshold, feature_names, dataset_hash)

    # Create a DataFrame with the original indices
    X_selected_df = pd.DataFrame(X_selected)

    # Split the data into training and testing sets
    X_train_df, X_test_df, y_train, y_test, train_indices, test_indices = train_test_split(
        X_selected_df, y, X_selected_df.index, test_size=0.3, random_state=42, stratify=y)

    # Convert back to numpy arrays for PCA and further processing
    X_train = X_train_df.values
    X_test = X_test_df.values

    # Apply PCA to retain the specified variance
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Separate the training data based on class
    X_pca_class_0 = X_train_pca[y_train == 0]
    X_pca_class_1 = X_train_pca[y_train == 1]

    # Calculate mean and covariance for each class
    mean_empirical_0 = np.mean(X_pca_class_0, axis=0)
    cov_empirical_0 = np.cov(X_pca_class_0, rowvar=False)
    mean_empirical_1 = np.mean(X_pca_class_1, axis=0)
    cov_empirical_1 = np.cov(X_pca_class_1, rowvar=False)


    final_reports = []
    classification_reports = []

    class_counts = y_train.value_counts()
    total_samples = len(y_train)
    proportion_class_0 = class_counts[0] / total_samples
    proportion_class_1 = class_counts[1] / total_samples

    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):
        n_samples_class_0 = int(n_samples * proportion_class_0)
        n_samples_class_1 = int(n_samples * proportion_class_1)

        indices_class_0 = np.where(y_train == 0)[0][:n_samples_class_0]
        indices_class_1 = np.where(y_train == 1)[0][:n_samples_class_1]
        selected_indices = np.hstack((indices_class_0, indices_class_1))

        X_selected_samples = X_train[selected_indices]
        X_selected_samples_pca = X_train_pca[selected_indices]
        y_selected_samples = y_train.iloc[selected_indices]

        # Train and evaluate the model with PCA
        report_pca, conf_matrix_pca, training_time_pca, n_features_pca, predictions_pca, error_rate_pca = train_and_evaluate(
            X_selected_samples_pca, X_test_pca, y_selected_samples, y_test, use_pca=True)

        # Train and evaluate the model without PCA
        report_no_pca, conf_matrix_no_pca, training_time_no_pca, n_features_no_pca, predictions_no_pca, error_rate_no_pca = train_and_evaluate(
            X_selected_samples, X_test, y_selected_samples, y_test, use_pca=False)

        # Extract precision and f-score for class 0
        precision_class_0 = report_pca["0"]["precision"]
        fscore_class_0 = report_pca["0"]["f1-score"]

        # Store the results
        report_summary = {
            "Sample Size": n_samples,
            "Training Time PCA (s)": training_time_pca,
            "Number of Features PCA": n_features_pca,
            "Error Rate PCA": error_rate_pca,
            "Error Rate No PCA": error_rate_no_pca,
            "Precision Class 0": precision_class_0,
            "F-Score Class 0": fscore_class_0,
            "Confusion Matrix PCA": conf_matrix_pca.tolist()
        }
        final_reports.append(report_summary)

        # Collect classification reports for JSON storage
        classification_report_filtered = {label: metrics for label, metrics in report_pca.items() if
                                          isinstance(metrics, dict)}
        classification_reports.append({**classification_report_filtered, "Sample Size": n_samples})

        # Print summarized report for the current sample size
        print_summarized_report(n_samples, training_time_pca, n_features_pca, report_pca, conf_matrix_pca, error_rate_pca,
                                training_time_no_pca, n_features_no_pca, report_no_pca, conf_matrix_no_pca, error_rate_no_pca)

    # Final comparison report
    final_report_df = pd.DataFrame(final_reports)
    print_final_report(final_report_df)



    # Save analysis results to a JSON file
    analysis_data = {
        "selected_indices": sorted(X_selected_df.index[selected_indices].tolist()),  # Indices from the original X that are in X_train
        "pca_variance": 0.99,
        "mean_empirical_0": mean_empirical_0.tolist(),
        "cov_empirical_0": cov_empirical_0.tolist(),
        "mean_empirical_1": mean_empirical_1.tolist(),
        "cov_empirical_1": cov_empirical_1.tolist(),
        "num_features_after_pca": n_features_pca,
        "final_report_df": final_reports,
        "classification_reports": classification_reports
    }
    store_analysis_results(json_output_path, analysis_data)

    # Plot the error curve
    plot_error_curve(n_samples_list, final_report_df)

if __name__ == "__main__":
    lab_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Lab.csv')
    json_output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'analysis_results.json')
    corrompidas = ['rssi', 'amp2', 'amp8', 'amp22', 'amp29', 'amp48']
    # n_samples_list = [2 ** 8, 2 ** 10, 2 ** 12, 2 ** 13, 2 ** 14, 26269, 2 ** 15]
    run_analysis(lab_data_path, json_output_path, corrompidas)#, n_samples_list=n_samples_list)
