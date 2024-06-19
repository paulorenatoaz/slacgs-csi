import math
import os
import json
import hashlib
import threading
import time
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


def print_summarized_report(n_samples, training_time, report, conf_matrix, error_rate,
                            n_features_no_pca):
    print(f"\n\nReport for N={n_samples}:")
    print(f"Training Time: {training_time:.2f} seconds")
    # print(f"Number of Features after PCA: {n_features}")
    print(f"Classification Report:\n {pd.DataFrame(report).transpose()}")
    print(f"Confusion Matrix:\n {conf_matrix}")
    print(f"Error Rate: {error_rate:.4f}")
    print(f"Number of Features before PCA: {n_features_no_pca}")


def print_final_report(final_report_df):
    print("\nFinal Comparison Report:")
    print(final_report_df)


def plot_error_curve(n_samples_list, final_report_df, n_features):
    plt.figure(figsize=(10, 6))

    # Plot the f-score for class 0
    plt.plot(n_samples_list, final_report_df["Mean F-Score Class 0"], marker='s', linestyle='--', label='F-Score Class 0')

    # Plot the precision for class 0
    plt.plot(n_samples_list, final_report_df["Mean Precision Class 0"], marker='x', linestyle='--',
             label='Precision Class 0')

    plt.plot(n_samples_list, final_report_df["Mean Error Rate"], marker='o', linestyle='--', label='error rate')
    plt.title('Error Rate, Precision Class 0, and F-Score Class 0 vs. Number of Samples for ' + str(n_features) + ' features')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Error Rate')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(10 ** 2, 10 ** 5)
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'graphs', 'real_data_error_curve_' + str(n_features) + '_features.png')

    #create directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def store_analysis_results(file_path, analysis_data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        json.dump(analysis_data, file, indent=4)


def train_and_evaluate_train_test(X_train, X_test, y_train, y_test, use_pca=True):

    # Converting to arrays
    X_test = X_test.values
    y_test = y_test.values

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features_pca = False
    # Apply PCA to retain the specified variance
    if use_pca:
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        n_features_pca = X_train.shape[1]

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

    return report, conf_matrix, training_time, error_rate, n_features_pca


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

    return report, conf_matrix, training_time, predictions, error_rate



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


def create_balanced_subset(class1, class2, subset_size, n_components_class1, n_components_class2, component_labels1, component_labels2, random_state=42):
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
    # Convert to numpy arrays
    class1 = np.array(class1)
    class2 = np.array(class2)

    # Calculate original class proportions
    total_samples = len(class1) + len(class2)
    proportion_class1 = len(class1) / total_samples


    X_subset = []
    y_subset = []

    # Calculate the number of samples to draw from each class
    samples_class1 = int(np.round(subset_size * proportion_class1))
    samples_class2 = subset_size - samples_class1

    # Calculate the number of samples to draw from each component for class 1
    for component in range(n_components_class1):
        component_indices = np.where(component_labels1 == component)[0]
        n_samples_component = len(component_indices)
        n_samples_to_draw = int(np.round(samples_class1 * n_samples_component / len(class1)))
        sampled_indices = np.random.choice(component_indices, n_samples_to_draw, replace=False)
        X_subset.extend(class1[sampled_indices])
        y_subset.extend([0] * n_samples_to_draw)  # Label for class 1

    # Calculate the number of samples to draw from each component for class 2
    for component in range(n_components_class2):
        component_indices = np.where(component_labels2 == component)[0]
        n_samples_component = len(component_indices)
        n_samples_to_draw = int(np.round(samples_class2 * n_samples_component / len(class2)))
        sampled_indices = np.random.choice(component_indices, n_samples_to_draw, replace=False)
        X_subset.extend(class2[sampled_indices])
        y_subset.extend([1] * n_samples_to_draw)  # Label for class 2

    # Convert lists to numpy arrays
    X_subset, y_subset = np.array(X_subset), y_subset

    return X_subset, y_subset


def run_analysis_train_test(lab_data_path, json_output_path, excluded_sub_carr, n_features_to_remove, n_samples_list=None):
    n_samples_list = [2 ** 10, 2 ** 11, 2 ** 12] if n_samples_list is None else n_samples_list

    X, y, feature_names = load_data(lab_data_path, excluded_sub_carr)
    print_dataset_report(X, excluded_sub_carr, y, feature_names)

    dataset_hash = calculate_hash(lab_data_path)
    X, deleted_indices = select_features(X, y, n_features_to_remove , feature_names, dataset_hash)


    # store the original number of features before PCA
    n_features = X.shape[1]
    final_reports = []
    classification_reports = []
    total_samples = len(y)

    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):

        r = math.floor(2 ** 14 / np.sqrt(n_samples)) if n_samples < 2 ** 14 else 4
        errors = []

        # Initialize accumulators for metrics
        total_training_time = 0
        total_conf_matrix = np.zeros((2, 2), dtype=int)
        total_classification_report = None


        for round_num in tqdm(range(r), desc=f"Analysis for N={n_samples}", leave=False):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=round_num)

            # Separate the training data based on class
            X_class_0 = X_train[y_train == 0]
            X_class_1 = X_train[y_train == 1]


            # get random values from set for both classes
            np.random.seed(round_num)

            optimal_components_class0_bic, optimal_components_class0_aic = estimate_n_components(X_class_0,
                                                                                                 max_components=10)
            optimal_components_class1_bic, optimal_components_class1_aic = estimate_n_components(X_class_1,
                                                                                                 max_components=10)

            optimal_components_class0 = optimal_components_class0_bic
            optimal_components_class1 = optimal_components_class1_bic

            component_labels0, component_labels1 = get_gmm_components_labels(X_class_0, X_class_1,
                                                                             optimal_components_class0,
                                                                             optimal_components_class1)

            X_train, y_train = create_balanced_subset(X_class_0, X_class_1, n_samples,
                                                                            optimal_components_class0,
                                                                            optimal_components_class1,
                                                                            component_labels0, component_labels1)


            # Train and evaluate the model
            report, conf_matrix, training_time, error_rate, n_features_pca = train_and_evaluate_train_test(X_train,
                                                                                                           X_test,
                                                                                                           y_train,
                                                                                                           y_test)

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


        # Extract precision and f-score for class 0
        precision_class_0 = mean_classification_report["0"]["precision"]
        fscore_class_0 = mean_classification_report["0"]["f1-score"]

        # Store the results
        report_summary = {
            "Number of Features before PCA": n_features,
            "Sample Size": n_samples,
            "Mean Error Rate": average_error_rate,
            "Number of Features after PCA": n_features,
            "Mean Training Time (s)": mean_training_time,
            "Mean Precision Class 0": precision_class_0,
            "Mean F-Score Class 0": fscore_class_0,
            "Mean Confusion Matrix": mean_conf_matrix,
        }
        final_reports.append(report_summary)

        # Collect classification reports for JSON storage
        classification_report_filtered = {label: metrics for label, metrics in report.items() if
                                          isinstance(metrics, dict)}
        classification_reports.append({**classification_report_filtered, "Sample Size": n_samples})

        # Print summarized report for the current sample size
        print_summarized_report(n_samples, training_time, report, conf_matrix, error_rate, n_features)

    # Final comparison report
    final_report_df = pd.DataFrame(final_reports)
    print_final_report(final_report_df)

    # Save analysis results to a JSON file
    analysis_data = {
        "n_features_before_pca": n_features,
        "n_samples_list": n_samples_list,
        "deleted_features": deleted_indices,
        "n_features_after_pca": n_features,
        "final_report_df": final_reports,
        "pca_variance": 0.95,
        "classification_reports": classification_reports
    }
    store_analysis_results(json_output_path, analysis_data)

    # Plot the error curve
    plot_error_curve(n_samples_list, final_report_df, n_features)

def estimate_n_components(data, max_components=10):
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
		gmm = GaussianMixture(n_components=n_components, random_state=42)
		gmm.fit(data)
		bics.append(gmm.bic(data))
		aics.append(gmm.aic(data))

	optimal_components_bic = n_components_range[np.argmin(bics)]
	optimal_components_aic = n_components_range[np.argmin(aics)]

	return optimal_components_bic, optimal_components_aic


def run_analysis_crossval(lab_data_path, json_output_path, corrompidas, n_features_to_remove, n_samples_list=None):
    n_samples_list = [2 ** 10, 2 ** 11, 2 ** 12] if n_samples_list is None else n_samples_list

    X, y, feature_names = load_data(lab_data_path, corrompidas)
    print_dataset_report(X, corrompidas, y, feature_names)

    dataset_hash = calculate_hash(lab_data_path)
    X, deleted_indices = select_features(X, y, n_features_to_remove , feature_names, dataset_hash)

    # store the original number of features before PCA
    n_features_no_pca = X.shape[1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # # Apply PCA to retain the specified variance
    # pca = PCA(n_components=0.95)
    # X = pca.fit_transform(X)
    # n_features = X.shape[1]

    final_reports = []
    classification_reports = []

    class_counts = y.value_counts()
    total_samples = len(y)

    # Separate the training data based on class
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    optimal_components_class0_bic, optimal_components_class0_aic = estimate_n_components(X_class_0, max_components=10)
    optimal_components_class1_bic, optimal_components_class1_aic = estimate_n_components(X_class_1, max_components=10)

    optimal_components_class0 = optimal_components_class0_bic
    optimal_components_class1 = optimal_components_class1_bic

    component_labels0, component_labels1 = get_gmm_components_labels(X_class_0, X_class_1, optimal_components_class0, optimal_components_class1)



    for n_samples in tqdm(n_samples_list, desc="Sample Size Progress"):

        max_n_splits = 10
        min_n_splits = 2
        n_splits = math.floor((max_n_splits/(0.9*total_samples)) * n_samples) if max_n_splits/total_samples * n_samples > min_n_splits else min_n_splits
        n_samples_subset = int(n_splits / (n_splits - 1)) * n_samples


        r = math.floor(2 ** 10 / np.sqrt(n_samples)) if n_samples < 2 ** 10 else 4
        errors = []

        # Initialize accumulators for metrics
        total_training_time = 0
        total_conf_matrix = np.zeros((2, 2), dtype=int)
        total_classification_report = None

        for round_num in tqdm(range(r), desc=f"Analysis for N={n_samples}", leave=False):

            # get random values from set for both classes
            np.random.seed(round_num)

            X_selected_samples, y_selected_samples = create_balanced_subset(X_class_0, X_class_1, n_samples_subset, optimal_components_class0,
                                   optimal_components_class1, component_labels0, component_labels1)

            # Train and evaluate the model
            report, conf_matrix, training_time, predictions, error_rate = train_and_evaluate_crossval(
                X_selected_samples, y_selected_samples, n_splits)

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


        # Extract precision and f-score for class 0
        precision_class_0 = mean_classification_report["0"]["precision"]
        fscore_class_0 = mean_classification_report["0"]["f1-score"]

        # Store the results
        report_summary = {
            "Sample Size": n_samples,
            "Number of Features before PCA": n_features_no_pca,
            # "Number of Features after PCA": n_features,
            "Mean Error Rate": average_error_rate,
            "Mean Training Time (s)": mean_training_time,
            "Mean Precision Class 0": precision_class_0,
            "Mean F-Score Class 0": fscore_class_0,
            "Mean Confusion Matrix": mean_conf_matrix,
            "n_splits": n_splits
        }
        final_reports.append(report_summary)

        # Collect classification reports for JSON storage
        classification_report_filtered = {label: metrics for label, metrics in report.items() if
                                          isinstance(metrics, dict)}
        classification_reports.append({**classification_report_filtered, "Sample Size": n_samples})

        # Print summarized report for the current sample size
        print_summarized_report(n_samples, training_time, report, conf_matrix, error_rate, n_features_no_pca)

    # Final comparison report
    final_report_df = pd.DataFrame(final_reports)
    print_final_report(final_report_df)



    # Save analysis results to a JSON file
    analysis_data = {
        "n_samples_list": n_samples_list,
        "deleted_features": deleted_indices,
        "n_features_before_pca": n_features_no_pca,
        # "n_features_after_pca": n_features,
        "final_report_df": final_reports,
        "pca_variance": 0.95,
        "classification_reports": classification_reports
    }
    store_analysis_results(json_output_path, analysis_data)

    # Plot the error curve
    plot_error_curve(n_samples_list, final_report_df, n_features_no_pca)

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
        run_analysis_train_test(lab_data_path, json_output_path, excluded_sub_carr, n_features_to_remove, n_samples_list)