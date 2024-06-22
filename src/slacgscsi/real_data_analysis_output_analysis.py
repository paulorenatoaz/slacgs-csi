import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from .config import PROJECT_DIR


def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


def prepare_dataframe(data):
    df_list = []
    ids = []
    for entry in data:
        if entry['analysis_data']['test_size']:
            analysis_data = entry['analysis_data']
            ids.append(entry['id'])
            for n_samples, metrics in zip(analysis_data['n_samples_list'], zip(analysis_data['error_rate_list'], analysis_data['analysis_report']['mean_precision_class_0'].values(),
                                                analysis_data['analysis_report']['mean_fscore_class_0'].values(), analysis_data['analysis_report']['mean_precision_class_1'].values(),
                                                analysis_data['analysis_report']['mean_fscore_class_1'].values(), analysis_data['analysis_report']['mean_training_time'].values())):
                df_list.append({
                    'n_features': analysis_data['n_features'],
                    'n_samples': n_samples,
                    'error_rate': metrics[0],
                    'precision_class_0': metrics[1],
                    'fscore_class_0': metrics[2],
                    'precision_class_1': metrics[3],
                    'fscore_class_1': metrics[4],
                    'training_time': metrics[5],
                    'duration(h)': analysis_data['duration(h)'],
                    'test_size': analysis_data['test_size'],
                    'max_gmm_components': analysis_data['max_gmm_components'],
                    'pca_retained_variance': analysis_data['pca_retained_variance'],
                })
        df = pd.DataFrame(df_list)
    return df, ids

def plot_metrics_vs_n(df, ids):
    metrics = ['error_rate', 'precision_class_0', 'fscore_class_0', 'precision_class_1', 'fscore_class_1']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for n_features in df['n_features'].unique():
            subset = df[df['n_features'] == n_features]
            plt.plot(subset['n_samples'], subset[metric], marker='o', linestyle='-', label=f'{n_features} features')
        plt.xscale('log')
        plt.xlabel('Number of Samples (log scale)')
        plt.ylabel(metric.replace('_', ' ').title())
        crossvalidation = True if not df['test_size'].unique() else False
        test_size = df['test_size'].unique()[0]
        pca_retained_variance = df['pca_retained_variance'].unique()[0]
        max_gmm_components = df['max_gmm_components'].unique()[0]
        analysis_method = 'cross-validation' if crossvalidation else 'train_test_split=' + str(
            test_size)
        plt.title(f'Real Data Analysis {ids}\n{metric.replace("_", " ").title()} vs Number of Samples\n ' +
                  'pca = ' + str(pca_retained_variance) + '; max_gmm_comp = ' + str(max_gmm_components) +
                  '; method: ' + analysis_method)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PROJECT_DIR, 'data', 'outputs', 'graphs',  f'real_data_analysis_{ids}_{metric}_vs_n.png'))
        plt.show()

def plot_metrics_vs_dimensions(df, ids):
    metrics = ['error_rate', 'precision_class_0', 'fscore_class_0']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for n_samples in df['n_samples'].unique():
            subset = df[df['n_samples'] == n_samples]
            plt.plot(subset['n_features'], subset[metric], marker='o', linestyle='-', label=f'{n_samples} samples')
        plt.xlabel('Number of Features')
        plt.ylabel(metric.replace('_', ' ').title())

        test_size = df['test_size'].unique()[0]
        pca_retained_variance = df['pca_retained_variance'].unique()[0]
        max_gmm_components = df['max_gmm_components'].unique()[0]
        crossvalidation = True if not df['test_size'].unique() else False
        analysis_method = 'cross-validation' if crossvalidation else 'train_test_split=' + str(
            test_size)
        plt.title(f'Real Data Analysis {ids}\n{metric.replace("_", " ").title()} vs Number of Features\n' +
                  'pca = ' + str(pca_retained_variance) + '; max_gmm_comp = ' + str(max_gmm_components) +
                  '; method: ' + analysis_method)

        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PROJECT_DIR, 'data', 'outputs', 'graphs',  f'real_data_analysis_{ids}_{metric}_vs_dim.png'))
        plt.show()



def plot_duration_vs_dimensions(df, ids):
    plt.figure(figsize=(10, 6))

    plt.plot(df['n_features'], df['duration(h)'], marker='o', linestyle='-')
    plt.xlabel('Number of Features')
    plt.ylabel('Duration (hours)')

    test_size = df['test_size'].unique()[0]
    pca_retained_variance = df['pca_retained_variance'].unique()[0]
    max_gmm_components = df['max_gmm_components'].unique()[0]
    crossvalidation = True if not df['test_size'].unique() else False
    analysis_method = 'cross-validation=' if crossvalidation else 'train_test_split=' + str(
        test_size)
    plt.title(f'Real Data Analysis {ids}\nDuration vs Number of Features\n' +
              'pca = ' + str(pca_retained_variance) + '; max_gmm_comp = ' + str(max_gmm_components) +
              '; method: ' + analysis_method)

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PROJECT_DIR, 'data', 'outputs', 'graphs',  f'real_data_analysis_{ids}_duration_vs_dim.png'))
    plt.show()

def plot_training_time_vs_n(df, ids):
    plt.figure(figsize=(10, 6))
    for n_features in df['n_features'].unique():
        subset = df[df['n_features'] == n_features]
        plt.plot(subset['n_samples'], subset['training_time'], marker='o', linestyle='-', label=f'{n_features} features')
    plt.xscale('log')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Training Time (seconds)')

    test_size = df['test_size'].unique()[0]
    pca_retained_variance = df['pca_retained_variance'].unique()[0]
    max_gmm_components = df['max_gmm_components'].unique()[0]
    crossvalidation = True if not df['test_size'].unique() else False
    analysis_method = 'cross-validation=' if crossvalidation else 'train_test_split=' + str(
        test_size)
    plt.title(f'Real Data Analysis {ids}\nTraining Time vs Number of Samples\n' +
              'pca = ' + str(pca_retained_variance) + '; max_gmm_comp = ' + str(max_gmm_components) +
              '; method: ' + analysis_method)

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PROJECT_DIR, 'data', 'outputs', 'graphs',  f'real_data_analysis_{ids}_training_time_vs_n.png'))
    plt.show()

def plot_training_time_vs_dimensions(df, ids):
    plt.figure(figsize=(10, 6))
    for n_samples in df['n_samples'].unique():
        subset = df[df['n_samples'] == n_samples]
        plt.plot(subset['n_features'], subset['training_time'], marker='o', linestyle='-', label=f'{n_samples} samples')
    plt.xlabel('Number of Features')
    plt.ylabel('Training Time (seconds)')

    test_size = df['test_size'].unique()[0]
    pca_retained_variance = df['pca_retained_variance'].unique()[0]
    max_gmm_components = df['max_gmm_components'].unique()[0]
    crossvalidation = True if not df['test_size'].unique() else False
    analysis_method = 'cross-validation=' if crossvalidation else 'train_test_split=' + str(
        test_size)
    plt.title(f'Real Data Analysis {ids}\nTraining Time vs Number of Features\n' +
              'pca = ' + str(pca_retained_variance) + '; max_gmm_comp = ' + str(max_gmm_components) +
              '; method: ' + analysis_method)

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PROJECT_DIR, 'data', 'outputs', 'graphs',  f'real_data_analysis_{ids}_training_time_vs_dim.png'))
    plt.show()

def generate_real_data_analysis_plots():
    json_file_path = os.path.join(PROJECT_DIR, 'data', 'outputs', 'analysis_results.json')
    data = load_data(json_file_path)
    df, ids = prepare_dataframe(data)

    # plot_metrics_vs_n(df, ids)
    # plot_metrics_vs_dimensions(df, ids)

    plot_duration_vs_dimensions(df, ids)
    plot_training_time_vs_n(df, ids)
    plot_training_time_vs_dimensions(df, ids)

