import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import itertools
from matplotlib.patches import Ellipse
import os

def load_data(lab_data_path, features_to_remove):
    # Load the dataset
    lab_data = pd.read_csv(lab_data_path)

    # Remove corrupted subcarriers
    X = lab_data.drop(columns=['rotulo'] + features_to_remove)
    y = lab_data['rotulo'].apply(lambda x: 1 if x == 'ofensivo' else 0)

    return X, y, X.columns


def estimate_gmm_parameters(X, max_gmm_components=10):
    n_components_range = range(1, max_gmm_components+1)
    bic = []
    models = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        models.append(gmm)
    best_model = models[np.argmin(bic)]
    return best_model.means_, best_model.covariances_, best_model.weights_, len(best_model.means_)

def plot_ellipses(ax, means, covariances, weights, colors):
    for mean, covar, weight, color in zip(means, covariances, weights, colors):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = Ellipse(mean, v[0], v[1], angle=180.0 + angle, edgecolor=color, facecolor=color, linewidth=2)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

def plot_features_2d(X, y, feature_names, output_path):
    feature_combinations = list(itertools.combinations(feature_names, 2))
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    classes = np.unique(y)

    for (feature1, feature2) in feature_combinations:
        fig, ax = plt.subplots(figsize=(10, 8))  # Increased figure size for better resolution
        colors = itertools.cycle(plt.get_cmap('tab10').colors)

        for cls in classes:
            class_data = X_scaled[y == cls]
            X_cls = class_data[[feature1, feature2]].values
            means, covariances, weights, n_components = estimate_gmm_parameters(X_cls)
            color = next(colors)
            ax.scatter(X_cls[:, 0], X_cls[:, 1], s=0.8, label=f'Class {cls} (components: {n_components})', color=color)
            plot_ellipses(ax, means, covariances, weights, [color]*len(means))

        ax.set_title(f'{feature1} vs {feature2}')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.legend()

        filename = f'{feature1}_vs_{feature2}.png'
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath, dpi=300)  # Save figure with high resolution
        plt.close(fig)
    return output_path


def main():
    lab_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Lab.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'outputs', 'datapoints')
    removed_features = ['rssi', 'amp3', 'amp4', 'amp5', 'amp6', 'amp7', 'amp9', 'amp10',
                        'amp11', 'amp12', 'amp13', 'amp14', 'amp15', 'amp16', 'amp17', 'amp18',
                        'amp19', 'amp20', 'amp21', 'amp23', 'amp24', 'amp25', 'amp26', 'amp27',
                        'amp28', 'amp30', 'amp31', 'amp32', 'amp33', 'amp34', 'amp35', 'amp36',
                        'amp37', 'amp38', 'amp39', 'amp40', 'amp41', 'amp42', 'amp43', 'amp44',
                        'amp45', 'amp46', 'amp47', 'amp49', 'amp50', 'amp51', 'amp52']

    X, y, feature_names = load_data(lab_data_path, removed_features)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plot_features_2d(X, y, feature_names, output_path)


if __name__ == '__main__':
    main()
