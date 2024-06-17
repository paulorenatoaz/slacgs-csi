### slacgs-csi
Simulator for Loss Analysis of Classifiers on Gaussian Samples based on CSI data

### Optimizing Wi-Fi Security: A Study on Synthetic Data and Feature-Sample Trade-offs Using CSI

## Introduction

This study introduces a novel approach to Wi-Fi network security by leveraging synthetic data simulation and Channel 
State Information (CSI) for intrusion detection. We utilize machine learning techniques to analyze the trade-off between 
the number of features and sample sizes, aiming to optimize detection accuracy. Our methodology includes generating 
synthetic datasets that mimic real-world Wi-Fi environments and evaluating classifier performance across different 
scenarios. Our results demonstrate that synthetic data can effectively complement real-world data in enhancing 
Wi-Fi security, providing valuable insights into the optimal balance of features and samples for intrusion detection.

## src

This directory contains the source code for the simulator and the real-world data analysis. The simulator generates
synthetic data based on Gaussian samples and evaluates the performance of classifiers on the generated data. The real-world
data analysis uses CSI data collected from Wi-Fi networks to evaluate the performance of classifiers on real-world data.

## data

This directory contains:
- the real-world CSI data collected from Wi-Fi networks for the analysis. 
- outputs from the simulator for the synthetic data analysis and from the real-world data analysis.
- graphs and plots generated from the real-world data analysis and the synthetic data simulations.