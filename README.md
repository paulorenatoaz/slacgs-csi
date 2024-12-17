### slacgs-csi  
Simulator for Loss Analysis of Classifiers on Gaussian Samples based on CSI data  

### Optimizing Wi-Fi Security: A Study on Synthetic Data and Feature-Sample Trade-offs Using CSI Data

## Introduction  

This study introduces a novel approach to Wi-Fi network security by leveraging synthetic data simulation and Channel State Information (CSI) for intrusion detection.  

We utilize machine learning techniques to analyze the trade-off between the number of features and sample sizes, aiming to optimize detection accuracy.  

Our methodology includes:  
- Generating synthetic datasets that mimic real-world Wi-Fi environments  
  with the help of Gaussian Mixture Models learned from CSI data,  
  and performing simulations to evaluate the performance of classifiers  
  on the synthetic data.  

- Analyzing real-world CSI data collected from Wi-Fi networks  
  to evaluate the performance of classifiers across different scenarios  
  varying in feature sets and sample sizes.  

Our results demonstrate that synthetic data can effectively complement real-world data in enhancing Wi-Fi security, providing valuable insights into the optimal balance of features and samples for intrusion detection.  
Additionally, the analysis of the real-world data shows that the model can achieve high detection accuracy with a small number of features and samples.

## data  

This directory contains:

- The real-world CSI data collected from Wi-Fi networks for the analysis.  
- Outputs from the simulator for the synthetic data analysis  
  and from the real-world data analysis.  
- Graphs and plots generated from the real-world data analysis  
  and the synthetic data simulations.  

## real_data_analysis  

This module contains the code for the real-world data analysis using the CSI data collected from Wi-Fi networks.  

We evaluate the model for different sample sizes and feature sets to search for the optimal trade-off between the number of features and sample sizes.  

## simulator  

This module contains the code for the simulator that generates synthetic data based on Gaussian mixture models learned from the real-world data.
We evaluate the performance of classifiers on the synthetic data to analyze the impact of different feature sets and sample sizes on the detection accuracy.  
