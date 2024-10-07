# Business Understanding
 
Vehicle classification is a critical component of Intelligent Traffic Systems (ITS), contributing to the efficiency of smart cities by improving traffic management, reducing congestion, and enhancing road safety. Accurately identifying vehicle types helps optimize toll collection, traffic flow monitoring, and parking lot utilization.

# Project Objective
- The goal of this project is to classify five distinct types of vehicles.
- Various machine learning and deep learning models were applied, including: Support Vector Machine (SVM), k-nearest neighbor (k-NN), Random Forest, Ensemble Learning (Stacking and Voting) Convolutional Neural Network (CNN), Fully connected neural network
-  Dataset:  [Access the Dataset here](https://data.mendeley.com/datasets/htsngg9tpc/3)

# Project Overview
This project involved several key steps to classify five distinct types of vehicles:
- **Data Import and Feature Extraction**: Imported the dataset and used libraries like NumPy, OpenCV, and Sklearn. Employed techniques like HOG, color histograms, LBP, LPQ, and Gabor features for vehicle classification.
- **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets for model evaluation.
- **Pre-processing and Data Augmentation**: Resized images, converted them to grayscale, and applied augmentation (brightness adjustment and rotation).
- **Model Training and Evaluation**: Trained models including SVM, k-NN, CNN, and FCNN. Hyperparameter tuning using GridSearchCV was performed, and the models were evaluated using accuracy, precision, and recall metrics.



