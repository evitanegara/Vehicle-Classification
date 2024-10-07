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

# Project Result
- CNN Accuracy: 96.87%, outperforming FCNN (74.32%).
- Machine learning models like SVM and Random Forest, utilizing feature extraction techniques (HOG, LBP, LPQ, and Gabor features), outperformed deep learning, with accuracies exceeding 95%.
- Feature Extraction: Techniques like HOG, LBP, and Gabor significantly boosted machine learning performance.
- The stacking classifier achieved the best performance, with an accuracy of 99.37%, demonstrating the effectiveness of combining models for robust predictions.
- CNN outperformed FCNN in image classification due to its ability to automatically extract spatial patterns, but machine learning models still surpassed deep learning in this project due to effective feature extraction.
- Hyperparameter Tuning: Enhanced model accuracy by optimizing learning rates and hidden units.professionals in making informed decisions regarding property values.

# Conlusion
The project successfully classified vehicles, with machine learning models like SVM and Random Forest surpassing 95% accuracy. The stacking model achieved 99.37%, while CNN reached 96.87%. Hyperparameter tuning improved model performance, and transfer learning could enhance future deep learning results.

# Contact
For any questions or inquiries, please contact evitanegara@gmail.com 




