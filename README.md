# Vehicle Classification for Intelligent Traffic Systems Using Machine Learning and Deep Learning
# Project Overview
This project applies traditional machine learning and deep learning approaches to the problem of multi-class image classification, focusing on vehicle identification, a critical component of Intelligent Traffic Systems (ITS). The ability to accurately classify vehicle types plays a key role in optimizing toll systems, traffic monitoring, and parking utilization in smart cities. Using a dataset of 4,356 vehicle images across five classes (sedan, pick-up, SUV, hatchback, and others), the project evaluates handcrafted feature-based machine learning models (SVM, KNN, Random Forest, Voting, Stacking) and deep learning architectures (Fully Connected Neural Network, CNN). The study highlights the impact of feature engineering, data augmentation, and model tuning on classification accuracy, generalization, and real-world deployment potential.

## Dataset Overview

- Source: [Mendeley Data – Vehicle Classification Dataset](https://data.mendeley.com/datasets/htsngg9tpc/3)
- Location: Captured at Loei Rajabhat University, Thailand
- Volume: 4,356 labeled vehicle images
- Classes: Sedan, Pick-up, SUV, Hatchback, Other (includes motorcycles and vans)

| Vehicle Type | Description                              |
|--------------|------------------------------------------|
| Sedan        | Standard passenger vehicles              |
| Pick-up      | Light-duty trucks                        |
| SUV          | Sport Utility Vehicles                   |
| Hatchback    | Compact cars with rear cargo access      |
| Other        | Includes motorcycles and small vans      |


## Executive Summary
This project demonstrates how handcrafted features such as HOG, LBP, LPQ, and Gabor can be leveraged to outperform deep learning models in small to medium scale image datasets. The best performing model was the Stacking Classifier (Accuracy: 99.37%), followed by KNN (99.00%) and the Voting Classifier (98.60%). CNNs reached 96.87% accuracy, clearly outperforming the Fully Connected Neural Network (74.32%) but falling short of feature based models due to limited data. Data augmentation (brightness and rotation), stratified K-Fold cross-validation, and hyperparameter tuning significantly enhanced model performance. The results emphasize that in scenarios with limited training data, machine learning paired with strong feature engineering can outperform even deep learning architectures.

## Project Workflow
### Data Import and Feature Engineering
- Imported libraries: NumPy, OpenCV, Skimage, Pandas, Scikit-learn
- Implemented five feature extraction techniques:
  - HOG: Captured edge orientations using gradient histograms
  - LBP: Extracted local texture patterns from pixel neighborhoods
  - LPQ: Generated blur-invariant phase-based descriptors
  - Gabor: Applied multi-scale frequency and orientation filters for texture recognition
  - Color Histogram: Represented image color distributions

### Data Preprocessing
- Resized all images to 256×256 pixels
- Converted images to grayscale to ensure consistency during feature extraction
- Created image augmentation functions:
  - Brightness adjustment (range: 0.5 to 2.0)
  - Random rotation between -15° and 15°
- Applied augmentation only to training data to avoid test leakage
  <p align="center">
  <img src="https://github.com/user-attachments/assets/0527a4f6-1197-47e0-bb78-32f8b41bdaeb" alt="Model 2 Forecast" width="700"/>
</p>

### Dataset Splitting
- Performed an 80/20 train-test split
- Used Stratified K-Fold cross-validation (k = 10) to maintain class balance
- Augmentation effectively doubled the size of the training set

### Model Building and Tuning
### Machine Learning
- Trained the following models using handcrafted features:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest (RF)
- Optimized hyperparameters using GridSearchCV with 10-fold cross-validation:
  - SVM: C values, kernel types
  - KNN: Number of neighbors, weights, distance metrics
  - RF: Number of estimators, max depth, split criteria
- Constructed ensemble models:
  - Voting Classifier for majority-based predictions
  - Stacking Classifier using a meta-model to combine base learners

### Deep Learning
- CNN:
  - Three Conv2D layers with MaxPooling and ReLU activation
  - Dense output layer for final classification
  - Hyperparameters tuned: optimizer type, dropout rate, learning rate
- FCNN:
  - Fully connected neural network with multiple dense layers
  - Tuned for optimal learning rate and number of hidden units
- Applied Dropout and BatchNormalization to reduce overfitting and stabilize training

### Model Performance Summary
  | Model               | Accuracy (%) |
|--------------------|--------------|
| Stacking Classifier | **99.37**    |
| K-Nearest Neighbor  | 99.00        |
| Voting Classifier   | 98.60        |
| Random Forest       | 98.00        |
| SVM                 | 94.00        |
| CNN                 | 96.87        |
| FCNN                | 74.32        |

  
## Highlights
- Stacking Classifier achieved the highest overall accuracy at 99.37%, offering the most reliable classification performance across all vehicle types.
- K-Nearest Neighbor (KNN) delivered strong results with 99.00% accuracy, benefiting from robust handcrafted features like HOG and LPQ.
- Voting Classifier reached 98.60% accuracy, effectively balancing prediction stability and class generalization.
- Random Forest maintained high performance at 98.00%, especially effective when paired with Gabor filters and color histograms.
- CNN was the top-performing deep learning model, achieving 96.87% accuracy through effective spatial feature learning and augmentation techniques
- SVM achieved 94.00% accuracy, demonstrating solid generalization with well-tuned polynomial and RBF kernels.
- Fully Connected Neural Network (FCNN) showed the weakest performance at 74.32%, limited by its inability to capture spatial hierarchies in image data.

## Key Takeaways
- Feature Engineering Drives Performance : Machine learning models like KNN, SVM, and Random Forest performed exceptionally well due to the use of handcrafted features (HOG, LBP, LPQ, Gabor, color histograms), capturing shape, texture, and color with high fidelity.
- Stacking and Voting Boost Reliability : Ensemble techniques significantly improved prediction robustness and accuracy. The Stacking Classifier was the top performer by combining the strengths of multiple base models, while Voting offered stability and reduced misclassifications.
- CNNs Excel with Augmentation : Despite a relatively small dataset, CNNs learned spatial patterns effectively through convolutional layers, and augmentation techniques (rotation, brightness adjustment) enhanced generalization to unseen data.
- Model Choice Should Align with Data : Machine learning models outperformed deep learning due to the effectiveness of manual feature extraction. For smaller datasets, traditional ML paired with strong features is often more powerful than end-to-end deep learning.
- FCNNs Lack Spatial Awareness : The Fully Connected Neural Network underperformed, highlighting the importance of spatial feature handling for image-based tasks. CNNs' architectural advantage in spatial learning made them far more suitable for this task.
- Cross-Validation Ensures Generalization: The use of Stratified K-Fold and GridSearchCV allowed robust hyperparameter tuning, helping models generalize better and avoid overfitting, even with class imbalance and limited data.

# Contact
For any questions or inquiries, please contact evitanegara@gmail.com 




