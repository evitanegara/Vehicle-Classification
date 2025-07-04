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
  <img src="https://github.com/user-attachments/assets/0527a4f6-1197-47e0-bb78-32f8b41bdaeb" alt="Model 2 Forecast" width="500"/>
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
- Feature-based machine learning models outperformed deep learning approaches  
  Models using HOG, LPQ, and LBP achieved over 98% accuracy, outperforming CNNs due to effective manual feature extraction.

- CNNs remained competitive  
  Despite the limited dataset size, CNNs achieved 96.87% accuracy by learning local and global spatial hierarchies.

- Stacking was the best overall model  
  Achieved 99.37% accuracy with consistent sensitivity and specificity, making it the most reliable approach.

- KNN combined simplicity with high performance  
  Delivered 99% accuracy when paired with robust feature sets, showcasing its effectiveness in well-engineered pipelines.

- FCNN showed limitations  
  Struggled due to lack of spatial awareness, which CNNs handled better through convolutional learning mechanisms.


## Key Takeaways
- Feature engineering is critical  
  For small to moderately sized datasets, handcrafted features like HOG, Gabor, and LBP significantly boost performance.

- Ensemble models Improve Result 
  Voting and Stacking classifiers leverage multiple base learners, improving both accuracy and model stability.

- CNNs require more data  
  Deep learning approaches can rival or surpass traditional models when enough data and augmentation are available, but tend to underperform on limited datasets.

- Augmentation and cross-validation improve generalization  
  Techniques like brightness adjustments, random rotations, and stratified K-Fold splitting helped mitigate overfitting.

- Model selection depends on data structure  
  CNNs are well-suited for large-scale, raw image datasets, while traditional ML models perform best when paired with meaningful, engineered features.


# Contact
For any questions or inquiries, please contact evitanegara@gmail.com 




