# Project Overview
There are 2 components to this assignment.
## **Component 1** - Classification task on music recordings based on data in GTZAN dataset 
This component involves designing and training a feedforward deep neural network (DNN) for a classification task. 
The DNN consists of three hidden layers with 128 neurons each, using ReLU activation and dropout of 0.2. 
The output layer uses sigmoid activation for binary classification.

### Model Architecture
Hidden Layers: Three layers with 128 neurons each, using ReLU activation and dropout of 0.2.
Output Layer: Sigmoid activation for binary classification.
Optimizer: Adam optimizer with a learning rate of 0.001.
Loss Function: Binary Cross-Entropy (BCE) loss.

### Training Process
Dataset Split: Training set (70%) and testing set (30%).
Optimal Batch Size: 32.
Optimal Hidden Neurons: 256.
Optimal Epochs: 12.
Early Stopping: Implemented with a patience of 3 epochs.

### Evaluation Metrics
Accuracy: Recorded for both training and testing sets across epochs.
Loss: BCE loss tracked for training and testing sets.

### Tools and Libraries Used
Python: As the primary programming language.
PyTorch: For building and training the neural network.
NumPy and Pandas: For data manipulation and analysis.
Matplotlib: For plotting results.
Scikit-learn: For data preprocessing and splitting.
Scipy: For handling audio files (if applicable).

### Code Structure
The code is structured into several sections:
Data Preparation: Loading and preprocessing the dataset.
Model Definition: Defining the DNN architecture.
Training Loop: Training the model with early stopping.
Evaluation: Assessing model performance on the test set.

### Results
Accuracy and Loss Plots: Visualizing training and test accuracy and loss over epochs to analyze model performance and convergence.

### Below are some of the work done within this component:
#### Part A, Q2: Optimal Batch Size 
This section focuses on determining the optimal batch size for mini-batch gradient descent by training the neural network and evaluating performances for different batch sizes.
Batch Sizes: Explored batch sizes of 32, 64, 128, and 256.
Mean Cross-Validation Accuracies: Plotted against different batch sizes.

#### Part A, Q3: Optimal Number of Hidden Neurons
This section involves finding the optimal number of hidden neurons for the first hidden layer of the network.
Hidden Neurons: Varied the number of neurons in the first hidden layer.
Cross-Validation: Used to evaluate model performance across different configurations.
Mean Cross-Validation Accuracies: Compared across different neuron configurations.

#### Part A, Q4: Local Feature Importance with SHAP Force Plot
This section understanding the feature importance and which features have more impact on the regression task.
SHAP (SHapley Additive exPlanations) to explain individual predictions.

## **Component 2** - Regression Task based on HDB flat prices in Singapore
This project focuses on building and evaluating a deep learning model for predicting HDB resale prices using tabular data. 
The dataset contains both numeric and categorical features, requiring preprocessing steps such as encoding and normalization before training the model.

### The main tasks included:
Preprocessing the dataset to split it into training (years ≤ 2020) and test sets (year 2021).
Implementing a feedforward neural network using the PyTorch Tabular library.
Evaluating the model's performance on test data from years 2022 and 2023 to analyze degradation and identify potential causes.
Detecting feature drift using the Alibi Detect library.

### Tools and Libraries Used
PyTorch Tabular
DataConfig: Defined target variable (resale_price) and specified continuous (dist_to_nearest_stn, floor_area_sqm) and categorical features (month, town, etc.).
TrainerConfig: Configured training parameters, including automatic learning rate tuning, batch size (1024), and maximum epochs (50).
CategoryEmbeddingModelConfig: Created a feedforward neural network with one hidden layer containing 50 neurons.
OptimizerConfig: Used Adam optimizer for training.
TabularModel: Integrated all configurations to initialize and train the model.

### Evaluation Metrics
Calculated RMSE (Root Mean Squared Error) and R² scores to assess the model's predictive performance on test datasets.

### Alibi Detect
Used the TabularDrift function to analyze feature drift between training data (year ≤ 2020) and test data (year 2023). 
This step helps identify changes in data distributions that may cause model degradation.

### Results (Model Performance)
#### Year 2022 Test Data
RMSE: 138,303.56
R² Score: 0.099
Interpretation: The model explained approximately 9.9% of the variance in resale prices, indicating moderate predictive power.

#### Year 2023 Test Data
RMSE: 171,472.95
R² Score: -0.551
Interpretation: A negative R² score suggests that the model performed worse than a simple mean-based prediction, indicating significant degradation.

### Feature Drift Detection
Using Alibi Detect, drift was detected in several features when comparing training data to year 2023 test data. 
The drift indicates changes in feature distributions over time, which likely contributed to the model's degraded performance.

### Observations on Model Degradation
The results demonstrate clear degradation in performance over time:
The R² score dropped significantly from 0.099 (2022) to -0.551 (2023).

#### Possible reasons include:
Covariate Shift: Changes in feature distributions.
Concept Drift: Altered relationships between features and target variable (resale_price).

### Conclusion
This assignment highlights the importance of monitoring data distribution shifts in real-world machine learning applications, 
especially for tabular datasets with temporal components. Future work could involve implementing strategies like retraining models periodically
or using adaptive learning techniques to mitigate degradation caused by feature drift.
