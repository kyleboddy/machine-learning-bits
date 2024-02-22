# Deep Learning Environment for Data Science

![deeplearninmidjourney](https://github.com/kyleboddy/machine-learning-bits/assets/746351/6b2b1b9b-0fd8-4119-86e3-662db49d831b)

## Summary

This Dockerfile creates a robust, high-performance computing environment tailored for deep learning applications in data science, sports analytics, and research domains. Leveraging the power of NVIDIA CUDA 12.3.1, it provides a seamless experience for GPU-accelerated computing tasks. The environment is designed to support a wide range of deep learning and data processing workflows, making it ideal for analysts, sports scientists, and data scientists looking to push the boundaries of research and experimentation.

### Key Features

- **NVIDIA CUDA Support**: Utilizes NVIDIA's CUDA 12.3.1 base image for GPU-accelerated processing, enabling complex computational tasks at significantly reduced times.
- **Comprehensive Development Tools**: Includes essential development tools and libraries such as Git, Vim, and build-essential, facilitating a versatile development workspace.
- **Advanced Data Analysis with R & Python**: Comes with R and Python 3, along with popular libraries like Pandas, NumPy, SciPy, Matplotlib, and Seaborn, catering to a wide range of data analysis and visualization needs.
- **Machine Learning & Deep Learning Frameworks**: Pre-installed frameworks like TensorFlow, PyTorch, LightGBM, and XGBoost empower users to develop sophisticated machine learning and deep learning models.
- **Hugging Face Transformers and Datasets**: Integrates Hugging Face's Transformers and Datasets libraries, offering access to pre-trained models and a plethora of datasets for natural language processing (NLP) tasks.
- **Database Connectivity**: Features connectors for MySQL and MariaDB, allowing seamless integration with database systems for data ingestion and analysis.
- **Web Development Support**: Includes Node.js and PHP, enabling the development of web applications and services within the same environment.
- **SSH Server Setup**: Configured with an SSH server for secure remote connections, enhancing collaboration and remote development capabilities.

## Domains of Research and Experimentation

This Docker environment is versatile, catering to various domains:

- **Sports Analytics**: Analyze player performance, predict game outcomes, and optimize training regimens using machine learning models.
- **Data Science**: Dive into datasets with powerful data processing and visualization tools to uncover insights and inform decision-making processes.
- **Natural Language Processing (NLP)**: Leverage pre-trained models from the Transformers library for tasks like sentiment analysis, text classification, and language generation.
- **Computer Vision**: Utilize libraries like OpenCV and Dlib for image processing and facial recognition projects, supported by the computational power of CUDA-enabled GPUs.
- **Statistical Modeling**: Employ R and Python for statistical analyses, hypothesis testing, and data exploration in academic and commercial research projects.

This Dockerfile is engineered to be a cornerstone for your deep learning projects, providing a comprehensive toolkit that spans across various domains, enabling you to transform raw data into actionable insights and innovations.

## Sample Programs in Python, R, and PHP for Environment Testing

Below are sample programs demonstrating basic data science tasks in Python, R, and PHP. These examples are designed to test the environment setup and showcase some of the capabilities provided by the installed packages.

### Python Example: Basic Data Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.arange(0., 5., 0.2)
y = np.sin(x)

# Create a simple line plot
plt.plot(x, y, '-o', label='Sin(x)')
plt.title('Simple Line Plot in Python')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()

# Save the figure
plt.savefig('/workspace/python_plot.png')
```

This Python script generates a line plot of the sine function and saves it as 'python_plot.png' in the '/workspace' directory.

### R Example: Data Manipulation with dplyr


``` R
library(dplyr)

# Create a sample data frame
df <- data.frame(
  Name = c('Alice', 'Bob', 'Charlie', 'David', 'Eva'),
  Age = c(25, 30, 35, 40, 45),
  Score = c(85, 90, 88, 95, 80)
)

# Use dplyr to filter and summarize data
result <- df %>%
  filter(Age > 30) %>%
  summarise(AverageScore = mean(Score))

# Print the result
print(result)
```

The R script filters the data frame to include only individuals over 30 years old and calculates the average score among them.

### PHP Example: Simple Data Processing and JSON Encoding

``` php
<?php

// Create an associative array
$data = array(
  "name" => "John Doe",
  "age" => 30,
  "scores" => array(70, 80, 90)
);

// Calculate the average score
$averageScore = array_sum($data["scores"]) / count($data["scores"]);
$data["averageScore"] = $averageScore;

// Encode the data as a JSON string
$jsonData = json_encode($data, JSON_PRETTY_PRINT);

// Print the JSON
echo $jsonData;
?>
```

This PHP script creates an associative array with some data, calculates the average of the scores, and prints the data as a formatted JSON string.

## Python Advanced Example Program

And finally, here's a more advanced Python program to test out with an explanation to follow:

``` Python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import pearsonr

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to Pandas DataFrame for more complex manipulations
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Explore the dataset (simple example: compute Pearson correlation coefficients between features)
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr, _ = pearsonr(df[feature_names[i]], df[feature_names[j]])
        print(f"Pearson Correlation between {feature_names[i]} and {feature_names[j]}: {corr:.3f}")

# Data Preprocessing
## Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Feature Extraction with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Model Evaluation
y_pred = model.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
```

## Advanced Python Program Explanation

### Data Loading
The program begins by loading the Iris dataset using `Scikit-learn`. This dataset includes 150 samples of iris flowers with four features (sepal length, sepal width, petal length, petal width) and a target variable indicating the iris species.

### Data Exploration
It computes and prints the Pearson correlation coefficients between each pair of features using `SciPy`, demonstrating a simple data exploration technique.

### Data Preprocessing
- The dataset is split into training and testing sets using `train_test_split`.
- `StandardScaler` is applied to scale features, which is crucial for many machine learning algorithms.
- Principal Component Analysis (PCA) is used for feature extraction, reducing the dimensionality of the data while retaining most of the variance.

### Model Training
A `RandomForestClassifier` is trained on the PCA-transformed and scaled training data.

### Model Evaluation
The trained model is evaluated on the test set, and the classification report, including precision, recall, and F1-score for each class, is printed.

This program covers various aspects of a typical machine learning workflow, from data loading and preprocessing to model training and evaluation, making it a solid example of using advanced data science tools in Python.
