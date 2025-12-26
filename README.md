# Customer Segmentation using PCA & K-Means Clustering

### About The Project

This project implements an end-to-end machine learning pipeline to segment bank customers based on their Demographics and Financial behavior. Unlike traditional clustering approaches, this project utilizes **Principal Component Analysis (PCA)** to handle high-dimensional data and multicollinearity, ensuring more distinct and interpretable customer segments.

The goal is to transform raw customer data into actionable business insights, allowing the marketing team to target specific personas (e.g., "High-Net-Worth Seniors" vs. "Young Spenders") effectively.

### Key Features
* **Unified Analysis**: Integrates demographic and financial data for a 360-degree customer view.
* **Dimensionality Reduction**: Uses PCA to distill noise and complex features into key components.
* **Robust Preprocessing**: Includes outlier handling (Z-Score) and specific encoding strategies for ordinal vs. nominal data.
* **Actionable Profiling**: Translates mathematical clusters into clear business personas.

### Built With
* **Python** (Core Logic)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-learn** (PCA, K-Means, Preprocessing)
* **Seaborn & Matplotlib** (Data Visualization)
* **Jupyter Notebook**

### Dataset
The model is trained on `clustering_data.csv`, containing customer attributes such as:
* **Demographics**: Age, Education, Marital Status, Area.
* **Financials**: Annual Income, Relationship Balance, Product Ownership (Savings, Deposits, Loans).

### Methodology
The project follows a structured Data Science workflow:

1.  **Data Preprocessing**:
    * **Cleaning**: Handling missing values and duplicates.
    * **Outlier Removal**: Using Z-Score to remove extreme values that distort clustering.
    * **Encoding**: Applying **Label Encoding** for ordinal data (e.g., Education Level) and **One-Hot Encoding** for nominal data (e.g., Area).
    * **Scaling**: Standardizing data using `StandardScaler`.

2.  **Dimensionality Reduction (PCA)**:
    * Reducing the dataset dimensions (from ~23 features to 3 principal components).
    * This step addresses the "Curse of Dimensionality" and improves the clustering algorithm's efficiency.

3.  **Model Training**:
    * Applying **K-Means Clustering** on the PCA-transformed data.
    * Determining the optimal number of clusters ($k=3$).

4.  **Evaluation & Profiling**:
    * Validating cluster quality using **Silhouette Score** (~0.39).
    * Analyzing the centroids to define customer personas (e.g., by Income, Age, and Balance).

### Results
The model successfully identified 3 distinct customer segments. By using PCA, the model achieved a **Silhouette Score of ~0.39**, indicating well-separated clusters suitable for real-world business application.

### How to Run
1.  Clone this repository.
2.  Install the required libraries:
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Open `Clustering_Model.ipynb` in Jupyter Notebook.
4.  Run all cells to execute the pipeline from Preprocessing to Visualization.