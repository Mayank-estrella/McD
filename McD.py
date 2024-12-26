import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Loads the McDonald's dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

# Load dataset
file_path = 'mcdonalds.csv'
mcdonalds_df = load_data(file_path)
print(mcdonalds_df.info())

def preprocess_data(df):
    """
    Preprocesses the McDonald's dataset:
    - Converts Yes/No columns to binary.
    - Maps categorical columns to numeric scales.
    
    Parameters:
        df (pd.DataFrame): Original dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Convert Yes/No columns to binary (1 for Yes, 0 for No)
    yes_no_columns = df.columns[:11]
    df[yes_no_columns] = df[yes_no_columns].applymap(lambda x: 1 if x == "Yes" else 0)

    # Convert 'Like' to a numeric scale
    like_mapping = {
        "I hate it!-5": -5, "-4": -4, "-3": -3, "-2": -2, "-1": -1, "0": 0,
        "+1": 1, "+2": 2, "+3": 3, "+4": 4, "I love it!+5": 5
    }
    df['Like'] = df['Like'].map(like_mapping)

    # Map 'VisitFrequency' to ordinal scale
    visit_mapping = {
        "Never": 0, "Less than once a month": 1, "Once a month": 2,
        "Every three months": 3, "Once a week": 4, "More than once a week": 5
    }
    df['VisitFrequency'] = df['VisitFrequency'].map(visit_mapping)

    return df

# Preprocess the data
mcdonalds_df = preprocess_data(mcdonalds_df)
print(mcdonalds_df.head())

def perform_eda(df):
    """
    Performs exploratory data analysis (EDA):
    - Displays summary statistics.
    - Generates visualizations for key attributes.
    
    Parameters:
        df (pd.DataFrame): Preprocessed dataset.
    """
    # Summary statistics
    print(df.describe())

    # Visualize the distribution of "Like"
    sns.histplot(df['Like'], bins=11, kde=True)
    plt.title("Distribution of Like Scores")
    plt.xlabel("Like Score")
    plt.ylabel("Frequency")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

# Perform EDA
perform_eda(mcdonalds_df)

def determine_optimal_clusters(features):
    """
    Determines the optimal number of clusters using the elbow method.
    
    Parameters:
        features (pd.DataFrame): Features for clustering.
    
    Returns:
        None
    """
    wcss = []  # Within-cluster sum of squares
    range_clusters = range(2, 9)
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range_clusters, wcss, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.grid()
    plt.show()

# Features for clustering
features = mcdonalds_df.iloc[:, :11]
determine_optimal_clusters(features)

def perform_clustering(features, n_clusters):
    """
    Performs k-means clustering and assigns cluster labels.
    
    Parameters:
        features (pd.DataFrame): Features for clustering.
        n_clusters (int): Optimal number of clusters.
    
    Returns:
        np.ndarray: Cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(features)

# Perform clustering
optimal_clusters = 4
mcdonalds_df['Cluster'] = perform_clustering(features, optimal_clusters)

def visualize_clusters(df):
    """
    Visualizes the distribution of clusters.
    
    Parameters:
        df (pd.DataFrame): Dataset with cluster assignments.
    """
    plt.figure(figsize=(8, 6))
    cluster_counts = df['Cluster'].value_counts(sort=False)
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.show()

# Visualize clusters
visualize_clusters(mcdonalds_df)

def save_results(df, output_file):
    """
    Saves the dataset with cluster assignments to a CSV file.
    
    Parameters:
        df (pd.DataFrame): Dataset with cluster assignments.
        output_file (str): Path to the output CSV file.
    """
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Save results
output_file = 'clustered_mcdonalds.csv'
save_results(mcdonalds_df, output_file)