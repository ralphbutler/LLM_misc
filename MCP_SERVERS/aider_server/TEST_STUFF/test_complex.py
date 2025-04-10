
# this cost 13 cents with plain claude
# Simpler Change: Convert the program from using file-based document loading to supporting both file-based and string-based inputs.
# This would require:
#    - Modifying the constructor to accept an optional list of text strings
#    - Updating the load_documents method to handle both cases
#    - Adjusting the document naming logic
#    - Updating relevant documentation strings
#    - Modifying the argument parser to support the new functionality

# Change the program to use a more flexible dimensionality reduction approach by replacing the fixed PCA implementation with a configurable dimensionality reduction system that can switch between PCA, t-SNE, and UMAP based on user input.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class ScientificTextAnalyzer:
    """A class for analyzing scientific text documents using NLP techniques."""

    def __init__(self, input_source=None, documents_list=None, num_clusters=5):
        """Initialize the analyzer with data path or document list and clustering parameters.

        Args:
            input_source (str, optional): Path to directory containing text files. Defaults to None.
            documents_list (list[str], optional): List of text strings. Defaults to None.
            num_clusters (int): Number of clusters for KMeans clustering.

        Raises:
            ValueError: If neither input_source nor documents_list is provided, or if both are provided.
        """
        if not input_source and not documents_list:
            raise ValueError("Either input_source (file path) or documents_list must be provided.")
        if input_source and documents_list:
            raise ValueError("Provide either input_source (file path) or documents_list, not both.")

        self.input_source = input_source
        self.documents_list_input = documents_list # Store the input list if provided
        self.num_clusters = num_clusters
        self.documents = []
        self.document_names = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_matrix = None
        self.pca = PCA(n_components=2)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.cluster_labels = None

    def load_documents(self):
        """Load text documents from the specified source (directory or list)."""
        if self.documents_list_input:
            print("Loading documents from input list...")
            self.documents = self.documents_list_input
            self.document_names = [f"doc_{i}" for i in range(len(self.documents))]
            print(f"Loaded {len(self.documents)} documents from list.")
        elif self.input_source and os.path.isdir(self.input_source):
            print(f"Loading documents from directory: {self.input_source}...")
            for filename in os.listdir(self.input_source):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.input_source, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            self.documents.append(content)
                            self.document_names.append(filename)
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}")
            print(f"Loaded {len(self.documents)} documents from directory.")
        else:
            print(f"Error: Input source '{self.input_source}' is not a valid directory or no input list provided.")
            # Or raise an error, depending on desired behavior
            return # Exit loading if source is invalid

    def vectorize_text(self):
        """Convert text documents to TF-IDF feature vectors."""
        print("Vectorizing documents...")
        self.feature_matrix = self.vectorizer.fit_transform(self.documents)
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
    def perform_clustering(self):
        """Perform KMeans clustering on the document vectors."""
        print(f"Performing KMeans clustering with {self.num_clusters} clusters...")
        self.cluster_labels = self.kmeans.fit_predict(self.feature_matrix)
        
    def reduce_dimensions(self):
        """Reduce dimensions for visualization using PCA."""
        print("Reducing dimensions with PCA...")
        reduced_features = self.pca.fit_transform(self.feature_matrix.toarray())
        return reduced_features
        
    def visualize_clusters(self):
        """Visualize document clusters in 2D space."""
        reduced_features = self.reduce_dimensions()
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_features[:, 0], 
            reduced_features[:, 1], 
            c=self.cluster_labels, 
            cmap='viridis', 
            alpha=0.7
        )
        
        # Add document names as labels
        for i, name in enumerate(self.document_names):
            plt.annotate(name, (reduced_features[i, 0], reduced_features[i, 1]), fontsize=8)
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Document Clusters Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.tight_layout()
        plt.savefig('document_clusters.png')
        plt.show()
        
    def analyze(self):
        """Run the complete analysis pipeline."""
        self.load_documents()
        self.vectorize_text()
        self.perform_clustering()
        self.visualize_clusters()
        
        # Create a DataFrame with clustering results
        results = pd.DataFrame({
            'Document': self.document_names,
            'Cluster': self.cluster_labels
        })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Analyze scientific text documents from files or strings.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data_path', type=str, help='Path to directory with text files (.txt)')
    input_group.add_argument('--input_strings', type=str, nargs='+', help='List of text strings to analyze')

    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters for KMeans')

    args = parser.parse_args()

    analyzer = None
    if args.data_path:
        analyzer = ScientificTextAnalyzer(input_source=args.data_path, num_clusters=args.num_clusters)
    elif args.input_strings:
        analyzer = ScientificTextAnalyzer(documents_list=args.input_strings, num_clusters=args.num_clusters)

    if analyzer:
        results = analyzer.analyze()
        print("\nClustering Results:")
    else:
        # This case should ideally not be reached due to the mutually exclusive group and required=True
        # but adding a safeguard.
        print("Error: Could not initialize analyzer. Check input arguments.")
        return
    print(results)
    
    
if __name__ == "__main__":
    main()
