"""
Customer Segmentation Clustering Analysis
Implements K-means, DBSCAN, and other clustering algorithms with evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self, data_path="data/customer_features.csv"):
        """Initialize the customer segmentation class"""
        self.data_path = data_path
        self.df = None
        self.features_df = None
        self.scaled_features = None
        self.scaler = StandardScaler()
        self.clusters = {}
        self.evaluation_metrics = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the customer features data"""
        print("Loading and preprocessing customer features...")
        
        # Load the features data
        self.features_df = pd.read_csv(self.data_path, index_col=0)
        
        # Handle missing values
        print(f"Missing values before preprocessing: {self.features_df.isnull().sum().sum()}")
        
        # Fill missing values
        numeric_columns = self.features_df.select_dtypes(include=[np.number]).columns
        self.features_df[numeric_columns] = self.features_df[numeric_columns].fillna(
            self.features_df[numeric_columns].median()
        )
        
        # Handle categorical variables
        categorical_columns = self.features_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.features_df[col] = self.features_df[col].fillna('Unknown')
        
        print(f"Missing values after preprocessing: {self.features_df.isnull().sum().sum()}")
        
        # Select features for clustering (exclude categorical for now)
        clustering_features = [
            'recency_days', 'frequency', 'monetary_value', 'total_spent',
            'avg_order_value', 'order_value_std', 'total_freight', 'avg_freight',
            'avg_installments', 'avg_days_between_orders', 'total_items',
            'avg_items_per_order'
        ]
        
        self.df = self.features_df[clustering_features].copy()
        
        # Handle outliers using IQR method
        for col in self.df.columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        print(f"Preprocessed data shape: {self.df.shape}")
        return self.df
    
    def scale_features(self):
        """Scale features for clustering"""
        print("Scaling features...")
        self.scaled_features = self.scaler.fit_transform(self.df)
        print(f"Scaled features shape: {self.scaled_features.shape}")
        return self.scaled_features
    
    def find_optimal_k(self, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        print("Finding optimal number of clusters...")
        
        if self.scaled_features is None:
            self.scale_features()
        
        # Calculate metrics for different k values
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.scaled_features, cluster_labels))
        
        # Find optimal k based on silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        # Plot the results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Elbow method
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        # Silhouette scores
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True)
        
        # Davies-Bouldin scores
        axes[2].plot(k_range, davies_bouldin_scores, 'go-')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Davies-Bouldin Score')
        axes[2].set_title('Davies-Bouldin Analysis')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_k, {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores
        }
    
    def perform_kmeans_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_k()
        
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.scaled_features)
        
        # Store results
        self.clusters['kmeans'] = {
            'labels': cluster_labels,
            'model': kmeans,
            'n_clusters': n_clusters
        }
        
        # Calculate evaluation metrics
        silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
        davies_bouldin = davies_bouldin_score(self.scaled_features, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.scaled_features, cluster_labels)
        
        self.evaluation_metrics['kmeans'] = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz
        }
        
        print(f"K-means Results:")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.3f}")
        
        return cluster_labels
    
    def perform_dbscan_clustering(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering"""
        print(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(self.scaled_features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"DBSCAN Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        
        if n_clusters > 1:
            silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
            davies_bouldin = davies_bouldin_score(self.scaled_features, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(self.scaled_features, cluster_labels)
            
            self.evaluation_metrics['dbscan'] = {
                'silhouette_score': silhouette_avg,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
            
            print(f"  Silhouette Score: {silhouette_avg:.3f}")
            print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
            print(f"  Calinski-Harabasz Score: {calinski_harabasz:.3f}")
        
        # Store results
        self.clusters['dbscan'] = {
            'labels': cluster_labels,
            'model': dbscan,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
        return cluster_labels
    
    def perform_hierarchical_clustering(self, n_clusters=None):
        """Perform hierarchical clustering"""
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_k()
        
        print(f"Performing hierarchical clustering with {n_clusters} clusters...")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = hierarchical.fit_predict(self.scaled_features)
        
        # Store results
        self.clusters['hierarchical'] = {
            'labels': cluster_labels,
            'model': hierarchical,
            'n_clusters': n_clusters
        }
        
        # Calculate evaluation metrics
        silhouette_avg = silhouette_score(self.scaled_features, cluster_labels)
        davies_bouldin = davies_bouldin_score(self.scaled_features, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.scaled_features, cluster_labels)
        
        self.evaluation_metrics['hierarchical'] = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz
        }
        
        print(f"Hierarchical Results:")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.3f}")
        
        return cluster_labels
    
    def reduce_dimensions(self, method='pca', n_components=2):
        """Reduce dimensions for visualization"""
        print(f"Reducing dimensions using {method.upper()}...")
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'pca', 'tsne', or 'umap'")
        
        reduced_features = reducer.fit_transform(self.scaled_features)
        
        print(f"Reduced features shape: {reduced_features.shape}")
        return reduced_features, reducer
    
    def visualize_clusters(self, method='kmeans', dim_reduction='pca'):
        """Visualize clusters in 2D"""
        if method not in self.clusters:
            print(f"No clustering results found for {method}")
            return
        
        print(f"Visualizing {method} clusters using {dim_reduction.upper()}...")
        
        # Reduce dimensions
        reduced_features, reducer = self.reduce_dimensions(method=dim_reduction)
        
        # Get cluster labels
        cluster_labels = self.clusters[method]['labels']
        
        # Create visualization
        fig = px.scatter(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            color=cluster_labels,
            title=f'Customer Segments - {method.upper()} Clustering ({dim_reduction.upper()})',
            labels={'x': f'{dim_reduction.upper()} Component 1', 'y': f'{dim_reduction.upper()} Component 2'},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        fig.show()
        
        # Save the plot
        fig.write_html(f'plots/{method}_clusters_{dim_reduction}.html')
        
        return fig
    
    def analyze_cluster_characteristics(self, method='kmeans'):
        """Analyze characteristics of each cluster"""
        if method not in self.clusters:
            print(f"No clustering results found for {method}")
            return
        
        print(f"Analyzing cluster characteristics for {method}...")
        
        cluster_labels = self.clusters[method]['labels']
        
        # Add cluster labels to the dataframe
        analysis_df = self.df.copy()
        analysis_df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = analysis_df.groupby('cluster').agg({
            'recency_days': ['mean', 'std'],
            'frequency': ['mean', 'std'],
            'monetary_value': ['mean', 'std'],
            'total_spent': ['mean', 'std'],
            'avg_order_value': ['mean', 'std'],
            'total_items': ['mean', 'std'],
            'avg_items_per_order': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
        
        # Calculate cluster sizes
        cluster_sizes = analysis_df['cluster'].value_counts().sort_index()
        
        print(f"\nCluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(analysis_df)) * 100
            print(f"  Cluster {cluster_id}: {size} customers ({percentage:.1f}%)")
        
        print(f"\nCluster Statistics:")
        print(cluster_stats)
        
        # Save cluster analysis
        cluster_stats.to_csv(f'data/cluster_analysis_{method}.csv')
        
        return cluster_stats, cluster_sizes
    
    def compare_clustering_methods(self):
        """Compare different clustering methods"""
        print("Comparing clustering methods...")
        
        methods = list(self.evaluation_metrics.keys())
        
        comparison_df = pd.DataFrame(self.evaluation_metrics).T
        
        print("\nClustering Methods Comparison:")
        print(comparison_df.round(3))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
        titles = ['Silhouette Score (Higher is Better)', 'Davies-Bouldin Score (Lower is Better)', 'Calinski-Harabasz Score (Higher is Better)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            comparison_df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(title)
            axes[i].set_xlabel('Clustering Method')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/clustering_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def run_complete_analysis(self):
        """Run complete clustering analysis"""
        print("=" * 60)
        print("CUSTOMER SEGMENTATION ANALYSIS")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Scale features
        self.scale_features()
        
        # Find optimal k
        optimal_k, k_analysis = self.find_optimal_k()
        
        # Perform different clustering methods
        self.perform_kmeans_clustering(n_clusters=optimal_k)
        self.perform_dbscan_clustering()
        self.perform_hierarchical_clustering(n_clusters=optimal_k)
        
        # Compare methods
        comparison_df = self.compare_clustering_methods()
        
        # Analyze best method (highest silhouette score)
        best_method = comparison_df['silhouette_score'].idxmax()
        print(f"\nBest clustering method: {best_method}")
        
        # Visualize clusters
        self.visualize_clusters(method=best_method, dim_reduction='pca')
        self.visualize_clusters(method=best_method, dim_reduction='umap')
        
        # Analyze cluster characteristics
        cluster_stats, cluster_sizes = self.analyze_cluster_characteristics(method=best_method)
        
        return {
            'best_method': best_method,
            'optimal_k': optimal_k,
            'comparison_df': comparison_df,
            'cluster_stats': cluster_stats,
            'cluster_sizes': cluster_sizes,
            'k_analysis': k_analysis
        }

def main():
    """Main function to run customer segmentation analysis"""
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Initialize and run analysis
    segmentation = CustomerSegmentation()
    results = segmentation.run_complete_analysis()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Best method: {results['best_method']}")
    print(f"Optimal clusters: {results['optimal_k']}")
    print("Results saved to:")
    print("  - plots/ directory for visualizations")
    print("  - data/cluster_analysis_*.csv for cluster statistics")
    
    return segmentation, results

if __name__ == "__main__":
    segmentation, results = main()
