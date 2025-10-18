"""
Data Exploration for Customer Segmentation Project
Analyzes the Olist e-commerce dataset structure and relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_datasets():
    """Load all Olist datasets"""
    data_path = Path("data")
    
    datasets = {}
    datasets['customers'] = pd.read_csv(data_path / "olist_customers_dataset.csv")
    datasets['orders'] = pd.read_csv(data_path / "olist_orders_dataset.csv")
    datasets['order_items'] = pd.read_csv(data_path / "olist_order_items_dataset.csv")
    datasets['order_payments'] = pd.read_csv(data_path / "olist_order_payments_dataset.csv")
    datasets['products'] = pd.read_csv(data_path / "olist_products_dataset.csv")
    
    return datasets

def explore_dataset_structure(datasets):
    """Explore basic structure of each dataset"""
    print("=" * 60)
    print("DATASET STRUCTURE EXPLORATION")
    print("=" * 60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"Sample data:\n{df.head(3)}")
        print("-" * 40)

def analyze_customer_behavior(datasets):
    """Analyze customer behavior patterns"""
    print("\n" + "=" * 60)
    print("CUSTOMER BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    # Merge datasets to get customer behavior
    df = datasets['customers'].merge(
        datasets['orders'], on='customer_id', how='left'
    ).merge(
        datasets['order_items'], on='order_id', how='left'
    ).merge(
        datasets['order_payments'], on='order_id', how='left'
    )
    
    print(f"Combined dataset shape: {df.shape}")
    
    # Basic customer statistics
    print(f"\nTotal unique customers: {df['customer_unique_id'].nunique()}")
    print(f"Total orders: {df['order_id'].nunique()}")
    print(f"Date range: {df['order_purchase_timestamp'].min()} to {df['order_purchase_timestamp'].max()}")
    
    # Customer order patterns
    customer_stats = df.groupby('customer_unique_id').agg({
        'order_id': 'nunique',
        'price': ['sum', 'mean'],
        'freight_value': 'sum',
        'payment_value': 'sum',
        'order_purchase_timestamp': ['min', 'max']
    }).round(2)
    
    customer_stats.columns = ['total_orders', 'total_spent', 'avg_order_value', 
                             'total_freight', 'total_payment', 'first_order', 'last_order']
    
    print(f"\nCustomer Order Statistics:")
    print(customer_stats.describe())
    
    return df, customer_stats

def create_behavioral_features(df):
    """Create behavioral features for clustering"""
    print("\n" + "=" * 60)
    print("BEHAVIORAL FEATURE ENGINEERING")
    print("=" * 60)
    
    # Convert timestamp columns
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Calculate RFM metrics
    rfm_features = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (df['order_purchase_timestamp'].max() - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'payment_value': 'sum'  # Monetary
    }).rename(columns={
        'order_purchase_timestamp': 'recency_days',
        'order_id': 'frequency',
        'payment_value': 'monetary_value'
    })
    
    # Additional behavioral features
    behavioral_features = df.groupby('customer_unique_id').agg({
        'price': ['sum', 'mean', 'std'],
        'freight_value': ['sum', 'mean'],
        'payment_installments': 'mean',
        'customer_state': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'customer_city': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    })
    
    behavioral_features.columns = ['total_spent', 'avg_order_value', 'order_value_std',
                                 'total_freight', 'avg_freight', 'avg_installments',
                                 'primary_state', 'primary_city']
    
    # Merge features
    features_df = rfm_features.merge(behavioral_features, left_index=True, right_index=True)
    
    # Calculate additional metrics
    features_df['avg_days_between_orders'] = df.groupby('customer_unique_id')['order_purchase_timestamp'].apply(
        lambda x: x.diff().dt.days.mean() if len(x) > 1 else 0
    )
    
    features_df['total_items'] = df.groupby('customer_unique_id')['order_item_id'].count()
    features_df['avg_items_per_order'] = features_df['total_items'] / features_df['frequency']
    
    print(f"Features dataset shape: {features_df.shape}")
    print(f"Features columns: {list(features_df.columns)}")
    print(f"\nFeatures summary:\n{features_df.describe()}")
    
    return features_df

def main():
    """Main exploration function"""
    print("Starting Customer Segmentation Data Exploration...")
    
    # Load datasets
    datasets = load_datasets()
    
    # Explore structure
    explore_dataset_structure(datasets)
    
    # Analyze behavior
    df, customer_stats = analyze_customer_behavior(datasets)
    
    # Create features
    features_df = create_behavioral_features(df)
    
    # Save processed features
    features_df.to_csv("data/customer_features.csv")
    print(f"\nFeatures saved to: data/customer_features.csv")
    
    return datasets, df, features_df

if __name__ == "__main__":
    datasets, df, features_df = main()
