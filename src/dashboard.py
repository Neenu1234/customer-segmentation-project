"""
Customer Segmentation Dashboard
Interactive Streamlit dashboard for customer segmentation analysis and persona visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .persona-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .cluster-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class CustomerSegmentationDashboard:
    def __init__(self):
        """Initialize the dashboard"""
        self.data_path = Path("data")
        self.features_df = None
        self.cluster_stats = None
        self.personas = None
        self.load_data()
    
    def load_data(self):
        """Load all required data files"""
        try:
            # Load customer features
            if (self.data_path / "customer_features.csv").exists():
                self.features_df = pd.read_csv(self.data_path / "customer_features.csv", index_col=0)
            
            # Load cluster statistics
            cluster_files = list(self.data_path.glob("cluster_analysis_*.csv"))
            if cluster_files:
                self.cluster_stats = pd.read_csv(cluster_files[0], index_col=0)
            
            # Load personas
            if (self.data_path / "customer_personas.json").exists():
                with open(self.data_path / "customer_personas.json", 'r') as f:
                    self.personas = json.load(f)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🎯 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                AI-Powered Customer Insights & Persona Generation
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_overview_metrics(self):
        """Render overview metrics"""
        if self.features_df is None:
            st.warning("Customer features data not available. Please run data exploration first.")
            return
        
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Customers",
                value=f"{len(self.features_df):,}",
                help="Number of unique customers in the dataset"
            )
        
        with col2:
            avg_order_value = self.features_df['avg_order_value'].mean()
            st.metric(
                label="Avg Order Value",
                value=f"${avg_order_value:.2f}",
                help="Average order value across all customers"
            )
        
        with col3:
            total_revenue = self.features_df['monetary_value'].sum()
            st.metric(
                label="Total Revenue",
                value=f"${total_revenue:,.0f}",
                help="Total revenue from all customers"
            )
        
        with col4:
            avg_frequency = self.features_df['frequency'].mean()
            st.metric(
                label="Avg Orders/Customer",
                value=f"{avg_frequency:.1f}",
                help="Average number of orders per customer"
            )
    
    def render_customer_distribution(self):
        """Render customer distribution charts"""
        if self.features_df is None:
            return
        
        st.subheader("📈 Customer Behavior Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Distribution
            fig_rfm = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Recency (Days)', 'Frequency (Orders)', 'Monetary Value ($)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_rfm.add_trace(
                go.Histogram(x=self.features_df['recency_days'], name='Recency', nbinsx=30),
                row=1, col=1
            )
            fig_rfm.add_trace(
                go.Histogram(x=self.features_df['frequency'], name='Frequency', nbinsx=20),
                row=1, col=2
            )
            fig_rfm.add_trace(
                go.Histogram(x=self.features_df['monetary_value'], name='Monetary', nbinsx=30),
                row=1, col=3
            )
            
            fig_rfm.update_layout(height=400, showlegend=False, title_text="RFM Analysis Distribution")
            st.plotly_chart(fig_rfm, use_container_width=True)
        
        with col2:
            # Order Value vs Frequency scatter
            fig_scatter = px.scatter(
                self.features_df.sample(min(5000, len(self.features_df))),  # Sample for performance
                x='frequency',
                y='avg_order_value',
                size='monetary_value',
                color='recency_days',
                title="Customer Behavior Patterns",
                labels={
                    'frequency': 'Order Frequency',
                    'avg_order_value': 'Average Order Value ($)',
                    'recency_days': 'Days Since Last Purchase'
                },
                color_continuous_scale='viridis'
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def render_clustering_results(self):
        """Render clustering analysis results"""
        if self.cluster_stats is None:
            st.warning("Cluster analysis results not available. Please run clustering analysis first.")
            return
        
        st.subheader("🎯 Clustering Results")
        
        # Cluster sizes
        cluster_sizes = self.cluster_stats.groupby(self.cluster_stats.index).size()
        if len(cluster_sizes) == 0:
            cluster_sizes = pd.Series([len(self.cluster_stats)] * len(self.cluster_stats), index=self.cluster_stats.index)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster size pie chart
            fig_pie = px.pie(
                values=cluster_sizes.values,
                names=[f"Cluster {i}" for i in cluster_sizes.index],
                title="Customer Distribution by Cluster",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Cluster characteristics radar chart
            if len(self.cluster_stats) > 0:
                # Select first cluster for radar chart
                cluster_id = self.cluster_stats.index[0]
                cluster_data = self.cluster_stats.loc[cluster_id]
                
                categories = ['Recency', 'Frequency', 'Monetary', 'Avg Order Value', 'Total Items']
                values = [
                    cluster_data.get('recency_days_mean', 0),
                    cluster_data.get('frequency_mean', 0),
                    cluster_data.get('monetary_value_mean', 0),
                    cluster_data.get('avg_order_value_mean', 0),
                    cluster_data.get('total_items_mean', 0)
                ]
                
                # Normalize values for radar chart
                max_values = [self.features_df['recency_days'].max(), 
                             self.features_df['frequency'].max(),
                             self.features_df['monetary_value'].max(),
                             self.features_df['avg_order_value'].max(),
                             self.features_df['total_items'].max()]
                
                normalized_values = [v/max_v if max_v > 0 else 0 for v, max_v in zip(values, max_values)]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=categories,
                    fill='toself',
                    name=f'Cluster {cluster_id}',
                    line_color='blue'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title=f"Cluster {cluster_id} Characteristics"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
    
    def render_cluster_comparison(self):
        """Render cluster comparison table"""
        if self.cluster_stats is None:
            return
        
        st.subheader("📋 Cluster Comparison")
        
        # Create comparison table
        comparison_cols = [
            'recency_days_mean', 'frequency_mean', 'monetary_value_mean',
            'avg_order_value_mean', 'total_items_mean', 'avg_items_per_order_mean'
        ]
        
        comparison_df = self.cluster_stats[comparison_cols].round(2)
        comparison_df.columns = [
            'Recency (Days)', 'Frequency', 'Monetary Value ($)',
            'Avg Order Value ($)', 'Total Items', 'Avg Items/Order'
        ]
        
        st.dataframe(comparison_df, use_container_width=True)
    
    def render_personas(self):
        """Render customer personas"""
        if self.personas is None:
            st.warning("Customer personas not available. Please run persona generation first.")
            return
        
        st.subheader("👥 Customer Personas")
        
        # Persona selector
        persona_options = {f"Cluster {k}": v for k, v in self.personas.items()}
        selected_persona_name = st.selectbox(
            "Select a persona to view:",
            options=list(persona_options.keys()),
            help="Choose a customer segment to view detailed persona information"
        )
        
        if selected_persona_name:
            persona = persona_options[selected_persona_name]
            
            # Display persona
            st.markdown(f'<div class="persona-card">', unsafe_allow_html=True)
            
            # Persona header
            st.markdown(f'<h3 class="cluster-header">{persona["persona_name"]}</h3>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cluster Size", f"{persona['cluster_size']:,}")
            with col2:
                st.metric("Percentage", f"{persona['cluster_percentage']:.1f}%")
            with col3:
                st.metric("Avg Order Value", f"${persona['key_metrics']['avg_order_value']:.2f}")
            with col4:
                st.metric("Total Revenue", f"${persona['key_metrics']['monetary_value']:.2f}")
            
            # Persona details
            if persona.get('demographics'):
                st.markdown("**Demographics:**")
                st.write(persona['demographics'])
            
            if persona.get('behavioral_patterns'):
                st.markdown("**Behavioral Patterns:**")
                st.write(persona['behavioral_patterns'])
            
            if persona.get('psychographics'):
                st.markdown("**Psychographics:**")
                st.write(persona['psychographics'])
            
            if persona.get('marketing_recommendations'):
                st.markdown("**Marketing Recommendations:**")
                st.write(persona['marketing_recommendations'])
            
            if persona.get('business_value'):
                st.markdown("**Business Value:**")
                st.write(persona['business_value'])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with navigation and info"""
        st.sidebar.title("🎯 Navigation")
        
        st.sidebar.markdown("### Quick Stats")
        if self.features_df is not None:
            st.sidebar.metric("Total Customers", f"{len(self.features_df):,}")
            st.sidebar.metric("Avg Order Value", f"${self.features_df['avg_order_value'].mean():.2f}")
        
        if self.cluster_stats is not None:
            st.sidebar.metric("Clusters Found", len(self.cluster_stats))
        
        if self.personas is not None:
            st.sidebar.metric("Personas Generated", len(self.personas))
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.info("""
        This dashboard showcases customer segmentation analysis using:
        - **K-means clustering** for customer grouping
        - **RFM analysis** for behavioral insights
        - **OpenAI GPT** for persona generation
        - **Interactive visualizations** for insights
        """)
        
        st.sidebar.markdown("### Data Sources")
        st.sidebar.text("• Olist E-commerce Dataset")
        st.sidebar.text("• Customer transaction data")
        st.sidebar.text("• Behavioral features")
        st.sidebar.text("• AI-generated personas")
    
    def run(self):
        """Run the complete dashboard"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Clustering", "📋 Comparison", "👥 Personas"])
        
        with tab1:
            self.render_overview_metrics()
            st.markdown("---")
            self.render_customer_distribution()
        
        with tab2:
            self.render_clustering_results()
        
        with tab3:
            self.render_cluster_comparison()
        
        with tab4:
            self.render_personas()

def main():
    """Main function to run the dashboard"""
    dashboard = CustomerSegmentationDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
